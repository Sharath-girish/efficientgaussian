
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import yaml
import torch
import shutil
import torchvision
import numpy as np
import subprocess as sp
from os import makedirs
from random import Random
from PIL import Image
from utils.loss_utils import l1_loss, ssim, scaled_l1_loss
from gaussian_renderer import render, network_gui
from lpipsPyTorch import lpips
import torch.utils.benchmark as benchmark
from collections import defaultdict
import torchvision.transforms.functional as tf
from pathlib import Path
import sys
import socket
from scene import Scene, GaussianModelSQ
from compress.decoders import LatentDecoder
from compress.inf_loss import EntropyLoss
from utils.general_utils import safe_state, sample_camera_order, mean_distances
import uuid
import hashlib
from collections import OrderedDict
from tqdm import tqdm
from utils.general_utils import DecayScheduler
from utils.image_utils import psnr, resize_image, downsample_image, blur_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return memory_used_values


def training(seed, dataset, opt, pipe, quantize, saving_iterations, checkpoint_iterations, checkpoint, debug_from, parse_args):
    first_iter = 0
    generator = Random(0)
    wandb_enabled = WANDB_FOUND and parse_args.use_wandb
    quantize.use_shift = [bool(el) for el in quantize.use_shift]
    quantize.use_gumbel = [bool(el) for el in quantize.use_gumbel]
    quantize.gumbel_period = [bool(el) for el in quantize.gumbel_period]

    gaussians = GaussianModelSQ(dataset.sh_degree, quantize)
    distortion_loss = EntropyLoss(gaussians.prob_models, lambdas=gaussians.ent_lambdas, noise_freq=quantize.noise_freq)
    resize_scale_sched = DecayScheduler(
                                        total_steps=int(opt.resize_period*(opt.iterations+1)),
                                        decay_name='cosine',
                                        start=opt.resize_scale,
                                        end=1.0,
                                        )
    temperature_scheds = OrderedDict()
    for i,param in enumerate(gaussians.param_names):
        temperature_scheds[param] = DecayScheduler(
                                        total_steps=opt.iterations+1,
                                        decay_name='exp',
                                        start=1.0,
                                        end=quantize.temperature[i],
                                        params={'temperature': quantize.temperature[i], 'decay_period': quantize.gumbel_period[i]},
                                        )

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    cam_centers = torch.zeros(opt.iterations,3)
    if checkpoint and os.path.exists(checkpoint):
        (model_params, first_iter, cam_centers) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print("Restored model from checkpoint {}, starting from iteration {}".format(checkpoint, first_iter))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    if opt.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    cur_size, cur_psnr = 0, 0
    cam_mean_distances = mean_distances(scene.getTrainCameras(), generator)
    cam_scales = cam_mean_distances / cam_mean_distances.mean()
    cam_scales = torch.pow(cam_scales, 4)/torch.pow(cam_scales, 4).mean()

    net_training_time = 0

    best_state_dict = None
    best_train_psnr = 0.0
    best_iter = 0
    for iteration in range(first_iter, opt.iterations + 1): 
        # if iteration==(opt.iterations//2):
        #     gaussians.to_hc()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration, quantize)
        for i,param in enumerate(gaussians.param_names):
            gaussians.latent_decoders[param].temperature = temperature_scheds[param](iteration)
            gaussians.latent_decoders[param].use_gumbel = ((iteration / opt.iterations) < quantize.gumbel_period[i]) and quantize.use_gumbel[i]

        if (iteration-1) % 10 == 0:
            for i,param in enumerate(gaussians.param_names):
                if isinstance(gaussians.latent_decoders[param], LatentDecoder):
                    gaussians.latent_decoders[param].normalize(gaussians._latents[param])

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        cam_idx = generator.randint(0, len(viewpoint_stack)-1)
        dist_scale = cam_scales[cam_idx]
        viewpoint_cam = viewpoint_stack.pop(cam_idx)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if opt.transform == "resize":
            gt_image = resize_image(gt_image, resize_scale_sched(iteration))
        elif "blur" in opt.transform and resize_scale_sched(iteration)!=1.0:
            if (iteration-1) % 100 == 0:
                transform = blur_image(resize_scale_sched(iteration), opt.transform)
            gt_image = transform(gt_image)
        elif opt.transform == "downsample":
            gt_image = downsample_image(gt_image, resize_scale_sched(iteration))

        cam_centers[iteration-1] = viewpoint_cam.camera_center

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.use_amp):
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, image_shape=gt_image.shape,
                                get_infl=iteration<opt.prune_until_iter)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            entropy_loss, _ = distortion_loss.loss(gaussians._latents, iteration, is_val=(quantize.noise_freq == 0))
            loss += entropy_loss

        if opt.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # Log and save
            iter_time = iter_start.elapsed_time(iter_end)
            net_training_time += iter_time
            psnr_train, psnr_test = training_report(tb_writer, wandb_enabled, dataset.wandb_log_images, iteration, Ll1, loss, 
                                                    l1_loss, cur_size, iter_time, net_training_time, dataset.testing_interval, 
                                                    opt.iterations-opt.search_best_iters, scene, render, (pipe, background))

            if psnr_test:
                cur_psnr = psnr_test

            if (iteration -1)% dataset.log_interval == 0 or iteration == opt.iterations:
                cur_size = gaussians.size()/8/(10**6)
                log_dict = {
                            "Loss": f"{ema_loss_for_log:.{5}f}",
                            "Num points": f"{gaussians._xyz.shape[0]}",
                            "Size (MB)": f"{cur_size:.{2}f}",
                            "Resize": f"{resize_scale_sched(iteration):.{2}f}",
                            "PSNR": f"{cur_psnr:.{2}f}",
                            }
                progress_bar.set_postfix(log_dict)
                progress_bar.update(dataset.log_interval)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # scene.save(iteration)
                scene.save_compressed(iteration, quantize)

            if iteration < opt.prune_until_iter:
                gaussians.add_influence_stats(render_pkg["influence"])

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if iteration <= opt.densify_from_iter or \
                    (iteration > opt.densify_from_iter and ((iteration-opt.densify_from_iter) % opt.densification_interval >= opt.accumulate_fraction*(opt.densification_interval-1))): 
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % len(scene.train_cameras[1.0]) == 0:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # if iteration<1000:
                    #     print('\n[ITER {}] Densifying'.format(iteration))
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % opt.infl_prune_interval == 0 and iteration<opt.prune_until_iter:
                gaussians.prune_influence(quantile_threshold=opt.quantile_threshold)

            # Optimizer step
            if iteration < opt.iterations:
                if opt.use_amp:
                    scaler.step(gaussians.optimizer)
                    scaler.update()
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Used for intermediate checkpointing and resuming only
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration, cam_centers), 
                           os.path.join(scene.model_path,"chkpnt" + str(iteration) + ".pth"))

            if (dataset.checkpoint_interval > 0 and iteration % dataset.checkpoint_interval == 0):
                torch.save((gaussians.capture(), iteration, cam_centers), os.path.join(scene.model_path, "resume_ckpt.pth"))

            if psnr_train and psnr_train > best_train_psnr:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Save attributes in uncompressed point cloud .ply format (for visualization)
                if dataset.save_ply:
                    scene.save_best()
                # Save attributes in compress pkl format
                scene.save_best_compressed(quantize)
                best_iter = iteration
                # best_state_dict = gaussians.capture_best_state()

    if dataset.save_ply:
        scene.link_best(best_iter)
    scene.link_best_compressed(best_iter)
    # gaussians.restore_best_state(best_state_dict, opt)
    if os.path.exists(os.path.join(scene.model_path, "resume_ckpt.pth")):
        os.remove(os.path.join(scene.model_path, "resume_ckpt.pth"))

    if wandb_enabled:
        wandb.run.summary['training_time'] = net_training_time/1000

    return net_training_time/1000, best_iter

def prepare_output_and_logger(args, all_args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    # Create wandb logger
    if WANDB_FOUND and args.use_wandb:
        wandb_project = args.wandb_project
        wandb_run_name = args.wandb_run_name
        wandb_entity = args.wandb_entity
        wandb_mode = args.wandb_mode
        id = hashlib.md5(wandb_run_name.encode('utf-8')).hexdigest()
        # name = os.path.basename(args.model_path) if wandb_run_name is None else wandb_run_name
        name = os.path.basename(args.source_path)+'_'+str(id)
        wandb.init(
            project=wandb_project,
            name=name,
            entity=wandb_entity,
            config=all_args,
            sync_tensorboard=False,
            dir=args.model_path,
            mode=wandb_mode,
            id=id,
            resume=True
        )
    return tb_writer

def training_report(tb_writer, wandb_enabled, wandb_log_images, iteration, Ll1, loss, l1_loss, size, 
                    iter_time, elapsed, testing_interval, search_best, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', iter_time, iteration)
        tb_writer.add_scalar('elapsed', elapsed, iteration)

    if wandb_enabled:
        wandb.log({"train_loss_patches/l1_loss": Ll1.item(), 
                   "train_loss_patches/total_loss": loss.item(), 
                   "num_points": scene.gaussians.get_xyz.shape[0],
                   "iter_time": iter_time,
                   "elapsed": elapsed,
                   "size": size
                   }, step=iteration)

    # Report test and samples of training set
    if iteration % testing_interval ==0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)
        if iteration>=search_best:
            validation_configs = ({'name': 'train', 'cameras' : scene.getTrainCameras()},
                                    {'name': 'test', 'cameras' : scene.getTestCameras()},)
        psnr_configs = {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    _, viewpoint.image_height, viewpoint.image_width = gt_image.shape
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_interval:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        if wandb_enabled and wandb_log_images:
                            if iteration == testing_interval:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): 
                                           wandb.Image(gt_image[None].detach().cpu().numpy(), caption="ground_truth")}, 
                                           step=iteration),
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): 
                                       wandb.Image(image[None].detach().cpu().numpy(), caption="render")}, 
                                       step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                psnr_configs[config['name']] = psnr_test
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if wandb_enabled:
                    wandb.log({config['name'] + '/loss_viewpoint/l1_loss': l1_test, config['name'] + '/loss_viewpoint/psnr': psnr_test}, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)


        if wandb_enabled:
            try:
                wandb.log({"scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.detach().cpu().numpy()), "total_points": scene.gaussians.get_xyz.shape[0]}, step=iteration)
            except Exception as e:
                print("total points ", scene.gaussians.get_xyz.shape[0])
                print("opacity min max", scene.gaussians.get_opacity.detach().cpu().numpy().min(), scene.gaussians.get_opacity.detach().cpu().numpy().max())
                raise e
            
        torch.cuda.empty_cache()
        return psnr_configs['train'] if 'train' in psnr_configs else None, psnr_configs['test']
    return None, None
        

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save_images, use_amp):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if save_images:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

    preds, gts, names = [], [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        if save_images:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        else:
            preds.append(rendering)
            gts.append(gt)
            names.append('{0:05d}'.format(idx))

    return preds, gts, names

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, quantize: QuantizeParams, 
                skip_train : bool, skip_test : bool, wandb_enabled: bool, use_amp: bool):
    with torch.no_grad():

        quantize.use_shift = [bool(el) for el in quantize.use_shift]
        quantize.use_gumbel = [bool(el) for el in quantize.use_gumbel]
        quantize.gumbel_period = [bool(el) for el in quantize.gumbel_period]
        gaussians = GaussianModelSQ(dataset.sh_degree, quantize)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_ply=True)
        scene.gaussians.decode_latents()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        images = {}
        if not skip_train:
            train_preds, train_gts, train_names = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                                                 gaussians, pipeline, background, dataset.save_images, use_amp)
            if train_preds:
                images["train"] = (train_preds, train_gts, train_names)
            else:
                images["train"] = None

        if not skip_test:
            test_preds, test_gts, test_names = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                                               gaussians, pipeline, background, dataset.save_images, use_amp)
            if test_preds:
                images["test"] = (test_preds, test_gts, test_names)
            else:
                images["test"] = None
    
        if wandb_enabled:
            wandb.log({"rendering_mem": get_gpu_memory()[0]}, step=1)
        fps = measure_fps(scene, gaussians, pipeline, background, use_amp)
        if wandb_enabled:
            wandb.log({"rendering_mem": get_gpu_memory()[0]}, step=2)

        if wandb_enabled:
            wandb.run.summary['FPS'] = fps
    return images, fps, scene.loaded_iter

def render_fn(views, gaussians, pipeline, background, use_amp):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        for view in views:
            render(view, gaussians, pipeline, background)

def measure_fps(scene, gaussians, pipeline, background, use_amp):
    with torch.no_grad():
        views = scene.getTrainCameras() + scene.getTestCameras()
        t0 = benchmark.Timer(stmt='render_fn(views, gaussians, pipeline, background, use_amp)',
                            setup='from __main__ import render_fn',
                            globals={'views': views, 'gaussians': gaussians, 'pipeline': pipeline, 
                                     'background': background, 'use_amp': use_amp},
                            )
        time = t0.timeit(50)
        fps = len(views)/time.median
        print("Rendering FPS: ", fps)
    return fps
        


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render)[:3, :, :].cuda())
        gts.append(tf.to_tensor(gt)[:3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(images, scene_dir, iteration, wandb_enabled=False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test" / "ours_{}".format(iteration)

        gt_dir = test_dir/ "gt"
        renders_dir = test_dir / "renders"
        if images["test"]:
            renders, gts, image_names = images["test"]
        else:
            renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            render, gt = renders[idx].unsqueeze(0)[:, :3, :, :].cuda(), gts[idx].unsqueeze(0)[:, :3, :, :].cuda()
            ssims.append(ssim(render, gt))
            psnrs.append(psnr(render, gt))
            lpipss.append(lpips(render, gt, net_type='vgg'))

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        if wandb_enabled:
            wandb.run.summary['PSNR'] = torch.tensor(psnrs).mean()
            wandb.run.summary['SSIM'] = torch.tensor(ssims).mean()
            wandb.run.summary['LPIPS'] = torch.tensor(lpipss).mean()

        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
    except Exception as e:
        raise e
        print("Unable to compute metrics for model", scene_dir)

    return full_dict

if __name__ == "__main__":

    # Config file is used for argument defaults. Command line arguments override config file.
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, config['model_params'])
    op = OptimizationParams(parser, config['opt_params'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp = QuantizeParams(parser, config['quantize_params'])

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--retrain', action='store_true', default=False)
    parser.add_argument('--retest', action='store_true', default=False)
    parser.add_argument('--delete_pc', action='store_true', default=False)

    parser.add_argument("--render_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    # args.save_iterations.append(args.iterations)
    
    lp_args = lp.extract(args)
    op_args = op.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)
    print('Running on ', socket.gethostname())
    print("Optimizing " + args.source_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.resume:
        args.start_checkpoint = os.path.join(args.model_path, "resume_ckpt.pth")

    wandb_enabled=(WANDB_FOUND and lp_args.use_wandb)
    tb_writer = prepare_output_and_logger(lp_args, args)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    best_iter = -1
    if wandb_enabled:
        wandb.run.summary['GPU'] = torch.cuda.get_device_name(0).split()[-1]
    if not args.skip_train:
        if os.path.exists(os.path.join(args.model_path,"results_training.json")) and not args.retrain:
            print("Training complete at {}".format(args.model_path))
        else:
            training_time, best_iter = training(args.seed, lp_args, op_args, pp_args, qp_args, 
                                                args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
            full_dict = {args.model_path: {}}
            full_dict[args.model_path].update({"Training time": training_time})
            with open(os.path.join(args.model_path,"results_training.json"), 'w') as fp:
                json.dump(full_dict[args.model_path], fp, indent=True)

    if not args.skip_test:
        if os.path.exists(os.path.join(args.model_path,"results.json")) and not args.retest:
            print("Testing complete at {}".format(args.model_path))
        else:
            images, fps, loaded_iter = render_sets(lp_args, best_iter, pp_args, qp_args, 
                                    args.skip_train, args.skip_test, wandb_enabled, op_args.use_amp)
            if wandb_enabled:
                wandb.log({"rendering_mem": get_gpu_memory()[0]}, step=3)
            full_dict = evaluate(images, args.model_path, loaded_iter, wandb_enabled)
            if wandb_enabled:
                wandb.log({"rendering_mem": get_gpu_memory()[0]}, step=4)
            full_dict[args.model_path].update({"FPS": fps})
            with open(os.path.join(args.model_path,"results.json"), 'w') as fp:
                json.dump(full_dict[args.model_path], fp, indent=True)
    # open(os.path.join(args.model_path, "complete"), "a").close()
    if os.path.exists(os.path.join(args.model_path, "point_cloud")) and args.delete_pc:
        shutil.rmtree(os.path.join(args.model_path, "point_cloud"))
    if os.path.exists(os.path.join(args.model_path, "point_cloud_best")) and args.delete_pc:
        shutil.rmtree(os.path.join(args.model_path, "point_cloud_best"))

    # All done
    print("\nTraining complete.")
