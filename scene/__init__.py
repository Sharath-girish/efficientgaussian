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
import shutil
import random
import json
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.cameras import RenderCamera
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_sq import GaussianModelSQ
from arguments import ModelParams
from utils.system_utils import mkdir_p
from utils.graphics_utils import generate_ellipse_path, transform_poses_pca, generate_spiral_path, unpad_poses, pad_poses
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_ply=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        generator = random.Random(args.shuffle_seed)

        if load_iteration and load_ply:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, eval=True)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, eval=True)
        elif os.path.exists(os.path.join(args.source_path, "nerf", "out_train.json")):
            print("Found out_train.json file, assuming RTMV data set!")
            scene_info = sceneLoadTypeCallbacks["RTMV"](args.source_path, args.white_background, eval=True)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info = scene_info

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            generator.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            generator.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter and load_ply:
            ply_file = os.path.join(self.model_path,
                                    "point_cloud",
                                    "iteration_" + str(self.loaded_iter),
                                    "point_cloud.ply")
            if os.path.exists(ply_file) and (type(gaussians) == GaussianModel):
                self.gaussians.load_ply(ply_file)
            else:
                self.gaussians.load_compressed_pkl(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud_compressed.pkl"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_compressed(self, iteration, quantize_params):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_compressed_pkl(os.path.join(point_cloud_path, "point_cloud_compressed.pkl"),quantize_params)

    def save_best(self):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_best")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def save_best_compressed(self, quantize_params):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_best")
        self.gaussians.save_compressed_pkl(os.path.join(point_cloud_path, "point_cloud_compressed.pkl"),quantize_params)

    def link_best(self, iteration):
        src_dir = os.path.join(os.getcwd(), self.model_path, "point_cloud_best")
        dest_dir = os.path.join(os.getcwd(), self.model_path, "point_cloud/iteration_{}".format(iteration))
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        mkdir_p(dest_dir)
        os.symlink(os.path.join(src_dir, "point_cloud.ply"), os.path.join(dest_dir, "point_cloud.ply"))

    def link_best_compressed(self, iteration):
        src_dir = os.path.join(os.getcwd(), self.model_path, "point_cloud_best")
        dest_dir = os.path.join(os.getcwd(), self.model_path, "point_cloud/iteration_{}".format(iteration))
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        mkdir_p(dest_dir)
        os.symlink(os.path.join(src_dir, "point_cloud_compressed.pkl"), os.path.join(dest_dir, "point_cloud_compressed.pkl"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainCameraInfo(self):
        return self.scene_info.train_cameras

    def getTestCameraInfo(self):
        return self.scene_info.test_cameras
    
    def getRender360Cameras(self, scale=1.0):
        train_camera = self.train_cameras[scale][0]
        world_view = torch.stack([cam.world_view_transform for cam in self.train_cameras[scale]]).detach().cpu().numpy()
        c2w = np.transpose(np.linalg.inv(world_view), (0, 2, 1))
        poses, transform = transform_poses_pca(c2w)
        poses = generate_ellipse_path(poses, n_frames=240, const_speed=True, z_variation=0., z_phase=0.)
        # poses = generate_spiral_path()
        poses = unpad_poses(np.linalg.inv(transform) @ pad_poses(poses))
        R = poses[:, :3, :3]
        t = (- np.transpose(R, (0,2,1)) @ poses[:, :3, 3:4]).squeeze(-1)
        camlist = {scale:[]}
        for i in range(0,R.shape[0]):
            camlist[scale].append(RenderCamera(width=train_camera.image_width, height=train_camera.image_height,
                                        R=R[i], T=t[i], FoVx=train_camera.FoVx, FoVy=train_camera.FoVy,uid=i)
            )
        
        return camlist[scale]

    