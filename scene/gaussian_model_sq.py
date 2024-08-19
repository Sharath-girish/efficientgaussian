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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, HardConcrete, CompressedLatents
from torch import nn
import os
import pickle
from arguments import QuantizeParams
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from scene.gaussian_model import GaussianModel
from compress.decoders import LatentDecoder, CodebookQuantize, DecoderIdentity, DecoderLayer
from utils.graphics_utils import BasicPointCloud
from collections import OrderedDict
from compress.bitEstimator import BitEstimator
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModelSQ(GaussianModel):

    def __init__(self, sh_degree : int, latent_args: QuantizeParams):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree 
        self._latents = OrderedDict([(n,torch.empty(0)) for n in latent_args.param_names])
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.influence = torch.empty(0)
        self.infl_denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.split_generator = torch.Generator(device="cuda")
        self.split_generator.manual_seed(latent_args.split_seed)
        self.setup_functions()
        
        self.param_names = latent_args.param_names
        self.feature_dims = OrderedDict([
            ("xyz",3),
            ("features_dc",3),
            ("features_rest",3 * ((self.max_sh_degree + 1) ** 2 - 1)),
            ("scaling",3),
            ("rotation",4),
            ("opacity",1),
        ])
        self.prob_models, self.ent_lambdas, self.latent_decoders = OrderedDict(), OrderedDict(), OrderedDict()

        for i,param_name in enumerate(self.param_names):
            if latent_args.quant_type[i] == 'sq':
                self.latent_decoders[param_name] = LatentDecoder(
                    latent_dim=latent_args.latent_dim[i],
                    feature_dim=self.feature_dims[param_name],
                    ldecode_matrix=latent_args.ldecode_matrix[i],
                    norm=latent_args.latent_norm[i],
                    num_layers_dec=latent_args.num_layers_dec[i],
                    hidden_dim_dec=latent_args.hidden_dim_dec[i],
                    activation=latent_args.activation[i],
                    use_shift=latent_args.use_shift[i],
                    ldec_std=latent_args.ldec_std[i],
                    use_gumbel=latent_args.use_gumbel[i],
                    diff_sampling=latent_args.diff_sampling[i]
                ).cuda()
                # self.latent_decoders[param_name].reset_parameters(
                #     'constant',1.0/latent_args.latent_dim[i])
                if latent_args.ent_lambda[i]>0.0:
                    self.prob_models[param_name] = BitEstimator(
                        latent_args.latent_dim[i],
                        num_layers=latent_args.prob_num_layers[i]
                    ).cuda()
                    self.ent_lambdas[param_name] = latent_args.ent_lambda[i]
            elif latent_args.quant_type[i] == 'vq':
                self.latent_decoders[param_name] = CodebookQuantize(
                    codebook_bitwidth=latent_args.codebook_bitwidth[i],
                    codebook_dim=self.feature_dims[param_name],
                    use_gumbel=latent_args.use_gumbel[i],
                ).cuda()
            else:
                self.latent_decoders[param_name]=DecoderIdentity()
        
        self.hc = HardConcrete(latent_args.hc_gamma, latent_args.hc_eta, latent_args.hc_temp).cuda()
        if latent_args.opacity_act == "sigmoid":
            self.opacity_activation = torch.sigmoid
            self.inverse_opacity_activation = inverse_sigmoid
        elif latent_args.opacity_act == "hc":
            self.opacity_activation = self.hc.forward
            self.inverse_opacity_activation = self.hc.invert

    def to_hc(self):
        op = self.get_opacity
        self._latents["opacity"].data = self.hc.invert(op)
        self.opacity_activation = self.hc.forward
        self.inverse_opacity_activation = self.hc.invert

    def capture(self):
        return (
            self.active_sh_degree,
            self._latents,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.influence,
            self.infl_denom,
            OrderedDict([(n, l.state_dict()) for n,l in self.latent_decoders.items()]),
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._latents,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        influence,
        infl_denom,
        ldec_dicts,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.influence = influence
        self.infl_denom = infl_denom
        for n in ldec_dicts:
            self.latent_decoders[n].load_state_dict(ldec_dicts[n])

    def capture_best_state(self):
        return (
            self.active_sh_degree,
            {k:v.detach().cpu() for k,v in self._latents.items()},
            self.max_radii2D.detach().cpu(),
            self.xyz_gradient_accum.detach().cpu(),
            self.denom.detach().cpu(),
            self.influence.detach().cpu(),
            self.infl_denom.detach().cpu(),
            OrderedDict([(n, {k: v.detach().cpu() for k, v in l.state_dict().items()}) for n,l in self.latent_decoders.items()]),
            self.spatial_lr_scale,
        )
        
    def restore_best_state(self, model_args, training_args):
        (self.active_sh_degree, 
        self._latents,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        influence,
        infl_denom,
        ldec_dicts,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self._latents = {k:v.cuda() for k,v in self._latents.items()}
        self.max_radii2D = self.max_radii2D.cuda()
        self.xyz_gradient_accum = xyz_gradient_accum.cuda()
        self.denom = denom.cuda()
        self.influence = influence.cuda()
        self.infl_denom = infl_denom.cuda()
        for n in ldec_dicts:
            self.latent_decoders[n].load_state_dict(ldec_dicts[n])
    @property
    def _xyz(self):
        xyz = self.latent_decoders["xyz"](self._latents["xyz"])
        return xyz
    
    @property
    def _features_dc(self):
        if isinstance(self.latent_decoders["features_dc"], DecoderIdentity):
            features_dc = self._latents["features_dc"]
        else:
            features_dc = self.latent_decoders["features_dc"](self._latents["features_dc"])
            features_dc = features_dc.reshape(features_dc.shape[0], 1, 3)
        return features_dc
    
    @property
    def _features_rest(self):
        if isinstance(self.latent_decoders["features_rest"], DecoderIdentity):
            features_rest = self._latents["features_rest"]
        else:
            features_rest = self.latent_decoders["features_rest"](self._latents["features_rest"])
            features_rest = features_rest.reshape(features_rest.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)
        return features_rest
    
    @property
    def _scaling(self):
        return self.latent_decoders["scaling"](self._latents["scaling"])
    
    @property
    def _rotation(self):
        return self.latent_decoders["rotation"](self._latents["rotation"])
    
    @property
    def _opacity(self):
        return self.latent_decoders["opacity"](self._latents["opacity"])
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    def parameters(self):
        return list(self._latents.values()) + \
                [param for decoder in self.latent_decoders.values() for param in list(decoder.parameters())]
    
    def named_parameters(self):
        parameter_dict = self._latents
        for n, decoder in self.latent_decoders.items():
            parameter_dict.update(
                {n+'.'+param_name:param for param_name, param in dict(decoder.named_parameters()).items()}
                )
        return parameter_dict
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._latents = OrderedDict([(n,None) for n in self.param_names])

        init = self.latent_decoders["xyz"].invert(fused_point_cloud)
        self._latents["xyz"] = nn.Parameter(init.requires_grad_(True))

        init = features[:,:,0:1].transpose(1, 2).contiguous()
        if not isinstance(self.latent_decoders["features_dc"], DecoderIdentity):
            init = self.latent_decoders["features_dc"].invert(init.flatten(start_dim=1).contiguous())
        self._latents["features_dc"] = nn.Parameter(init.requires_grad_(True))

        init = features[:,:,1:].transpose(1, 2).contiguous()
        if not isinstance(self.latent_decoders["features_rest"], DecoderIdentity):
            if isinstance(self.latent_decoders["features_rest"], LatentDecoder):
                init = torch.zeros((features.size(0),self.latent_decoders["features_rest"].latent_dim)).to(init).contiguous()
            elif isinstance(self.latent_decoders["features_rest"], CodebookQuantize):
                init = torch.zeros((features.size(0),self.latent_decoders["features_rest"].codebook_bitwidth)).to(init).contiguous()
        self._latents["features_rest"] = nn.Parameter(init.requires_grad_(True))


        init = self.latent_decoders["scaling"].invert(scales)
        self._latents["scaling"] = nn.Parameter(init.requires_grad_(True))

        init = self.latent_decoders["rotation"].invert(rots)
        self._latents["rotation"] = nn.Parameter(init.requires_grad_(True))

        init = self.latent_decoders["opacity"].invert(opacities)
        self._latents["opacity"] = nn.Parameter(init.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def size(self, use_torchac=False, use_prob_model=False):
        with torch.no_grad():
            latents_size = ldec_size = 0
            for param in self.param_names:
                ldec_size += self.latent_decoders[param].size(use_torchac)
                if isinstance(self.latent_decoders[param], DecoderIdentity):
                    p = self._latents[param]
                    latents_size += p.numel()*torch.finfo(p.dtype).bits
                else:
                    for dim in range(self._latents[param].size(1)):
                        weight = torch.round(self._latents[param][:,dim]).long()
                        unique_vals, counts = torch.unique(weight, return_counts = True)
                        if not use_prob_model:
                            probs = counts/torch.sum(counts)
                        else:
                            assert self.prob_model is not None
                            probs = self.prob_model(unique_vals+0.5,single_channel=dim) - self.prob_model(unique_vals-0.5,single_channel=dim)

                        if not use_torchac:
                            information_bits = torch.clamp(-1.0 * torch.log(probs + 1e-10) / np.log(2.0), 0, 1000)
                            size_bits = torch.sum(information_bits*counts).item()
                            latents_size += size_bits
                        else:
                            import torchac
                            cdf = torch.cumsum(probs,dim=0)
                            cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
                            cdf = cdf.unsqueeze(0).repeat(self.codebook.size(0),1)
                            cdf = cdf/cdf[:,-1:]
                            
                            weight = weight - weight.min()
                            unique_vals, counts = torch.unique(weight, return_counts = True)
                            mapping = torch.zeros((weight.max().item()+1))
                            mapping[unique_vals] = torch.arange(unique_vals.size(0)).to(mapping)
                            weight = mapping[weight]
                            cdf = torch.cumsum(counts/counts.sum(),dim=0)
                            cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
                            cdf = cdf.unsqueeze(0).repeat(weight.size(0),1)
                            cdf = cdf/cdf[:,-1:] # Normalize the final cdf value just to keep torchac happy
                            byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.detach().cpu().to(torch.int16), \
                                                                    check_input_bounds=True)
                            latents_size += len(byte_stream)*8

        return ldec_size+latents_size
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.influence = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.infl_denom = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.lr_scaling = OrderedDict()
        for i,param in enumerate(self.param_names):
            self.lr_scaling[param] = 1.0 if isinstance(self.latent_decoders[param], DecoderIdentity) else training_args.latents_lr_scaling[i]
        self.orig_lr = [training_args.position_lr_init, training_args.features_dc_lr, training_args.features_rest_lr, 
                        training_args.scaling_lr, training_args.rotation_lr, training_args.opacity_lr]
        lr = {
                'xyz':training_args.position_lr_init * self.spatial_lr_scale*self.lr_scaling["xyz"],
                'features_dc':training_args.features_dc_lr*self.lr_scaling["features_dc"],
                'features_rest':training_args.features_rest_lr*self.lr_scaling["features_rest"],
                'scaling':training_args.scaling_lr*self.lr_scaling["scaling"],
                'rotation':training_args.rotation_lr*self.lr_scaling["rotation"],
                'opacity':training_args.opacity_lr*self.lr_scaling["opacity"]
            }
        l = []
        for i,param in enumerate(self.param_names):
            l += [{'params': [self._latents[param]], 'lr': lr[param], "name": param}]
            if not isinstance(self.latent_decoders[param], DecoderIdentity):
                l += [{'params': self.latent_decoders[param].parameters(), 'lr': training_args.ldecs_lr[i], "name":f"ldec_{param}"}]

        for i, prob_model in enumerate(self.prob_models.values()):
            l += [{'params': prob_model.parameters(), 'lr': training_args.prob_models_lr, "name":f"prob_{self.param_names[i]}"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale*self.lr_scaling["xyz"],
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale*self.lr_scaling["xyz"],
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, latent_args):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] in self.param_names:
                idx = self.param_names.index(param_group["name"])
                if latent_args.latent_scale_norm[idx] == "div":
                    lr = self.orig_lr[idx]*self.lr_scaling[param_group["name"]]
                    lr /= self.latent_decoders[param_group["name"]].scale_norm()
                    param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(3 * ((self.max_sh_degree + 1) ** 2 - 1)):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
            
    def save_compressed_pkl(self, path, latent_args):
        mkdir_p(os.path.dirname(path))

        latents = OrderedDict()
        decoder_state_dict = OrderedDict()
        decoder_args = OrderedDict()
        for i,attribute in enumerate(self.param_names):
            if isinstance(self.latent_decoders[attribute], DecoderIdentity):
                latents[attribute] = self._latents[attribute].detach().cpu()
            else:
                latent = self._latents[attribute].detach().cpu()
                compressed_obj = CompressedLatents()
                compressed_obj.compress(latent)
                latents[attribute] = compressed_obj
                decoder_args[attribute] = {
                    'latent_dim': latent_args.latent_dim[i],
                    'feature_dim': self.feature_dims[attribute],
                    'ldecode_matrix': latent_args.ldecode_matrix[i],
                    'norm': latent_args.latent_norm[i],
                    'num_layers_dec': latent_args.num_layers_dec[i],
                    'hidden_dim_dec': latent_args.hidden_dim_dec[i],
                    'activation': latent_args.activation[i],
                    'use_shift': latent_args.use_shift[i],
                    'ldec_std': latent_args.ldec_std[i],
                    'use_gumbel': latent_args.use_gumbel[i],
                    'diff_sampling': latent_args.diff_sampling[i]
                }
                decoder_state_dict[attribute] = self.latent_decoders[attribute].state_dict().copy()

        with open(path,'wb') as f:
            pickle.dump({
                         'latents': latents,
                         'decoder_state_dict': decoder_state_dict,
                         'decoder_args': decoder_args,
            }, f)

    def load_compressed_pkl(self, path):

        with open(path,'rb') as f:
            data = pickle.load(f)
            latents = data['latents']
            decoder_state_dict = data['decoder_state_dict']
            decoder_args = data['decoder_args']

        for attribute in latents:
            if isinstance(latents[attribute], CompressedLatents):
                assert isinstance(self.latent_decoders[attribute], LatentDecoder)
                self.latent_decoders[attribute] = LatentDecoder(**decoder_args[attribute]).cuda()
                self.latent_decoders[attribute].load_state_dict(decoder_state_dict[attribute])
                self._latents[attribute] = nn.Parameter(latents[attribute].uncompress().cuda().requires_grad_(True))
            else:
                self._latents[attribute] = nn.Parameter(latents[attribute].cuda().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def decode_latents(self):
        with torch.no_grad():
            for param in self.param_names:
                if not isinstance(self.latent_decoders[param], DecoderIdentity):
                    decoded = self.latent_decoders[param](self._latents[param])
                    if param == "features_rest":
                        self._latents[param].data = decoded.reshape(self._latents[param].shape[0], 
                                                                                 (self.max_sh_degree + 1) ** 2 - 1, 3)
                    elif param == "features_dc":
                        self._latents[param].data = decoded.reshape(self._latents[param].shape[0], 1, 3)
                    else:
                        self._latents[param].data = decoded
                    self.latent_decoders[param] = DecoderIdentity()

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        latent_opacities = self.latent_decoders["opacity"].invert(opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(latent_opacities, "opacity")
        self._latents["opacity"] = optimizable_tensors["opacity"]

    def load_ply(self, path):
        raise NotImplementedError("Loading from ply not implemented, use load_compressed_pkl!")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        assert "ldec" not in name, "Latent decoder params cannot be replaced!"
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "ldec" in group["name"] or "prob" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        for name in self.param_names:
            self._latents[name] = optimizable_tensors[name]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.influence = self.influence[valid_points_mask]
        self.infl_denom = self.infl_denom[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "ldec" in group["name"] or "prob" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "opacity": new_opacities,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        for param in self.param_names:
            self._latents[param] = optimizable_tensors[param]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.influence = torch.cat((self.influence,torch.zeros((new_xyz.shape[0]), device="cuda")))
        self.infl_denom = torch.cat((self.infl_denom,torch.zeros((new_xyz.shape[0]), device="cuda")))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds, generator=self.split_generator)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_xyz = self.latent_decoders["xyz"].invert(new_xyz)

        if isinstance(self.latent_decoders["scaling"],DecoderIdentity):
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        elif isinstance(self.latent_decoders["scaling"],CodebookQuantize):
            self.latent_decoders["scaling"].codebook.data -= self.scaling_inverse_activation(torch.Tensor([0.8*N]))\
                                                                .to(self.latent_decoders["scaling"].codebook.data)
            new_scaling = self._latents["scaling"][selected_pts_mask].repeat(N,1)
        elif isinstance(self.latent_decoders["scaling"],LatentDecoder):
            last_layer = list(self.latent_decoders["scaling"].layers.children())[-1]
            assert isinstance(last_layer, DecoderLayer)
            if last_layer.shift is not None:
                last_layer.shift.data -= self.scaling_inverse_activation(torch.Tensor([0.8*N])).to(last_layer.shift.data)
            else:
                last_layer.shift = -nn.Parameter(self.scaling_inverse_activation(torch.Tensor([0.8*N])),
                                                 requires_grad=False).to(last_layer.scale)
            new_scaling = self._latents["scaling"][selected_pts_mask].repeat(N,1)

        
        new_rotation = self._latents["rotation"][selected_pts_mask].repeat(N,1)

        if isinstance(self.latent_decoders["features_dc"],DecoderIdentity):
            new_features_dc = self._latents["features_dc"][selected_pts_mask].repeat(N,1,1)
        else:
            new_features_dc = self._latents["features_dc"][selected_pts_mask].repeat(N,1)

        if isinstance(self.latent_decoders["features_rest"],DecoderIdentity):
            new_features_rest = self._latents["features_rest"][selected_pts_mask].repeat(N,1,1)
        else:
            new_features_rest = self._latents["features_rest"][selected_pts_mask].repeat(N,1)

        new_opacity = self._latents["opacity"][selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._latents["xyz"][selected_pts_mask]
        new_features_dc = self._latents["features_dc"][selected_pts_mask]
        new_features_rest = self._latents["features_rest"][selected_pts_mask]
        new_opacities = self._latents["opacity"][selected_pts_mask]
        new_scaling = self._latents["scaling"][selected_pts_mask]
        new_rotation = self._latents["rotation"][selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        # grads = torch.norm(self.xyz_gradient_accum, dim=-1, keepdim=True)/ self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum[update_filter] += viewspace_point_tensor.grad[update_filter,:2]
        self.denom[update_filter] += 1

    @torch.no_grad()
    def add_influence_stats(self, influence):
        self.influence += influence
        self.infl_denom[influence>0] += 1

    @torch.no_grad()
    def prune_influence(self, quantile_threshold):
        threshold = torch.quantile(self.influence,quantile_threshold)
        prune_mask = self.influence<=threshold
        self.prune_points(prune_mask)
        self.influence *= 0
        self.infl_denom *= 0