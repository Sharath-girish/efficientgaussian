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
import math
import sys
from datetime import datetime
import numpy as np
import random

eps = 1e-7
class HardConcrete(torch.nn.Module):

    def __init__(self, gamma=1.05, eta=-.05, temperature=2./3.):
        super(HardConcrete, self).__init__()
        self.gamma = gamma
        self.eta = eta
        self.log_alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True)

    def forward(self, x):
        # Clamp x to [0,1]
        x = torch.clamp(x, eps, 1-eps)
        # x = (x-torch.mean(x,dim=0)+0.5)/torch.max(1,torch.std(x,dim=0))
        gate = torch.sigmoid((torch.log(x) - torch.log(1-x) + self.log_alpha) / self.temperature)
        gate = gate * (self.gamma - self.eta) + self.eta
        return torch.clamp(gate, 0, 1)
    
    def invert(self,gate):
        # Invert the gate
        with torch.no_grad():
            x = (gate-self.eta)/(self.gamma-self.eta)
            x = inverse_sigmoid(x)
            x = torch.sigmoid(-self.log_alpha+x*self.temperature)
            return x
    
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent, seed=0):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))

def sample_camera_order(cameras, g):
    distances = torch.stack([cam.camera_center for cam in cameras])
    distances = torch.cdist(distances, distances)
    cam_idx = []
    for cur_iter in range(len(cameras)):
        if cur_iter == 0:
            cur_distance = torch.mean(distances,axis=1)
        else:
            prev_sample = cam_idx[-1]
            cur_distance = distances[prev_sample]

        probs = torch.softmax(cur_distance,dim=0)
        sampled_idx = torch.multinomial(probs,num_samples=1).item()
        distances[:, sampled_idx] = -1e10
        
        cam_idx.append(sampled_idx)

    assert len(np.unique(np.array(cam_idx))) == len(cameras)
    return cam_idx

def mean_distances(cameras, g):
    cam_center = torch.stack([cam.camera_center for cam in cameras])
    distances = torch.cdist(cam_center, cam_center)
    return torch.mean(distances,axis=1)

class DecayScheduler(object):
    '''A simple class for decaying schedule of various hyperparameters.'''

    def __init__(self, total_steps, decay_name='fix', start=0, end=0, params=None):
        self.decay_name = decay_name
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.params = params

    def __call__(self, step):
        if self.decay_name == 'fix':
            return self.start
        elif self.decay_name == 'linear':
            if step>self.total_steps:
                return self.end
            return self.start + (self.end - self.start) * step / self.total_steps
        elif self.decay_name == 'exp':
            if step>self.total_steps:
                return self.end
            return max(self.end, self.start*(np.exp(-np.log(1/self.params['temperature'])*step/self.total_steps/self.params['decay_period'])))
            # return self.start * (self.end / self.start) ** (step / self.total_steps)
        elif self.decay_name == 'inv_sqrt':
            return self.start * (self.total_steps / (self.total_steps + step)) ** 0.5
        elif self.decay_name == 'cosine':
            if step>self.total_steps:
                return self.end
            return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(step / self.total_steps * math.pi))
        else:
            raise ValueError('Unknown decay name: {}'.format(self.decay_name))
        

class CompressedLatents(object):
    
    def compress(self, latent):
        import torchac
        assert latent.dim() == 2, "Latent should be 2D"
        self.num_latents, self.latent_dim = latent.shape
        flattened = latent.flatten()

        weight = torch.round(flattened).int()
        unique_vals, counts = torch.unique(weight, return_counts = True)
        probs = counts/torch.sum(counts)
        tail_idx = torch.where(probs <= 1.0e-5)[0]
        tail_vals = unique_vals[tail_idx]
        self.tail_locs = {}
        for val in tail_vals:
            weight[weight == val] = unique_vals[counts.argmax()]
            self.tail_locs[val.item()] = torch.where(weight == val)[0].detach().cpu()
        unique_vals, counts = torch.unique(weight, return_counts = True)
        probs = counts/torch.sum(counts)
        weight = weight.detach().cpu()

        cdf = torch.cumsum(probs,dim=0)
        cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
        cdf = cdf/cdf[-1:] # Normalize the final cdf value just to keep torchac happy
        cdf = cdf.unsqueeze(0).repeat(flattened.size(0),1)
        
        mapping = {val.item():idx.item() for val,idx in zip(unique_vals,torch.arange(unique_vals.shape[0]))}
        self.mapping = mapping
        weight.apply_(mapping.get)
        byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.to(torch.int16))
        
        self.byte_stream, self.mapping, self.cdf = byte_stream, mapping, cdf[0].detach().cpu().numpy()

    def uncompress(self):
        import torchac
        cdf = torch.tensor(self.cdf).unsqueeze(0).repeat(self.num_latents*self.latent_dim,1)
        weight = torchac.decode_float_cdf(cdf, self.byte_stream)
        weight = weight.to(torch.float32)
        for val, locs in self.tail_locs.items():
            weight[locs] = val
        # weight = self.tail_decode(cdf, self.byte_stream, self.tail_vals, self.tail_idx)
        # weight = weight.to(torch.float32)
        inverse_mapping = {v:k for k,v in self.mapping.items()}
        weight.apply_(inverse_mapping.get)
        weight = weight.view(self.num_latents, self.latent_dim)
        return weight

# def compress(latent):
#     import torchac
#     assert latent.dim() == 2, "Latent should be 2D"
#     num_latents, latent_dim = latent.shape
#     flattened = latent.flatten()

#     weight = torch.round(flattened).int()
#     unique_vals, counts = torch.unique(weight, return_counts = True)
#     probs = counts/torch.sum(counts)
#     weight = weight.detach().cpu()

#     cdf = torch.cumsum(probs,dim=0)
#     cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
#     cdf = cdf.unsqueeze(0).repeat(flattened.size(0),1)
#     cdf = cdf/cdf[:,-1:] # Normalize the final cdf value just to keep torchac happy
    
#     mapping = {val.item():idx.item() for val,idx in zip(unique_vals,torch.arange(unique_vals.shape[0]))}
#     weight.apply_(mapping.get)
#     byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.to(torch.int16), \
#                                             check_input_bounds=True)
    
#     aux_info = {"num_latents":num_latents, "latent_dim":latent_dim,
#                 "mapping":mapping, "cdf":cdf[0].detach().cpu()}
#     return byte_stream, aux_info

# def uncompress(byte_stream, aux_info):
#     import torchac
#     cdf = aux_info["cdf"].unsqueeze(0).repeat(aux_info["num_latents"]*aux_info["latent_dim"],1)
#     weight = torchac.decode_float_cdf(cdf, byte_stream)
#     weight = weight.to(torch.float32)
#     inverse_mapping = {v:k for k,v in aux_info["mapping"].items()}
#     weight.apply_(inverse_mapping.get)
#     weight = weight.view(aux_info["num_latents"], aux_info["latent_dim"])
#     return weight
