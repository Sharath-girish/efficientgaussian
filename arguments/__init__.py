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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, config: dict, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            if key in config:
                value = config[key]
            elif key.startswith("_") and key[1:] in config:
                value = config[key[1:]]
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs="+", type=type(value[0]))
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()

        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class PrefixParamGroup:
    def __init__(self, parser: ArgumentParser, config: dict, prefixes:str, name : str, fill_none = False):
        group = parser.add_argument_group(name)

        for key, value in vars(self).items():
            t = type(value)

            if key in config:
                value = config[key]
            elif key.startswith("_") and key[1:] in config:
                value = config[key[1:]]
            value = value if not fill_none else None 
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            elif t == list:
                for i, prefix in enumerate(prefixes):
                    if prefix+'_'+key in config:
                        cur_value = config[prefix+'_'+key]
                        value[i] = cur_value
                    group.add_argument("--" + prefix+'_'+key, default=value[i], type=type(value[0]))
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):

        group = GroupParams()
        for param_name, value in vars(args).items():
            # if param_name in vars(self) or ("_" + param_name) in vars(self):
            setattr(group, param_name, value)

        for key, value in vars(self).items():
            t = type(value)
            if t == list:
                assert len(value) == len(self.param_names)
                updated_value = []
                for i,param_group in enumerate(self.param_names):
                    updated_value.append(getattr(group, param_group+'_'+key))
                setattr(group, key, updated_value)
        return group
    
class ModelParams(ParamGroup): 
    def __init__(self, parser, config, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.log_interval = 100
        self.shuffle_seed = 0
        self.save_images = False

        self.use_wandb = False
        self.wandb_project = "wandb-project"
        self.wandb_entity = "entity"
        self.wandb_run_name = "test-run"
        self.wandb_mode = "online"
        self.wandb_log_images = False
        self.wandb_tags = ""

        self.testing_interval = 100
        self.checkpoint_interval = 0
        super().__init__(parser, config, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class QuantizeParams(PrefixParamGroup): 
    def __init__(self, parser, config, sentinel=False):
        self.latent_dim = [3,1,16,3,4,1]
        self.latent_norm = ["none", "none", "none", "none", "none", "none"]
        self.latent_scale_norm = ["none"]*6
        self.ldecode_matrix = ["learnable"]*6
        self.use_shift = [1]*6
        self.ldec_std = [1.0]*6
        self.quant_type = ['none']*6
        self.temperature = [0.1]*6
        self.use_gumbel = [0,0,0,0,0,0]
        self.num_layers_dec = [0]*6
        self.hidden_dim_dec = [3,1,16,3,4,1]
        self.activation = ["relu"]*6
        self.diff_sampling = [1]*6
        self.gumbel_period = [0.95]*6
        self.split_seed = 0
        self.opacity_act = "sigmoid"
        self.hc_temp = 2./3.
        self.hc_gamma = 1.05
        self.hc_eta = -0.05
        self.prob_num_layers = [1]*6
        self.ent_lambda = [0.0]*6
        self.noise_freq = 0
        self.param_names = ["xyz", "features_dc", "features_rest", "scaling", "rotation", "opacity"]

        self.codebook_bitwidth = [16, 8, 8, 6, 6, 6]
        super().__init__(parser, config, self.param_names, "Latent Parameters", sentinel)

    
class PipelineParams(ParamGroup):
    def __init__(self, parser, config):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, config, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser, config):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_dc_lr = 0.0025
        self.features_rest_lr = self.features_dc_lr/20.0
        # self.latents_lr_scaling = [1.0]*6
        # self.ldecs_lr = [1.0e-4]*6
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.prob_models_lr = 1.0e-4

        self.position_ldecs_lr = 1.0e-4
        self.features_dc_ldecs_lr = 1.0e-4
        self.features_rest_ldecs_lr = 1.0e-4
        self.opacity_ldecs_lr = 1.0e-4
        self.scaling_ldecs_lr = 1.0e-4
        self.rotation_ldecs_lr = 1.0e-4

        self.position_lr_scaling = 1.0
        self.features_dc_lr_scaling = 1.0
        self.features_rest_lr_scaling = 1.0
        self.opacity_lr_scaling = 1.0
        self.scaling_lr_scaling = 1.0
        self.rotation_lr_scaling = 1.0

        self.use_amp = False
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.accumulate_fraction = 0.0
        self.search_best_iters = 0
        self.resize_period = 0.0
        self.resize_scale = 1.0
        self.transform = "downsample" # "blur", "resize", "downsample", "none"
        super().__init__(parser, config, "Optimization Parameters")

    def extract(self, args):
        g = super().extract(args)
        g.latents_lr_scaling = [self.position_lr_scaling, 
                                   self.features_dc_lr_scaling, 
                                   self.features_rest_lr_scaling, 
                                   self.scaling_lr_scaling, 
                                   self.rotation_lr_scaling, 
                                   self.opacity_lr_scaling]
        g.ldecs_lr = [self.position_ldecs_lr,
                         self.features_dc_ldecs_lr,
                         self.features_rest_ldecs_lr,
                         self.scaling_ldecs_lr,
                         self.rotation_ldecs_lr,
                         self.opacity_ldecs_lr]
        return g

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
