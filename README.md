# EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS
### ECCV 2024
[Sharath Girish](https://sharath-girish.github.io/), [Kamal Gupta](https://kampta.github.io/), [Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/) <br><br>
![Teaser image](assets/teaser.png)

Official implementation of the paper "EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS"

Abstract: *Recently, 3D Gaussian splatting (3D-GS) has gained popularity in novel-view scene synthesis. It addresses the challenges of lengthy training times and slow rendering speeds associated with Neural Radiance Fields (NeRFs). Through rapid, differentiable rasterization of 3D Gaussians, 3D-GS achieves real-time rendering and accelerated training.  They, however, demand substantial memory resources for both training and storage, as they require millions of Gaussians in their point cloud representation for each scene. We present a technique utilizing quantized embeddings to significantly reduce per-point memory storage requirements and a coarse-to-fine training strategy for a faster and more stable optimization of the Gaussian point clouds. Our approach develops a pruning stage which results in scene representations with fewer Gaussians, leading to faster training times and rendering speeds for real-time rendering of high resolution scenes. We reduce storage memory by more than an order of magnitude all while preserving the reconstruction quality. We validate the effectiveness of our approach on a variety of datasets and scenes preserving the visual quality while consuming 10-20x less memory and faster training/inference speed.*

## Cloning the Repository
The codebase is built off of the codebase of 3D-Gaussian Splatting (3D-GS) [here](https://github.com/graphdeco-inria/gaussian-splatting) by Kerbl et. al. The setup instructions are similar to their codebase with small changes in the libraries.
The repository contains submodules 
```shell
# SSH
git clone git@github.com:Sharath-girish/efficientgaussian.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/Sharath-girish/efficientgaussian.git --recursive
```

## Environment Setup

On Linux, it can be set up as
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

## Overview

The different components for the codebase have similar hardware and software requirements as 3D-GS but with an updated ```environment.yml``` as provided. The config files corresponding to different experiments are provided in the ```configs``` folder as yaml files. These files override the default arguments at ```arguments/__init__.py```. They can be further overridden with command-line arguments. The model with compressible gaussians is provided at ```scene/gaussians_sq.py``` with the latent decoders at ```compress/decoders.py```

The main training and evaluation script is ```train_eval.py``` with the different training modes controllable using the CLI arguments as <br>
```--skip_train``` and ```--skip_test``` accordingly.
The final compressed object is stored as a pickle file at ```point_cloud_best/point_cloud_compressed.pkl``` within the experiment log directory.

## Hardware and Software Requirements

We use PyTorch version 1.12.1 along with CUDA version 11.3 as we find this configuration to work for running our experiments. Prior or subsequent versions might work but are not tested.
Training and evaluation requires a CUDA compatible GPU with Compute Capability 7.0+ and will likely consume less than 12 GB RAM for scenes in the MiP-NeRF360 dataset but can go up when also evaluating or if more Gaussians are created based on the densification interval.

## Training and Evaluation commands
To run the training optimizer, simply use

```shell
python train_eval.py --config configs/efficient-3dgs.yaml -s <path to COLMAP or NeRF Synthetic dataset> -m <path to log directory>
```

<details>
<summary><span style="font-weight: bold;">Standard Command Line Arguments for train_eval.py</span></summary>

  #### --config
  Path to the config file which loads the default arguments of the experiment setup
  
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --retrain
  Rerun training even if the completed checkpoint object is present in the experiment logs
  #### --retest
  Rerun testing even if previously evaluated
  #### --skip_train
  Skip training phase
  #### --skip_test
  Skip testing phase
  #### --save_images
  Save images during evaluation stage of the train or test camera set
  #### --delete_pc
  Do not store the trained point cloud at the end (if only obtaining metrics for the run)
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

The ```train_eval.py``` script also consists of additional arguments for hyperparameters of the latent quantization.
Each hyperparameter is a list of 6 attributes: position (```xyz```), rotation(```rotation```), scaling(```scaling```), opacity(```opacity```), base SH component for color(```features_dc```), remaining components(```features_rest```).
As explained in the paper, we compress rotation, opacity and features_rest as they consume bulk of the memory and are easily compressible. Changing the hyperparameter for each attribute can then be done as <br>
```--{attribute_name}_{hyperparameter_name}```. The default hyperparameters used are provided in the configs file
<br>
```configs/efficient-3dgs.yaml```. The list of hyperparameter name arguments for any given ```attribute_name``` is given below

<details>
<summary><span style="font-weight: bold;">Latent Quantization Command Line Arguments for train_eval.py</span></summary>

  #### --{attribute_name}_quant_type
  Quantization type for the attribute. Set to ```none``` to disable or ```sq``` to quantize
  #### --{attribute_name}_latent_dim
  Dimension of the latents
  #### --{attribute_name}_ldec_std
  Standard deviation for initialization of latent decoder parameters
  #### --{attribute_name}_lr_scaling
  LR scaling coefficient of the latents compared to the default uncompressed attribute learning rate.
  #### --{attribute_name}_ldecs_lr
  LR of the latent decoder parameters
  #### --{attribute_name}_latent_scale_norm
  Scaling the learning rate of the latents based on the decoder norm. Set to ```none``` for no scaling and ```div``` to divide learning rate by the decoder norm (typically faster, stable training)
  #### --{attribute_name}_ldecode_matrix
  Type of decode matrix. Set to ```learnable``` for fully learning decoder matrix parameters and ```dft``` to use DFT basis with learnable scaling coefficients.
 
</details>
<br>

Images with width greater than 1600 pixels are automatically resized to 1600 as in 3D-GS. This can be avoided by explicitly specifying resolution ```-r 1```.
The datasets can be obtained from the links provided in the 3D-GS [repository](https://github.com/graphdeco-inria/gaussian-splatting): [MipNeRF360](https://jonbarron.info/mipnerf360/), [Tanks&Temples and Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

### Evaluation
Evaluation executes the same script ```train_eval.py``` with the ```--skip_train``` option to skip retraining and jump to evaluation.
```shell
python train_eval.py --config --config configs/efficient-3dgs.yaml -s <path to COLMAP or NeRF Synthetic dataset> -m <path to log directory of saved model> --skip_train
```
The ```--save_images``` option uses the [render_sets](https://github.com/Sharath-girish/sparse_splat/blob/caacb18f73604f1b571011907a512c3168052832/train_eval.py#L423) function to render images in the train set (unless ```--skip_train``` is specified) and test set and saves them to disk. The rendering FPS is calculated as well. 
Metrics are then evaluated using the [evaluate](https://github.com/Sharath-girish/sparse_splat/blob/caacb18f73604f1b571011907a512c3168052832/train_eval.py#L496) function.

A 360 degree video of the scene can be created by running the ```render_360.py``` script by specifying the dataset flag ```-s``` and the model directory ```-m```.
