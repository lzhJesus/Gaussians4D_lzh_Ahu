from argparse import ArgumentParser
import os
import sys
import torch
from arguments import ModelHiddenParams
from scene.deformation import deform_network
from train import setup_seed

if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")

    hp = ModelHiddenParams(parser)
    parser.add_argument("--configs", type=str, default = "/media/data3/code/lzh/4DGaussian/arguments/dynerf/cut_roasted_beef.py")
    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    deformation = deform_network(hp.extract(args))
    weight_dict = torch.load("/media/data3/code/lzh/4DGaussian/output/dynerf/cut_roasted_beef/point_cloud/iteration_14000/deformation.pth",map_location="cuda")
    deformation.load_state_dict(weight_dict)
    deformation = deformation.to("cuda")
    
