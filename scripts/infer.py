# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from glob import glob
from cv2 import imwrite
import time
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from monoformer.models.model_wrapper import ModelWrapper
from monoformer.datasets.augmentations import resize_image, to_tensor
from monoformer.utils.horovod import hvd_init, rank, world_size, print0
from monoformer.utils.image import load_image
from monoformer.utils.config import parse_test_file
from monoformer.utils.load import set_debug
from monoformer.utils.depth import write_depth, inv2depth, viz_inv_depth
from monoformer.utils.logging import pcolor

import cv2

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)
#epoch=14_KITTI_raw-eigen_val_files-velodyne-abs_rel_gt=0.094.ckpt

def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)',default='.checkpoints/monoformer.ckpt')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    parser.add_argument('--config',type=str)
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args


@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half, save):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None
    start = time.time()

    # Load image
    image = load_image(input_file)
    original_image_shape = image.size
    # Resize and to tensor
    image_shape=(192,640)
    image = resize_image(image, image_shape)

    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)

    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=inv2depth(pred_inv_depth))
    else:
        # Prepare RGB image
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        #viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth) * 255
        # Concatenate both vertically
        # image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        image = viz_pred_inv_depth
        image = image[:, :, ::-1]
        image = cv2.resize(image,original_image_shape)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        imwrite(output_file, image)
    
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간



def main(args):

    # Initialize horovod
    hvd_init()
    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint,args.config)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)

    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)
    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if os.path.isdir(args.input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg','JPG']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [args.input]

    # Process each file
    for fn in files[rank()::world_size()]:
        infer_and_save_depth(
            fn, args.output, model_wrapper, image_shape, args.half, args.save)

if __name__ == '__main__':
    args = parse_args()
    main(args)
