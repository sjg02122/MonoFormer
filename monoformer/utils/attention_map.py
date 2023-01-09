import matplotlib.pyplot as plt
import torch 
import random
import argparse
import numpy as np
import os
import shutil

from glob import glob
from cv2 import imwrite
import torchvision

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

def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn

def visualize_attention(input, model,prediction,save_name):
    # input = input[0]
    # input = (input + 1.0)/2.0

    attn1 = model.pretrained.attention["attn_1"]
    attn2 = model.pretrained.attention["attn_2"]
    attn3 = model.pretrained.attention["attn_3"]
    attn4 = model.pretrained.attention["attn_4"]

    input = input.detach().cpu()
    attn1 = attn1.detach().cpu()
    attn2 = attn2.detach().cpu()
    attn3 = attn3.detach().cpu()
    attn4 = attn4.detach().cpu()
    # prediction = prediction


    input = input.type(torch.float32)
    attn1 = attn1.type(torch.float32)
    attn2 = attn2.type(torch.float32)
    attn3 = attn3.type(torch.float32)
    attn4 = attn4.type(torch.float32)
    # prediction = np.float32(prediction)

    # print(type(input),input.dtype)
    # print(type(attn1),attn1.dtype)
    # print(type(attn2),attn2.dtype)
    # print(type(attn3),attn3.dtype)
    # print(type(attn4),attn4.dtype)
    # print(input.size())
    input = input[0]
    input = torch.unsqueeze(input,0)


    h = [6,12,18,24]
    first_num =180
    # upper left

    # plt.axis('off')
    cmap_colors = ['magma','plasma']
    cmap_setting = cmap_colors[0]
    plt.imsave(save_name[:-4]+'_upper_6.png',get_mean_attention_map(attn1, first_num, input.shape),cmap=cmap_setting)
    plt.imsave(save_name[:-4]+'_upper_12.png',get_mean_attention_map(attn2, first_num, input.shape),cmap=cmap_setting)
    plt.imsave(save_name[:-4]+'_upper_18.png',get_mean_attention_map(attn3, first_num, input.shape),cmap=cmap_setting)
    plt.imsave(save_name[:-4]+'_upper_24.png',get_mean_attention_map(attn4, first_num, input.shape),cmap=cmap_setting)

    plt.imsave(save_name[:-4]+'_down_6.png',get_mean_attention_map(attn1, -1, input.shape),cmap=cmap_setting)
    plt.imsave(save_name[:-4]+'_down_12.png',get_mean_attention_map(attn2, -1, input.shape),cmap=cmap_setting)
    plt.imsave(save_name[:-4]+'_down_18.png',get_mean_attention_map(attn3, -1, input.shape),cmap=cmap_setting)
    plt.imsave(save_name[:-4]+'_down_24.png',get_mean_attention_map(attn4, -1, input.shape),cmap=cmap_setting)

    input = input[0]
    input = torch.unsqueeze(input,0)
    a = plt.figure()

    plt.subplot(3,4,1), plt.imshow(input.squeeze().permute(1,2,0)), plt.title("Input", fontsize=8), plt.axis("off")
    plt.subplot(3,4,2), plt.imshow(prediction), plt.set_cmap("magma"), plt.title("Prediction", fontsize=8), plt.axis("off")
    plt.subplot(345),
    plt.imshow(get_mean_attention_map(attn1, first_num, input.shape))
    plt.ylabel("Upper left corner", fontsize=8)
    plt.title(f"Layer {h[0]}", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])

    plt.subplot(346),
    plt.imshow(get_mean_attention_map(attn2, first_num, input.shape))
    plt.title(f"Layer {h[1]}", fontsize=8)
    plt.axis("off"),

    plt.subplot(347),
    plt.imshow(get_mean_attention_map(attn3, first_num, input.shape))
    plt.title(f"Layer {h[2]}", fontsize=8)
    plt.axis("off"),


    plt.subplot(348),
    plt.imshow(get_mean_attention_map(attn4, first_num, input.shape))
    plt.title(f"Layer {h[3]}", fontsize=8)
    plt.axis("off"),


    # lower right
    second_num = -160
    plt.subplot(3,4,9), plt.imshow(get_mean_attention_map(attn1, second_num, input.shape))
    plt.ylabel("Lower right corner", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])
    plt.subplot(3,4,10), plt.imshow(get_mean_attention_map(attn2, second_num, input.shape)), plt.axis("off")
    plt.subplot(3,4,11), plt.imshow(get_mean_attention_map(attn3, second_num, input.shape)), plt.axis("off")
    plt.subplot(3,4,12), plt.imshow(get_mean_attention_map(attn4, second_num, input.shape)), plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_name}")

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder',default=None)
    parser.add_argument('--output', type=str, help='Output file or folder',default=None)
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    args = parser.parse_args()
    # assert args.checkpoint.endswith('.ckpt'), \
    #     'You need to provide a .ckpt file as checkpoint'
    # assert args.image_shape is None or len(args.image_shape) == 2, \
    #     'You need to provide a 2-dimensional tuple as shape (H,W)'
    # assert (is_image(args.input) and is_image(args.output)) or \
    #        (not is_image(args.input) and not is_image(args.input)), \
    #     'Input and output must both be images or folders'
    return args


@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half,my_name):
    global path
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_folder=output_file
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)
    my_depth = torch.unsqueeze(pred_inv_depth,dim=0)
    my_depth = my_depth.transpose(1,2).transpose(2,3)
    my_depth = torch.squeeze(my_depth,dim=0)
    my_depth = my_depth.cpu().numpy()

    rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    # Prepare inverse depth
    viz_pred_inv_depth = viz_inv_depth(pred_inv_depth) * 255
    image_out = viz_pred_inv_depth
    output_file = my_name.replace('/','_',4)
    visualize_attention(image, model_wrapper.depth_net,image_out[:, :, ::-1],output_folder+output_file)


def main(args):
    global path
    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    f= open('/home/cv1/hdd/original_attendepth/m.txt','r')
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip().split(" "))

    files = []

    for i in file_list:
        for j in i:
            files.append(j)

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

    # if os.path.isdir(args.input):
    #     # If input file is a folder, search for image files
    #     files = []
    #     for ext in ['png', 'jpg']:
    #         files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
    #     files.sort()
    #     print0('Found {} files'.format(len(files)))
    # else:
    #     # Otherwise, use it as is
    #     files = [args.input]

    # Process each file
    # for fn in files[rank()::world_size()]:
    #     infer_and_save_depth(
    #         fn, args.output, model_wrapper, image_shape, args.half, args.save)
    output_path = '/home/cv1/hdd/original_attendepth/outputs/temp/'
    path = '/home/cv1/hdd/data/datasets/KITTI_raw/'
    config, state_dict = parse_test_file(args.checkpoint)
    model_wrapper.load_state_dict(state_dict)

    for i in files:
        # i = i+'.png'
        file_name = i.split('_')
        i=''
        i = i+file_name[0]+'_'+file_name[1]+'_'+file_name[2]+'/'+file_name[3]+'_'+file_name[4]+'_'+file_name[5]+'_'+file_name[6]+'_'+file_name[7]+'_'+file_name[8]+'/'+file_name[9]+'_'+file_name[10]+'/'+file_name[11]+'/'+file_name[12]+'.png'
        print(i)
        infer_and_save_depth(path+i, output_path, model_wrapper, image_shape, args.half,i)
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
