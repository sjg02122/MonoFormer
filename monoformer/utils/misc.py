import matplotlib.pyplot as plt
from numpy.lib.type_check import imag

from .vit import get_mean_attention_map
import torch 
import numpy as np
import random

import wandb

def visualize_attention(input, model,prediction,img):
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
    prediction = prediction.detach().cpu()


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
    a = plt.figure()
    # print(input.squeeze().permute(1,2,0).size())
    plt.subplot(3,4,1), plt.imshow(input.squeeze().permute(1,2,0)), plt.title("Input", fontsize=8), plt.axis("off")
    plt.subplot(3,4,2), plt.imshow(prediction.squeeze().permute(1,2,0)), plt.set_cmap("inferno"), plt.title("Prediction", fontsize=8), plt.axis("off")

    #print(input.size())

    h = [6,12,18,24]
    first_num =1
    # temp = get_mean_attention_map(attn1, first_num, input.shape)
    # print('attn1',temp.shape)
    # temp = get_mean_attention_map(attn2, first_num, input.shape)
    # print('attn2',temp.shape)
    # temp = get_mean_attention_map(attn3, first_num, input.shape)
    # print('attn3',temp.shape)
    # temp = get_mean_attention_map(attn4, first_num, input.shape)
    # print('attn4',temp.shape)
    # upper left
    plt.subplot(345),
    plt.imshow(get_mean_attention_map(attn1, first_num, input.shape)[0])
    plt.ylabel("Upper left corner", fontsize=8)
    plt.title(f"Layer {h[0]}", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])


    plt.subplot(346),
    plt.imshow(get_mean_attention_map(attn2, first_num, input.shape)[0])
    plt.title(f"Layer {h[1]}", fontsize=8)
    plt.axis("off"),

    plt.subplot(347),
    plt.imshow(get_mean_attention_map(attn3, first_num, input.shape)[0])
    plt.title(f"Layer {h[2]}", fontsize=8)
    plt.axis("off"),


    plt.subplot(348),
    plt.imshow(get_mean_attention_map(attn4, first_num, input.shape)[0])
    plt.title(f"Layer {h[3]}", fontsize=8)
    plt.axis("off"),


    # lower right
    second_num = -1
    plt.subplot(3,4,9), plt.imshow(get_mean_attention_map(attn1, second_num, input.shape)[0])
    plt.ylabel("Lower right corner", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])
    images =[]
    plt.subplot(3,4,10), plt.imshow(get_mean_attention_map(attn2, second_num, input.shape)[0]), plt.axis("off")
    plt.subplot(3,4,11), plt.imshow(get_mean_attention_map(attn3, second_num, input.shape)[0]), plt.axis("off")
    plt.subplot(3,4,12), plt.imshow(get_mean_attention_map(attn4, second_num, input.shape)[0]), plt.axis("off")
    plt.tight_layout()
    #plt.show()
    images.append(wandb.Image(a,caption='attn'))
    wandb.log({"attn":images})
    # num = str(round(random.random()*100000))
    # plt.savefig(f"/home/han/packnet-sfm/attn_map/{num}.png")
    # print("====================================================DONE===========================================")

# Copyright 2020 Toyota Research Institute.  All rights reserved.

from monoformer.utils.types import is_list

########################################################################################################################

def filter_dict(dictionary, keywords):
    """
    Returns only the keywords that are part of a dictionary
    Parameters
    ----------
    dictionary : dict
        Dictionary for filtering
    keywords : list of str
        Keywords that will be filtered
    Returns
    -------
    keywords : list of str
        List containing the keywords that are keys in dictionary
    """
    return [key for key in keywords if key in dictionary]

########################################################################################################################

def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n
    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated
    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var

########################################################################################################################

def same_shape(shape1, shape2):
    """
    Checks if two shapes are the same
    Parameters
    ----------
    shape1 : tuple
        First shape
    shape2 : tuple
        Second shape
    Returns
    -------
    flag : bool
        True if both shapes are the same (same length and dimensions)
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

########################################################################################################################
