# MonoFormer  

This repo is reference PyTorch implementation for training and testing depth estimation models using the method described in 
> Deep Digging into the Generalization of Self-supervised Monocular Depth Estimation  
> [Jinwoo Bae](https://jinu0418.tistory.com), Sungho Moon and [Sunghoon Im](https://sunghoonim.github.io/)  
> AAAI 2023 ([arxiv](https://arxiv.org/abs/2205.11083))  

<p align="center">
<img src="https://user-images.githubusercontent.com/33753741/211245370-7333887c-fff0-4af8-839c-b4ef78458474.gif" width="100%" height="100%">
</p>

Our code is based on [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm) of TRI.  
If you find our work useful in your research, please consider citing our papers :  
```
@inproceedings{bae2022monoformer,
  title={Deep Digging into the Generalization of Self-supervised Monocular Depth Estimation},
  author={Bae, Jinwoo and Moon, Sungho and Im, Sunghoon},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Setup
```
conda create -n monoformer python=3.7
git clone https://github.com/sjg02122/MonoFormer.git
pip install -r requirements.txt
```
We ran our experimentss with PyTorch 1.10.0+cu113, Python 3.7, A6000 GPU and Ubuntu 20.04.

## Datasets
You configure your datasets in config.py or other config yaml files. (DATA_PATH means your data root path.). 
In our experiments, we only use the KITTI datasets for training. Other datasets (e.g., ETH3D, DeMoN, and etc.) is used for testing.  

### KITTI
The KITTI (raw) datasets can be downloaded from the [KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php). If you want to download the datasets using command, please use the command of [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm).

