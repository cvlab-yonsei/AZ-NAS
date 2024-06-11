# Pytorch implementation of AZ-NAS
This repository contains the official implementation of the paper "AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search" presented at CVPR 2024.

For detailed information, please visit the [project website](https://cvlab.yonsei.ac.kr/projects/AZNAS/) or read the [paper on arXiv](https://arxiv.org/abs/2403.19232).

## Requirements
To reproduce our results, we provide `Dockerfile` to build a Docker image. 

Refer to the `Dockerfile` for details about the environment setup.

## Code
Instructions and example code for each search space can be found in the README file within the corresponding folder.

## Acknowledgement
Our implementation is based on several open-source projects. 

We express our gratitude to the authors and contributors of these projects. 

If you use any of their assets, please cite the corresponding papers appropriately:
- [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)
- [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects)
- [ZenNAS](https://github.com/idstcv/ZenNAS) 
- [ZiCo](https://github.com/SLDGroup/ZiCo)
- [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer)
- [TFTAS](https://github.com/decemberzhou/TF_TAS)

## Citation
If you find our work useful in your research, please cite our paper:
```
@inproceedings{lee2024assembling,
  title={{AZ-NAS}: Assembling Zero-Cost Proxies for Network Architecture Search},
  author={Lee, Junghyup and Ham, Bumsub},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
