# DWMamba
This is the project of paper "DWMamba: A Fast, Robust, and Adaptive Mamba Network  for Underwater Image Enhancement".

# Abstract 
Overcoming visual degradation in extreme environments is essential for improving underwater scene comprehension. While deep learning methods have integrated various perceptual capabilities and made significant progress, the high computational costs hinder their practical application in resource-limited underwater conditions. Additionally, when faced with diverse underwater degradations, existing methods lack effective modeling mechanisms to address the inconsistent attenuation across different color channels and space regions. To address these issues, we propose a lightweight and multi-scenario adapted network, DWMamba. Specifically, DWMamba introduces an innovative Adaptive State Space Module (ASSM) that uses a channel monitoring mechanism and a soft fusion strategy to capture global dependencies. By maintaining linear complexity, ASSM enhances the model's potential for handling non-uniform underwater degradation. Furthermore, leveraging explicit edge priors and region partitioning as cues, we design a Structure-guided Residual Fusion module (SGRF) to fuse shallow and deep features in a targeted manner, effectively enhancing degraded details and low-light textures. Extensive experiments demonstrate the impressive qualitative enhancement and quantitative performance of the proposed model. For example, DWMamba exceeds PUGAN in enhancement performance while reducing FLOPs by 93%, and exhibits excellent generalization in various extreme lighting conditions. Our code and model are available at https://github.com/WindySprint/DWMamba.

## Environment
```
1. Python 3.10.13
2. PyTorch 2.1.1
3. Torchvision 0.16.1
4. OpenCV-Python 4.8.1.78
5. NumPy 1.26.3
6. mamba-ssm 1.2.0.post1
7. opencv-python 4.9.0.80
```

## Test
```
1. Clone repo
2. Put the images in your folder path A
3. Change 'ori_images_path' to A and 'result_path' in test.py
4. Run test.py
5. Find results in 'result_path'
```

## Train
```
1. Put the orignal images and the GT images in your folder path A and B
2. Change 'ori_images_path' to A and 'enhan_images_path' to B in train.py
3. Change 'net_name' in train.py
4. Run train.py
5. Find trained net in 'checkpoint_path'/'net_name'
```

## Contact
If you have any questions, please contact: Zhixiong Huang(hzxcyanwind@mail.dlut.edu.cn)

