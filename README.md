# DWMamba
This is the project of paper "DWMamba: A Structure-Aware Adaptive State Space Network for Image Quality Improvement".

# Abstract 
Overcoming visual degradation in challenging imaging scenarios is essential for accurate scene understanding. Although deep learning methods have integrated various perceptual capabilities and achieved remarkable progress, their high computational cost limits practical deployment under resource-constrained conditions. Moreover, when confronted with diverse degradation types, existing methods often fail to effectively model the inconsistent attenuation across color channels and spatial regions. To tackle these challenges, we propose DWMamba, a degradation-aware and weight-efficient Mamba network for image quality enhancement. Specifically, DWMamba introduces an Adaptive State Space Module (ASSM) that employs a dual-stream channel monitoring mechanism and a soft fusion strategy to capture global dependencies. With linear computational complexity, ASSM strengthens the model’s ability to address non-uniform degradations. In addition, by leveraging explicit edge priors and region partitioning as guidance, we design a Structure-guided Residual Fusion (SGRF) module to selectively fuse shallow and deep features, thereby restoring degraded details and enhancing low-light textures. Extensive experiments demonstrate that the proposed network delivers superior qualitative and quantitative performance, with strong generalization to diverse extreme lighting conditions.  The code is available at https://github.com/WindySprint/DWMamba.

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
