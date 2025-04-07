# CSSF: Cross-Scale Semantic Fusion for Efficient Image Colorization

[![Pytorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

基于跨尺度语义映射的单阶段图像着色模型，在保持语义一致性的同时实现快速色彩迁移。
![image](https://github.com/user-attachments/assets/4422b580-8772-4e13-8c16-ee9e47aaeeb7)
## Contributions
✅ We propose a fast colorization network that can simultaneously perform semantic mapping and image colorization tasks. It not only reasonably embeds the colors of the reference image into the grayscale image but can learn the real-world color distribution from large-scale datasets to perform automatic colorization.

✅ We design a cross-scale feature extraction module to stack multilevel features with different resolutions to generate cross-scale cascaded features and boost feature interactions across scales.

✅ We construct a semantic mapping module to calculate the semantic- associated graph between the reference image and the grayscale image and transfers the color of the reference image to the grayscale image pixel by pixel according to the semantic similarity.

✅ In order to reduce color bleeding, a local color perception loss is addressed and meanwhile a plug-and-play scale-adaptive global color perception loss is constructed.

## 快速开始
### 安装依赖
pip install -r requirements.txt

## Qualitative Comparison
![image](https://github.com/user-attachments/assets/2229e3d1-373e-4780-92a3-886509c0cc67)
![image](https://github.com/user-attachments/assets/272d2e3c-5e79-4db1-93b1-9acb8322a0f7)

## Quantitative Comparison
![image](https://github.com/user-attachments/assets/03bff778-2ad2-407b-aa70-8e844dc645ad)
![image](https://github.com/user-attachments/assets/7daf1f30-4881-4cbe-b308-382b4af14d2a)

## Comparison of parameters and inference speed between state-of-the-art automatic colorization networks
![image](https://github.com/user-attachments/assets/5709f962-1592-4c0f-9889-5b1d39402ad2)
