# TransFusion-Pose

TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation     
Haoyu Ma, Liangjian Chen, Deying Kong, Zhe Wang, Xingwei Liu, Hao Tang, Xiangyi Yan, Yusheng Xie, Shih-Yao Lin and Xiaohui Xie     
In BMVC 2021  
[[Paper]](https://www.bmvc2021-virtualconference.com/assets/papers/0016.pdf)    [[Video]](https://www.bmvc2021-virtualconference.com/conference/papers/paper_0016.html)     


## Overview
* We propose the **TransFusion**, which apply the transformer architecture to multi-view 3D human pose estimation  
* We propose the epipolar field, a novel and more general form of epipolar line. It readily integrates with the transformer through our proposed geometry positional encoding to encode the 3D relationships among different views.   
* Extensive experiments are conducted to demonstrate that our TransFusion outperforms previous fusion methods on both Human 3.6M and SkiPose datasets, but requires substantially fewer parameters.  



## Installation

1. Clone this repo, and we'll call the directory that you cloned multiview-pose as ${POSE_ROOT}   
~~~
git clone git@github.com:HowieMa/TransFusion-Pose.git
~~~

2. Install dependencies. 
~~~
pip install -r requirements.txt
~~~

3. Download TransPose models pretrained on COCO. 
~~~
wget https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc3_d256_h1024_mh8.pth
~~~
You can also download it from the official website of [TransPose](https://github.com/yangsenius/TransPose)

Please download them under ${POSE_ROOT}/models, and make them look like this:
~~~
${POSE_ROOT}/models
└── pytorch
    └── coco
        └── tp_r_256x192_enc3_d256_h1024_mh8.pth
~~~



## Data preparation
#### Human 3.6M
For Human36M data, please follow [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to prepare images and annotations.

#### Ski-Pose
For Ski-Pose, please follow the instruction from their [website](https://www.epfl.ch/labs/cvlab/data/ski-poseptz-dataset/) to obtain the dataset.    
Once you download the **Ski-PosePTZ-CameraDataset-png.zip** and **ski_centers.csv**, unzip them and put into the same folder, named as ${SKI_ROOT}.    
Run `python data/preprocess_skipose.py ${SKI_ROOT}` to format it.   


Your folder should look like this:
~~~
${POSE_ROOT}
|-- data
|-- |-- h36m
    |-- |-- annot
        |   |-- h36m_train.pkl
        |   |-- h36m_validation.pkl
        |-- images
            |-- s_01_act_02_subact_01_ca_01 
            |-- s_01_act_02_subact_01_ca_02

|-- |-- preprocess_skipose.py
|-- |-- skipose  
    |-- |-- annot
        |   |-- ski_train.pkl
        |   |-- ski_validation.pkl
        |-- images
            |-- seq_103 
            |-- seq_103
~~~


## Training and Testing
#### Human 3.6M
~~~
# Training
python run/pose2d/train.py --cfg experiments-local/h36m/transpose/256_fusion_enc3_GPE.yaml --gpus 0,1,2,3

# Evaluation (2D)
python run/pose2d/valid.py --cfg experiments-local/h36m/transpose/256_fusion_enc3_GPE.yaml --gpus 0,1,2,3  

# Evaluation (3D)
python run/pose3d/estimate_tri.py --cfg experiments-local/h36m/transpose/256_fusion_enc3_GPE.yaml
~~~

#### Ski-Pose
~~~
# Training
python run/pose2d/train.py --cfg experiments-local/skipose/transpose/256_fusion_enc3_GPE.yaml --gpus 0,1,2,3

# Evaluation (2D)
python run/pose2d/valid.py --cfg experiments-local/skipose/transpose/256_fusion_enc3_GPE.yaml --gpus 0,1,2,3

# Evaluation (3D)
python run/pose3d/estimate_tri.py --cfg experiments-local/skipose/transpose/256_fusion_enc3_GPE.yaml
~~~

Our trained models can be downloaded from [here](https://drive.google.com/file/d/1XlzDZAsQCzvQkwOCIZiLgAP6jQ_RZGTg/view?usp=sharing)   


## Citation
If you find our code helps your research, please cite the paper:

~~~
@inproceedings{ma2021transfusion,
  title={TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation},
  author={Ma, Haoyu and Chen, Liangjian and Kong, Deying and Wang, Zhe and Liu, Xingwei and Tang, Hao and Yan, Xiangyi and Xie, Yusheng and Lin, Shih-Yao and Xie, Xiaohui},
  booktitle={British Machine Vision Conference},
  year={2021}
}
~~~


## Acknowledgement
* [Cross-view Fusion](https://github.com/microsoft/multiview-human-pose-estimation-pytorch)
* [TransPose](https://github.com/yangsenius/TransPose)
* [Epipolar Transformer](https://github.com/yihui-he/epipolar-transformers)  


