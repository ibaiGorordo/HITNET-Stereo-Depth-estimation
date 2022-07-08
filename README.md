# HITNET-Stereo-Depth-estimation
Python scripts for performing stereo depth estimation using the [HITNET Tensorflow model from Google Research](https://github.com/google-research/google-research/tree/master/hitnet).

![Hitnet stereo depth estimation](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation/blob/main/doc/img/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

# Requirements

 * **OpenCV**, **numpy** and **tensorflo**. **pafy** (`pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b`) and **youtube-dl** are required for youtube video inference. 
 * For the drivingStereo dataset, download the data from: https://drivingstereo-dataset.github.io/

# Tensorflow models
Download the tensorflow models from the [original repository](https://github.com/google-research/google-research/tree/master/hitnet) and save them into the **[models](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation/tree/main/models)** folder. 

# Examples

 * **Image inference**:
 
 ```
 python imageDepthEstimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoDepthEstimation.py
 ```
 
 * **DrivingStereo dataset inference**:
 
 ```
 python drivingStereoTest.py
 ```
 
  # [Inference video Example](https://youtu.be/ge2iN8Ga4Dg) 
 ![!Hitnet stereo depth estimation on video](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation/blob/main/doc/img/hitnetDepthEstimation.gif)

# References:
* Hitnet model: https://github.com/google-research/google-research/tree/master/hitnet
* DrivingStereo dataset: https://drivingstereo-dataset.github.io/
* Original paper: https://arxiv.org/abs/2007.12140
