# HITNET-Stereo-Depth-estimation
Python scripts for performing stereo depth estimation using the [HITNET Tensorflow model from Google Research](https://github.com/google-research/google-research/tree/master/hitnet).

![Hitnet stereo depth estimation](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation/blob/main/doc/img/out.jpg)

# Requirements

 * **OpenCV**, **numpy** and **tensorflo**. **pafy** and **youtube-dl** are required for youtube video inference. 
 * For the drivingStereo dataset, download the data from: https://drivingstereo-dataset.github.io/

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
