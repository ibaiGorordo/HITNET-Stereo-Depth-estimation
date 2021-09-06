import cv2
import tensorflow as tf
import numpy as np

from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img

# Select model type
# model_type = ModelType.middlebury
# model_type = ModelType.flyingthings
model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400.pb"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl.pb"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d.pb"


# Initialize model
hitnet_depth = HitNet(model_path, model_type)

# Load images
left_img = load_img("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
right_img = load_img("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

# Estimate the depth
disparity_map = hitnet_depth(left_img, right_img)

color_disparity = draw_disparity(disparity_map)
cobined_image = np.hstack((left_img, right_img, color_disparity))

cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
cv2.imshow("Estimated disparity", cobined_image)
cv2.waitKey(0)

cv2.imwrite("out.jpg", cobined_image)

cv2.destroyAllWindows()