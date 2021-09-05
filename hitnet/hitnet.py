import tensorflow as tf
import numpy as np
import time
import cv2
from hitnet.utils_hitnet import *


drivingStereo_config = CameraConfig(0.546, 1000)

class HitNet():

	def __init__(self, model_path, model_type=ModelType.eth3d, camera_config=drivingStereo_config):

		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0
		self.camera_config = camera_config

		# Initialize model
		self.model = self.initialize_model(model_path, model_type)

	def __call__(self, left_img, right_img):

		return self.estimate_disparity(left_img, right_img)

	def initialize_model(self, model_path, model_type):

		self.model_type = model_type

		with tf.io.gfile.GFile(model_path, "rb") as f:
			graph_def = tf.compat.v1.GraphDef()
			loaded = graph_def.ParseFromString(f.read())

		# Wrap frozen graph to ConcreteFunctions
		if self.model_type == ModelType.flyingthings:
			model = wrap_frozen_graph(graph_def=graph_def,
										inputs="input:0",
										outputs=["reference_output_disparity:0","secondary_output_disparity:0"])

		else:
			model = wrap_frozen_graph(graph_def=graph_def,
										inputs="input:0",
										outputs="reference_output_disparity:0")

		return model

	def estimate_disparity(self, left_img, right_img):

		input_tensor = self.prepare_input(left_img, right_img)

		# Perform inference on the image
		if self.model_type == ModelType.flyingthings:
			left_disparity, right_disparity = self.inference(input_tensor)
			self.disparity_map = left_disparity
		else:
			self.disparity_map = self.inference(input_tensor)

		return self.disparity_map

	def get_depth(self):
		return self.camera_config.f*self.camera_config.baseline/self.disparity_map

	def prepare_input(self, left_img, right_img):

		if (self.model_type == ModelType.eth3d):

			# Shape (1, None, None, 2)
			left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
			right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

			left_img = np.expand_dims(left_img,2)
			right_img = np.expand_dims(right_img,2)
			combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
		else:
			# Shape (1, None, None, 6)
			left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
			right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
			combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0


		return tf.convert_to_tensor(np.expand_dims(combined_img, 0), dtype=tf.float32)
		
	def inference(self, input_tensor):
		output = self.model(input_tensor)

		return np.squeeze(output)



	






