from enum import Enum
import tensorflow as tf
import numpy as np
import cv2
import urllib
from dataclasses import dataclass

class ModelType(Enum):
	eth3d = 0
	middlebury = 1
	flyingthings = 2

@dataclass
class CameraConfig:
    baseline: float
    f: float

def wrap_frozen_graph(graph_def, inputs, outputs):
	def _imports_graph_def():
		tf.compat.v1.import_graph_def(graph_def, name="")
	wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
	import_graph = wrapped_import.graph
	return wrapped_import.prune(
		tf.nest.map_structure(import_graph.as_graph_element, inputs),
		tf.nest.map_structure(import_graph.as_graph_element, outputs))

def draw_disparity(disparity_map):

	disparity_map = disparity_map.astype(np.uint8)
	norm_disparity_map = (255*((disparity_map-np.min(disparity_map))/(np.max(disparity_map) - np.min(disparity_map))))
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

def draw_depth(depth_map, max_dist):
	
	norm_depth_map = 255*(1-depth_map/max_dist)
	norm_depth_map[norm_depth_map < 0] =0
	norm_depth_map[depth_map == 0] =0

	return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

def load_img(url):
	req = urllib.request.urlopen(url)
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	return cv2.imdecode(arr, -1) # 'Load it as it is'


