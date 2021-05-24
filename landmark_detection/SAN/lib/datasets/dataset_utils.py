##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################

from scipy.ndimage.interpolation import zoom
from utils.file_utils import load_txt_file
import numpy as np
import copy, math

def convert68to49(points):
  points = points.copy()
  assert len(points.shape) == 2 and (points.shape[0] == 3 or points.shape[0] == 2) and points.shape[1] == 68, 'The shape of points is not right : {}'.format(points.shape)
  out = np.ones((68,)).astype('bool')
  out[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,60,64]] = False
  cpoints = points[:, out]
  assert len(cpoints.shape) == 2 and cpoints.shape[1] == 49
  return cpoints

def convert68to51(points):
  points = points.copy()
  assert len(points.shape) == 2 and (points.shape[0] == 3 or points.shape[0] == 2) and points.shape[1] == 68, 'The shape of points is not right : {}'.format(points.shape)
  out = np.ones((68,)).astype('bool')
  out[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]] = False
  cpoints = points[:, out]
  assert len(cpoints.shape) == 2 and cpoints.shape[1] == 51
  return cpoints
