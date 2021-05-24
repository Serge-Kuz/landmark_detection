##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
###                                                        ###
### Modified by Sergey Kuzmenko                            ###
### PIL library replaced to OpenCV                         ### 
###                                                        ###
##############################################################

from __future__ import division
import torch
import sys, math, random
import cv2
import numpy as np
import numbers
import types
import collections


class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img, points):
    for t in self.transforms:
      img, points = t(img, points)
    return img, points


class TrainScale2WH(object):
  """Rescale the input  numpy array to the given size.
  Args:
    size (sequence or int): Desired output size. If size is a sequence like
      (w, h), output size will be matched to this. If size is an int,
      smaller edge of the image will be matched to this number.
      i.e, if height > width, then image will be rescaled to
      (size * height / width, size)
    interpolation (int, optional): Desired interpolation. Default is
      ``cv2.INTER_AREA``
  """

  def __init__(self, target_size, interpolation=cv2.INTER_AREA):
    assert isinstance(target_size, tuple) or isinstance(target_size, list), 'The type of target_size is not right : {}'.format(target_size)
    assert len(target_size) == 2, 'The length of target_size is not right : {}'.format(target_size)
    assert isinstance(target_size[0], int) and isinstance(target_size[1], int), 'The type of target_size is not right : {}'.format(target_size)
    self.target_size   = target_size
    self.interpolation = interpolation

  def __call__(self, imgs, point_meta):
    """
    Args:
      img ( numpy array): Image to be scaled.
      points 3 * N numpy.ndarray [x, y, visiable]
    Returns:
       numpy array: Rescaled image.
    """

    # cv2.imshow('TrainScale2WH', imgs)
    # cv2.waitKey(0)

    point_meta = point_meta.copy()

    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]
    
    h, w, depth = imgs[0].shape[::1]
    ow, oh = self.target_size[0], self.target_size[1]
    point_meta.apply_scale( [ow*1./w, oh*1./h] )

    imgs = [ cv2.resize(img, (oh, ow), self.interpolation) for img in imgs ]
    if is_list == False: imgs = imgs[0]

    # cv2.imshow('TrainScale2WH', imgs)
    # cv2.waitKey(0)

    return imgs, point_meta

class Normalize(object):
  """Normalize an tensor image with mean and standard deviation.
  Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  Args:
    mean (sequence): Sequence of means for R, G, B channels respecitvely.
    std (sequence): Sequence of standard deviations for R, G, B channels
      respecitvely.
  """

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensors, points):
    """
    Args:
      tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
      Tensor: Normalized image.
    """

    if isinstance(tensors, list): is_list = True
    else:                         is_list, tensors = False, [tensors]

    for tensor in tensors:
      for t, m, s in zip(tensor, self.mean, self.std):
        #cv2.normalize(pic, pic, 1.0, -1.0, cv2.NORM_MINMAX)
        t = (t - m) / s
    
    if is_list == False: tensors = tensors[0]

    return tensors, points


class ToTensor(object):
  """Convert a `` numpy array`` or  to tensor.
  Converts a numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """

  def __call__(self, pics, points):
    """
    Args:
      pic (numpy.ndarray): Image to be converted to tensor.
      points 3 * N numpy.ndarray [x, y, visiable] or Point_Meta
    Returns:
      Tensor: Converted image.
    """
    ## add to support list
    if isinstance(pics, list): is_list = True
    else:                      is_list, pics = False, [pics]

    returned = []
    for pic in pics:
      if isinstance(pic, np.ndarray):

        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        returned.append( img.float().div(255) )
        continue

    if is_list == False:
      assert len(returned) == 1, 'For non-list data, length of answer must be one not {}'.format(len(returned))
      returned = returned[0]

    return returned, points





class PreCrop(object):
  """Crops the given  numpy array at the center.

  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (w, h), a square crop (size, size) is
      made.
  """

  def __init__(self, expand_ratio):
    assert expand_ratio is None or isinstance(expand_ratio, numbers.Number), 'The expand_ratio should not be {}'.format(expand_ratio)
    if expand_ratio is None:
      self.expand_ratio = 0
    else:
      self.expand_ratio = expand_ratio
    assert self.expand_ratio >= 0, 'The expand_ratio should not be {}'.format(expand_ratio)

  def __call__(self, imgs, point_meta):
 
    if isinstance(imgs, list): is_list = True
    else:                      is_list, imgs = False, [imgs]

    #w, h = imgs[0].size
    h, w, depth = imgs[0].shape[::1]

    box = point_meta.get_box().tolist()
    face_ex_w, face_ex_h = (box[2] - box[0]) * self.expand_ratio, (box[3] - box[1]) * self.expand_ratio
    x1, y1 = int(max(math.floor(box[0]-face_ex_w), 0)), int(max(math.floor(box[1]-face_ex_h), 0))
    x2, y2 = int(min(math.ceil(box[2]+face_ex_w), w)), int(min(math.ceil(box[3]+face_ex_h), h))
    
    #crop image
    imgs = [ img[y1:y2, x1:x2] for img in imgs ]

    h, w, depth = imgs[0].shape[::1]
    point_meta.set_precrop_wh( w, h, x1, y1, x2, y2)
    point_meta.apply_offset(-x1, -y1)

    point_meta.apply_bound(w, h)

    if is_list == False: imgs = imgs[0]

    # cv2.imshow('precrop', imgs)
    # cv2.waitKey(0)

    return imgs, point_meta


