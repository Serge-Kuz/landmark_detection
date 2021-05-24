#this aims to avoid `sys.path` changing outside this module
import sys
from os import path as osp

lib_path ='..\\landmark_detection\\landmark_detection'
lib_path =osp.dirname(osp.abspath(lib_path))
sys.path.insert(0, lib_path)

from landmark_detector import LandmarkDetector
import pytest
import cv2


weights_path = osp.join(lib_path, './SAN/snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar')
image_path = osp.join(lib_path, './Image.png')
bbox = (819.27, 432.15, 971.70, 575.87)
img_4_test = cv2.imread(image_path)
img_in_rbg = cv2.cvtColor(img_4_test, cv2.COLOR_BGR2RGB)  



def test_LandmarkDetector_None():
    '''
    Проверка инициализации None
    '''
    landmark_detector = None
    landmark_detector = LandmarkDetector(weights_path, device=None)
    
    assert landmark_detector is not None


def test_LandmarkDetector_cpu():
    '''
    Проверка инициализации cpu
    '''
    landmark_detector = None
    landmark_detector = LandmarkDetector(weights_path, device='cpu')
    
    assert landmark_detector is not None  


def test_LandmarkDetector_cuda():
    '''
    Проверка инициализации cuda
    '''
    landmark_detector = None
    landmark_detector = LandmarkDetector(weights_path, device='cuda')
    
    assert landmark_detector is not None    


def test_preprocess_image():
    '''
    Проверка препроцесса
    '''
    landmark_detector = None
    landmark_detector = LandmarkDetector(weights_path, device='cuda')
    res = landmark_detector._preprocess_image(img_in_rbg, bbox)
    
    assert res[0] is not None  and res[1] is not None


def test_predict():
    '''
    Проверка предикта
    '''
    landmark_detector = None
    landmark_detector = LandmarkDetector(weights_path, device='cuda')
    img_in = dict()
    img_in['image'] = img_in_rbg
    img_in['box'] = bbox
    prediction = landmark_detector.predict(img_in)

    assert prediction['landmarks'].shape[0] == 63  and  prediction[1] is None or prediction['error_message'] == ''

def test_predict_shape():
    '''
    Проверка предикта
    '''
    landmark_detector = None
    landmark_detector = LandmarkDetector(weights_path, device='cuda')
    img_in = dict()
    img_in['image'] = img_in_rbg
    img_in['box'] = bbox
    prediction = landmark_detector.predict(img_in)

    assert len(prediction['landmarks'].shape) == 3  and  prediction[1] is None or prediction['error_message'] == ''   