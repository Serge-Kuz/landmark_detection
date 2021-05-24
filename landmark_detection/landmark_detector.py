import sys
from os import path as osp
import traceback
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


#this aims to avoid `sys.path` changing outside this module
this_dir = osp.dirname(osp.abspath(__file__))
lib_path = osp.join(this_dir, 'SAN\\lib\\')
sys.path.insert(0, lib_path)
lib_path = osp.join(this_dir, 'SAN\\')
sys.path.insert(0, lib_path)
#sys.path.pop(0)

import models
#Модуль transforms переписан с заменой PIL на OpenCV
from san_vision import transforms_noPIL as transforms
from datasets.point_meta import Point_Meta
import cv2



class LandmarkDetector(object):
    ''' Обертка SAN Face Landmark Detector 

    Пример:
        ```
        image_path = './cache_data/cache/test_1.jpg'
        model_path = './snapshots/checkpoint_49.pth.tar'
        face = (819.27, 432.15, 971.70, 575.87)

        from LandmarkDetector import LandmarkDetector
        det = LandmarkDetector(model_path, device)
        locs, scores = det.predict(image_path, face)
        ```
    '''


    def __init__(self, model_path, device=None):
        '''
            module_path: путь к предобученной модели
            device: применерие CUDA -либо строка 'cuda' либо torch.device instance
                warning: допустимо только 'cpu' or 'cuda'
                по умолчанию 'cuda' если доступна
        '''
        assert type(model_path) is str, "model_path - д.б. непустой строкой"
        assert len(model_path.strip()) > 0, "model_path - д.б. непустой строкой"
        assert device is None or device == 'cpu' or device =='cuda' , "device может быть либо 'cpu', либо 'cuda'"

        self._model_path = model_path
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device

        if device == 'cuda':
            torch.backends.cudnn.enabled = True

        snapshot = torch.load(self._model_path, map_location=self._device)
        self._param = snapshot['args']

        self._transform  = transforms.Compose([
            transforms.PreCrop(self._param.pre_crop_expand),
            transforms.TrainScale2WH((self._param.crop_width, self._param.crop_height)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.ToTensor(),

        ])

        self._net = models.__dict__[self._param.arch](self._param.modelconfig, None)
        self._net.train(False).to(self._device)

        weights = models.remove_module_dict(snapshot['state_dict'])
        self._net.load_state_dict(weights)


    def _preprocess_image(self, image, face):
        """
        препроцессинг изображения:
        ■ принимает изображение numpy array,
        ■ возвращает готовый для подачи в сеть pytorch tensor;
        """
        assert type(image) is np.ndarray, "Изображение должно быть np.ndarray"
        
        meta = Point_Meta(self._param.num_pts, None, np.array(face), '', 'custom')
        image, meta = self._transform(image, meta)
        temp_save_wh = meta.temp_save_wh
        cropped_size = torch.IntTensor( [temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]] )

        return image, cropped_size



    def predict(self, img_in): 
        '''
        predict - основной метод для использования модели:
        ■ принимает словарь с полями image (изображение  pytorch tensor) и box
        (лист с координатами лица [x1, y1, x2, y2]),
        ■ возвращает словарь с полями landmarks (numpy array размера 68 x 3 -
        координаты x, y и вероятность p для каждой из 68 точек) и
        error_message (строка с текстом ошибки, если не удалось извлечь
        лэндмарки, пустая - в ином случае)
        '''       
        
        image = img_in['image']
        bbox_face = img_in['box']
        assert type(image) is np.ndarray, "Изображение должно быть np.ndarray"
        h, w, depth = image.shape[::1]
        err_message = ''
        landmarks = None

        image, bb = self._preprocess_image(image, bbox_face)

        try:
            # network forward
            with torch.no_grad():
                inputs = image.unsqueeze(0).to(self._device)
                _, batch_locs, batch_scos, _ = self._net(inputs)

            # obtain the locations on the image in the orignial size
            np_batch_locs, np_batch_scos, cropped_size = batch_locs.cpu().numpy(), batch_scos.cpu().numpy(), bb.numpy()
            locations, scores = np_batch_locs[0,:-1,:], np.expand_dims(np_batch_scos[0,:-1], -1)

            scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2) , cropped_size[1] * 1. / inputs.size(-1)

            locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]

            locations = locations.round().astype(int)
            landmarks = np.concatenate((locations, scores), axis=1)

        except:
            err_message = f'Unexpected Error: {traceback.format_exc()}'

            
        landmarks_res = dict()
        landmarks_res['landmarks'] = landmarks
        landmarks_res['error_message'] = err_message
  
        return landmarks_res       
