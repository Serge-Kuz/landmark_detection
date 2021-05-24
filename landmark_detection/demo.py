import argparse
import numpy as np
import cv2
import os
from os import path as osp
import sys
import torch
from pathlib import Path

#print(os.getcwd())
#Добавляем пути (если не прописаны в PYTHONPATH)
from os.path import dirname
sys.path.append(dirname(__file__))

this_dir = osp.dirname(osp.abspath(__file__))
lib_path = osp.join(this_dir, 'SAN\\lib\\')
sys.path.insert(0, lib_path)
lib_path = osp.join(this_dir, 'SAN\\')
sys.path.insert(0, lib_path)


from landmark_detector import LandmarkDetector 
def main(args):

    # #Для запуска из VS Code
    # weights_path = './landmark_detection/SAN/snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar'
    # img_test_path = './landmark_detection/Image.png'
    # bbox = (819.27, 432.15, 971.70, 575.87)   
    # device = 'cuda' 
    # save_path = 'temp_11.jpg'
    #Инициализация входных параметров
    if args.cpu == False:
        device = 'cuda'
    else:
        device = 'cpu'
    if device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA недоступна.'

    snapshot = Path(args.model)
    assert snapshot.exists(), 'Файл модели {:} отсутствует'
    weights_path = args.model

    assert len(args.face) == 4, 'Неверный формат BBox : {:}'.format(args.face)   
    bbox =  args.face

    snapshot = Path(args.image)
    assert snapshot.exists(), 'Файл изображения {:} не найден'
    img_test_path = args.image

    assert len(args.save_path.strip()) > 0, 'Не задан файл для сохранения' 
    save_path = args.save_path  

    def draw_prediction(prediction, img_4_test, save_path):
        '''
        Отображаем landmarks с использованием Opencv
        '''
        for point in prediction:
            cv2.circle(img_4_test,(int(round(point[0])), int(round(point[1]))),1,(0,0,255))
    
        cv2.imshow('img', img_4_test)
        cv2.waitKey(0)
        cv2.imwrite(save_path, img_4_test) 

    

    img_4_test = cv2.imread(img_test_path)
    
    # Преобразовываем в RGB, т.к. сетка обучена на RGB
    # специально НЕ используем numpy, так быстрее и правильнее
    img_in_rbg = cv2.cvtColor(img_4_test, cv2.COLOR_BGR2RGB) 
 
    landmark_detector = LandmarkDetector(weights_path, device=device)
    
    img_in = dict()
    img_in['image'] = img_in_rbg
    img_in['box'] = bbox

    #предсказываем
    prediction = landmark_detector.predict(img_in) 

    if prediction['error_message'].strip() == '':
        draw_prediction(prediction['landmarks'][:, [0, 1]], img_4_test, save_path)
        np.savetxt(f"{save_path}_prediction.csv", prediction['landmarks'], delimiter=",")
    else:
        print(prediction['error_message'])
   

    print('!!! Job is done !!!')
    print('----------------------------------------------------------------')
    print(f'Результаты выполнения:{save_path}- разметка лица на фото')
    print(f'                       {save_path}_prediction.csv - разметка лица np.array ')
    print('----------------------------------------------------------------')



#####################################
#пример командной строки
#python demo.py --image ./Image.png --model ./SAN/snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar --face 819.27 432.15 971.70 575.87 --save_path temp_11.jpg #--cpu
parser = argparse.ArgumentParser(description='Evaluate a single image by the trained model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image',            type=str,   help='The evaluation image path.')
parser.add_argument('--model',            type=str,   help='The snapshot to the saved detector.')
parser.add_argument('--face',  nargs='+', type=float, help='The coordinate [x1,y1,x2,y2] of a face')
parser.add_argument('--save_path',        type=str,   help='The path to save the visualization results')
parser.add_argument('--cpu',     action='store_true', help='Use CPU or not.')
args = parser.parse_args()

main(args)