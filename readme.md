#Модуль landmark_detection - тетектирование якорных точек на человеческом лице
  
Реализация базируется на алгоритме SAN из открытого репозитория: https://github.com/D-X-Y/landmark-detection  
Вместо PIL используется OpenCV  

##Установка 
Рекоммендуется создать Python Environment, требуемые пакеты перечислены в  requirements.txt  
Для корректной работы может так же потребоваться устанвить переменную окружения PYTHONPATH:  
....\landmark_detection\landmark_detection\SAN\lib  
....\landmark_detection\landmark_detection\SAN  
....\landmark_detection\landmark_detection  

##Пример запуска демо-приложения:  
Активация Python Environment  
cd ....\landmark_detection\landmark_detection  
Запуск на CPU python demo.py --image ./Image.png --model ./SAN/snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar --face 819.27 432.15 971.70 575.87 --save_path temp_11.jpg --cpu   
  
Запуск на CPU python demo.py --image ./Image.png --model ./SAN/snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar --face 819.27 432.15 971.70 575.87 --save_path temp_11.jpg 
   
В результате получаются 2-а файла:  
Результаты выполнения:temp_11.jpg - разметка лица на фото  
temp_11.jpg_prediction.csv - разметка лица сериализованный np.array
