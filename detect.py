import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\ASUS\Desktop\Datasets\路面交安设施检测\训练记录\train\yolo26s\base_0_0+base_1_0\exp\weights\best.pt') # select your model.pt path
    model.predict(source=r'C:\Users\ASUS\Desktop\Datasets\路面交安设施检测\test_images',
                #   conf=0.25,
                  project=r'C:\Users\ASUS\Desktop\Datasets\路面交安设施检测\test_images\result',
                  name='exp',
                  save=True,
                  imgsz=1280,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )