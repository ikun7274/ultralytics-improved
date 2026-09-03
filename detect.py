import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\ASUS\Desktop\ultralytics-improved\runs\exp-2-2-5\weights\best.pt') # select your model.pt path
    model.predict(source=r'C:\Users\ASUS\Desktop\base_0_0\images\val',
                  conf=0.25,
                  project=r'C:\Users\ASUS\Desktop\ultralytics-improved\runs',
                  name='exp',
                  save=True,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )