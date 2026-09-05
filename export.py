import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

if __name__ == '__main__':
    model = YOLO(r'C:\Users\ASUS\Desktop\ultralytics-improved\runs\exp-2-2-5\weights\best.pt')
    model.export(format='onnx', simplify=True)
    # 导出tensorrt模型，本项目的detect.py不支持用tensorrt导出的模型测试，如需测试请去官方Ultralytics中使用，使用方法也是一样
    # model.export(format='engine', simplify=True) 