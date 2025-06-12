import os
from ultralytics import YOLO

NAME = 'test_1'

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), 'models', 'pretrained'))

    model = YOLO(os.path.join(os.path.dirname(__file__), 'models', 'pretrained', 'yolo11n-seg.pt'))

    model.train(
        data=os.path.join(os.path.dirname(__file__), 'dataset', 'yolo', 'data.yaml'),
        epochs=10,
        batch=64,
        imgsz=640,
        project=os.path.join(os.path.dirname(__file__), 'train'),
        name=NAME + '_')

    model.val(project=os.path.join(os.path.dirname(__file__), 'val'))

    export_path = model.export(format='onnx')

    if 'test' in NAME:
        os.rename(export_path, os.path.join(os.path.dirname(__file__), 'models', 'test', NAME + '.onnx'))
    else:
        os.rename(export_path, os.path.join(os.path.dirname(__file__), 'models', 'final', NAME + '.onnx'))