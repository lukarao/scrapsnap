import os
from ultralytics import YOLO

EPOCHS = 50

if __name__ == '__main__':
    dataset_path = ''

    os.chdir(os.path.join(os.path.dirname(__file__), 'models', 'pretrained'))
    model = YOLO('yolo11n.pt')
    
    model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=EPOCHS, batch=64, imgsz=640, device=0, project=os.path.join(os.path.dirname(__file__), 'train'), name=f'{EPOCHS}_epochs')

    model.val(project=os.path.dirname(__file__))

    #export_path = model.export(format='onnx')
    #os.rename(export_path, os.path.join(os.path.dirname(__file__), 'models', 'final', f'{EPOCHS}_epochs.onnx'))