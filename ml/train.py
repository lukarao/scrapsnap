import os
import kagglehub
from ultralytics import YOLO

EPOCHS = 50

if __name__ == '__main__':
    # download dataset
    dataset_path = kagglehub.dataset_download('vencerlanz09/taco-dataset-yolo-format')

    # setup model
    os.chdir(os.path.join(os.path.dirname(__file__), 'models', 'pretrained'))
    model = YOLO('yolo11n.pt')
    
    # train model
    model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=EPOCHS, batch=64, imgsz=640, device=0, project=os.path.join(os.path.dirname(__file__), 'train'), name=f'{EPOCHS}_epochs')

    # validate model
    model.val(project=os.path.dirname(__file__))

    # test model on test.jpg
    results = model(os.path.join(os.path.dirname(__file__), 'test', 'test.jpg'))
    results[0].save(filename=os.path.join(os.path.dirname(__file__), 'test', f'{EPOCHS}_epochs.png'))

    # export model
    #export_path = model.export(format='onnx')
    #os.rename(export_path, os.path.join(os.path.dirname(__file__), 'models', 'final', f'{EPOCHS}_epochs.onnx'))