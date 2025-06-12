import os
from ultralytics import YOLO

EPOCHS = 50

model = YOLO('yolo11n.pt')

results = model(os.path.join(os.path.dirname(__file__), 'models', 'final', f'{EPOCHS}_epochs.onnx'))
results[0].save(filename=os.path.join(os.path.dirname(__file__), 'test', f'{EPOCHS}_epochs.png'))