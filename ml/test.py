import os
from ultralytics import YOLO

NAME = 'test_1'

model = YOLO(os.path.join(os.path.dirname(__file__), 'train', NAME + '_4', 'weights', 'best.pt'))

results = model(os.path.join(os.path.dirname(__file__), 'test', 'test.jpg'))
results[0].save(filename=os.path.join(os.path.dirname(__file__), 'test', NAME + '.png'))