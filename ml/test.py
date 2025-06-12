import glob
import os
from ultralytics import YOLO

NAME = 'test_2'

weights_path = glob.glob(os.path.join(os.path.dirname(__file__), 'train', NAME + '*', 'weights', 'best.pt'))[0]
model = YOLO(weights_path)

results = model(os.path.join(os.path.dirname(__file__), 'test', 'test.jpg'))
results[0].save(filename=os.path.join(os.path.dirname(__file__), 'test', NAME + '.png'))