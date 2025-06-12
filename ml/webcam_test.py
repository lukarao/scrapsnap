import glob
import os
import cv2
from ultralytics import YOLO

NAME = 'v1'
confidence_threshold = 0.25

weights_path = glob.glob(os.path.join(os.path.dirname(__file__), 'train', NAME + '*', 'weights', 'best.pt'))[0]
model = YOLO(weights_path)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if success:
        results = model(frame, conf=confidence_threshold)

        annotated_frame = results[0].plot()

        cv2.imshow('Press Q to quit', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()