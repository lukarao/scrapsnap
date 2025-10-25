import os

import cv2

from ultralytics import YOLO

NAME = 'v3'
confidence_threshold = 0.7

model_path = os.path.join(os.path.dirname(__file__), 'models', 'final', NAME + '.onnx')
model = YOLO(model_path, task='detect')

cap = cv2.VideoCapture(0)

while True:
	success, frame = cap.read()

	if success:
		results = model(frame, conf=confidence_threshold, max_det=1)

		annotated_frame = results[0].plot()

		cv2.imshow('Press Q to quit', annotated_frame)

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
