import time

import cv2
import numpy as np
from roboflow import Roboflow

from streaming.client import RemoteCamera

rf = Roboflow(api_key="Lwoy9YMydW8pVfjPJkxS")
project = rf.workspace().project("bottle-cap-hackathon")
model = project.version(1).model

print('model loaded')

rc = RemoteCamera("192.168.10.125", 9999)
rc.connect()

print('camera connected')

while True:
    start = time.perf_counter()
    frame = rc.get_frame()

    rgb_frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)

    predictions = model.predict(rgb_frame).json()['predictions']
    stop = time.perf_counter()

    for pred in predictions:
        class_name = pred['class']
        conf = pred['confidence']

        x = int(pred['x'])
        y = int(pred['y'])
        height = int(pred['height'])
        width = int(pred['width'])

        print(f"Class {class_name}, X: {x}, Y: {y}, Confidence: {conf:.2f}, time: {stop - start:.3f}s")

        x1 = x - width // 2
        y1 = y - height // 2
        x2 = x + width // 2
        y2 = y + height // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # draw the label
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
