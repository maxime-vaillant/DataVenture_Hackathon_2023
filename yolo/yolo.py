import cv2
import numpy as np
from matplotlib import pyplot as plt

from ultralytics import YOLO

from streaming.client import RemoteCamera
from roboflow import Roboflow

rf = Roboflow(api_key="Lwoy9YMydW8pVfjPJkxS")
project = rf.workspace().project("bottle-cap-hackathon")
model = project.version(1).model

rc = RemoteCamera("192.168.10.125", 9999)
rc.connect()

frame = rc.get_frame()

rgb_frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)

prediction = model.predict(rgb_frame).save("prediction.jpg")

rc.disconnect()
