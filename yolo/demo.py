import numpy as np
from super_gradients.training import models
from super_gradients.common.object_names import Models
from streaming.client import RemoteCamera
from matplotlib import pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO


def real_time_inference_v8():
    model = YOLO('/Users/pierreadorni/Downloads/best.onnx')
    rc = RemoteCamera("192.168.10.125", 9999)
    rc.connect()

    while True:
        img = rc.get_frame()
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        res = model(img)
        res_plotted = res[0].plot()
        cv2.imshow("video stream", res_plotted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def real_time_inference():
    # load model checkpoints/yolo_nas_binary/ckpt_best.pth
    model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
    # model = YOLO('/Users/pierreadorni/Downloads/best.pt')
    rc = RemoteCamera("192.168.10.125", 9999)
    rc.connect()

    while True:
        img = rc.get_frame()
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        results = [r for r in model.predict([img], conf=0.7)][0]
        class_names = results.class_names
        labels = results.prediction.labels.astype(int)
        confidence = results.prediction.confidence
        bboxes = results.prediction.bboxes_xyxy

        for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
            print(f"Found {class_names[label]} with confidence {conf} at {bbox}")
            # draw the bounding box
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw the label
            cv2.putText(img, f"{class_names[label]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("video stream", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    real_time_inference_v8()