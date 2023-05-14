import time

import numpy as np
from super_gradients.training import models
from super_gradients.common.object_names import Models
from streaming.client import RemoteCamera2 as RemoteCamera
from matplotlib import pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO


def IoU(box1, box2) -> float:
    """
    Computes the IoU between two bounding boxes
    :param box1: First bounding box (xyxy)
    :param box2: Second bounding box (xyxy)
    :return: IoU
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except ZeroDivisionError:
        iou = 0.0

    # return the intersection over union value
    return iou

CLASSES_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255),
}

def real_time_inference_v8():
    model = YOLO('/Users/pierreadorni/Downloads/yolov8nMulti25.pt')
    rc = RemoteCamera("192.168.10.125", 9999)
    rc.connect()

    counter = 0
    objects = []
    while True:
        t0 = time.time()
        img = rc.get_frame()
        t1 = time.time()
        print(f"Frame received in {t1 - t0} seconds")
        # img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        if img is None:
            continue

        res = model(img, device="mps")

        bboxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = res[0].boxes.conf.cpu().numpy()
        labels = res[0].boxes.cls.cpu().numpy()
        # label names
        label_names = [model.names[i] for i in labels]

        toDelete = []
        # remove classes that overlap
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            for j, (x1_, y1_, x2_, y2_) in enumerate(bboxes):
                if i != j and i not in toDelete and j not in toDelete and IoU([x1, y1, x2, y2], [x1_, y1_, x2_, y2_]) > 0.4:
                    if confs[i] > confs[j]:
                        # remove from lists
                        toDelete.append(j)
                    else:
                        # remove from lists
                        toDelete.append(i)
        # delete all the boxes that overlap
        bboxes = np.delete(bboxes, toDelete, axis=0)
        confs = np.delete(confs, toDelete, axis=0)
        labels = np.delete(labels, toDelete, axis=0)

        # plot the boxes
        res_plotted = img.copy()
        for (x1, y1, x2, y2), conf, label in zip(bboxes, confs, labels):
            cv2.rectangle(res_plotted, (x1, y1), (x2, y2), CLASSES_COLORS[label], 2)
            cv2.putText(res_plotted, f"{model.names[label]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLASSES_COLORS[label], 2)

        # for each bbox found
        objects_found_this_frame = set()
        for x1, y1, x2, y2 in res[0].boxes.xyxy:
            obj_found = False
            for obj in objects:
                # if the object is already in the list
                if IoU(obj["bbox"], [x1,y1,x2,y2]) > 0.4:
                    # update the bbox
                    obj["bbox"] = [x1,y1,x2,y2]
                    obj_found = True
                    objects_found_this_frame.add(obj["counter"])
                    break
            # if the object is not in the list
            if not obj_found:
                # add it
                objects.append({"bbox": [x1,y1,x2,y2], "counter": counter})
                objects_found_this_frame.add(counter)
                counter += 1

        # remove objects that were not found in this frame
        objects = [obj for obj in objects if obj["counter"] in objects_found_this_frame]

        # plot labels for each object
        for obj in objects:
            x1, y1, x2, y2 = [int(o.item()) for o in obj["bbox"]]
            print("obj", obj["counter"], "at", obj["bbox"])
            cv2.putText(res_plotted, f"obj {obj['counter']}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("video stream", res_plotted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def real_time_inference():
    # load model checkpoints/yolo_nas_binary/ckpt_best.pth
    model = models.get(Models.YOLO_NAS_S, checkpoint_path='/Users/pierreadorni/Downloads/ckpt_best.pth', num_classes=2)
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