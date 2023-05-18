import cv2
import numpy as np
from super_gradients.common.object_names import Models
from super_gradients.training import models
from ultralytics import YOLO

from config import SERVER_IP, SERVER_PORT
from streaming.client import RemoteCamera
from utils import intersection_over_union, add_to_queue, update_label

CLASSES_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255),
}


def real_time_inference_v8(model_path, device='cpu'):
    model = YOLO(model_path)
    rc = RemoteCamera(SERVER_IP, SERVER_PORT)
    rc.connect()

    counter = 0

    COUNTER = {i: 0 for i in range(len(model.names))}
    objects = []

    while True:
        img = rc.get_frame()

        if img is None:
            continue

        res = model(img, device=device)

        bboxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = res[0].boxes.conf.cpu().numpy()
        labels = res[0].boxes.cls.cpu().numpy()

        to_delete = []
        # remove classes that overlap
        for i, box in enumerate(bboxes):
            for j, box_ in enumerate(bboxes):
                if i != j and intersection_over_union(box, box_) > 0.8:
                    if confs[i] > confs[j]:
                        to_delete.append(j)
                    else:
                        to_delete.append(i)

        # delete all the boxes that overlap
        bboxes = np.delete(bboxes, to_delete, axis=0)
        confs = np.delete(confs, to_delete, axis=0)
        labels = np.delete(labels, to_delete, axis=0)

        # plot the boxes
        res_plotted = img.copy()

        for (x1, y1, x2, y2), conf, label in zip(bboxes, confs, labels):
            cv2.rectangle(res_plotted, (x1, y1), (x2, y2), CLASSES_COLORS[label], 2)
            cv2.putText(
                res_plotted,
                f"{model.names[label]} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                CLASSES_COLORS[label],
                2
            )

        # for each bbox found
        objects_found_this_frame = set()
        for bbox, label in zip(bboxes, labels):
            obj_found = False
            for obj in objects:
                # if the object is already in the list
                if intersection_over_union(obj["bbox"], bbox) > 0.4:
                    # update the bbox
                    obj["bbox"] = bbox
                    obj_found = True
                    objects_found_this_frame.add(obj["counter"])
                    add_to_queue(obj["label_queue"], label)
                    obj["label"] = update_label(obj["label_queue"])
                    break

            # if the object is not in the list
            if not obj_found:
                # add it
                objects.append({"bbox": bbox, "counter": counter, "label_queue": [label], "label": label})
                objects_found_this_frame.add(counter)
                counter += 1

        # remove objects that were not found in this frame
        new_objects = []

        for obj in objects:
            if obj["counter"] in objects_found_this_frame:
                new_objects.append(obj)
            else:
                COUNTER[obj['label']] += 1

        objects = new_objects

        for obj, conf in zip(objects, confs):
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            cv2.rectangle(res_plotted, (x1, y1), (x2, y2), CLASSES_COLORS[label], 2)
            cv2.putText(
                res_plotted,
                f"{model.names[label]} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                CLASSES_COLORS[label],
                2
            )

            cv2.putText(
                res_plotted,
                f"obj {obj['counter']}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        text = ''
        for i in range(len(model.names)):
            text += f"{model.names[i]}: {COUNTER[i]} {'-' if i != len(model.names) - 1 else ''} "

        cv2.putText(
            res_plotted,
            text,
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

        cv2.imshow("video stream", res_plotted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def real_time_inference_nas(model_path):
    # load model checkpoints/yolo_nas_binary/ckpt_best.pth
    model = models.get(Models.YOLO_NAS_S, checkpoint_path=model_path, num_classes=2)
    rc = RemoteCamera(SERVER_IP, SERVER_PORT)
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
            cv2.putText(
                img,
                f"{class_names[label]} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("video stream", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    real_time_inference_v8('models/yolov8nBinary.pt', device='mps')
