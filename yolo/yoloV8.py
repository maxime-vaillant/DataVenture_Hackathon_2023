from ultralytics import YOLO

# model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')

model.train(data='datasets/data.yaml', epochs=3)
metrics = model.val(data='datasets/valid')
results = model.test('datasets/test')
success = model.export(format='onnx')
