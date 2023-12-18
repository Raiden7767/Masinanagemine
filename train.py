from ultralytics import YOLO
model = YOLO('yolov8n.pt')

results = model.train(
data = 'datasets\dataset\data.yaml',
imgsz = 640,
epochs = 50,
batch = 10,
name = 'yolov8n_custom'
)