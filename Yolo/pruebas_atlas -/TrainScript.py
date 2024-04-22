from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
model.train(data = 'C:/Users/nacho/Documents/Programacion/webots_2023/RCJ-2024-Rescue-Simulation-Team-ABC/Yolo_atlas -', epochs=10, imgsz=500)