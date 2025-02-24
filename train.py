from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(data = "data.yaml", imgsz = 720, batch = 8, epochs=50, workers = 0, device="cpu")

model.predict(model="runs/detect/train/weights/best.pt", source="data/validation/images", save=True)