from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.predict(model="runs/detect/train/weights/best.pt", source="data/validation/images", save=True)
