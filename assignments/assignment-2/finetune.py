from ultralytics import YOLO

def train_car_parts_model():
    print("Loading base YOLOv8 Segmentation model...")
    model = YOLO("yolov8n-seg.pt") 
    

    results = model.train(
        data="carparts-seg.yaml", 
        epochs=30, 
        imgsz=640,
        workers = 0,
        device="cuda" 
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train_car_parts_model()