from ultralytics import YOLO

def train_car_parts_model():
    print("Loading base YOLOv8 Segmentation model...")
    # We use the '-seg' version because the carparts dataset is a segmentation dataset
    model = YOLO("yolov8n-seg.pt") 
    

    results = model.train(
        data="carparts-seg.yaml", 
        epochs=5, 
        imgsz=640,
        workers = 0,
        device="cuda" # Uses CPU. If your Docker container has GPU access, change to device=0
    )
    
    print("\nTraining complete!")
    print("Your new custom model is saved in the 'runs/segment/train/weights/' folder as 'best.pt'")

if __name__ == "__main__":
    train_car_parts_model()