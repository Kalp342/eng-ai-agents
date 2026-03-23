import os
from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt') 
    dataset_yaml_path = '/workspaces/eng-ai-agents/assignments/assignment-3/datasets/drone_dataset/data.yaml'

    results = model.train(
        data=dataset_yaml_path,
        epochs=20,             
        imgsz=640,             
        batch=16,              
        device='CUDA',             
        workers=0,
        project='drone_models',
        name='training_run_1'
    )

if __name__ == '__main__':
    train_model()