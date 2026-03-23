import cv2
import os
import glob
from ultralytics import YOLO

def process_videos(video_directory='./', output_directory='detections'):
    model_path = '/workspaces/eng-ai-agents/assignments/assignment-3/best.pt'
    model = YOLO(model_path) 
    
    
    os.makedirs(output_directory, exist_ok=True)
    
    
    video_files = glob.glob(os.path.join(video_directory, '*.mp4'))

    
    for video_idx, video_path in enumerate(video_files, start=1):
        video_name = os.path.basename(video_path)
        print(f"Processing Video {video_idx}: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 1 
        saved_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break 
            results = model(frame, verbose=False) 
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                # Save the frame
                save_path = os.path.join(output_directory, f"vid_{video_idx}_frame_{frame_idx}.jpg")
                cv2.imwrite(save_path, frame)
                saved_count += 1
                
            frame_idx += 1
            
        cap.release()
        print(f"Finished {video_name}. Saved {saved_count} frames with drones.\n")

if __name__ == "__main__":
    process_videos()