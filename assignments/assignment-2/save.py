import os
import glob
import pandas as pd
from ultralytics import YOLO

def build_video_index(frame_dir="frames", video_id="YcvECxtXoxQ", output_parquet="video_index.parquet"):
    """
    Runs YOLOv8 on extracted frames and saves the detections to a Parquet file.
    """
    model = YOLO("/workspaces/eng-ai-agents/assignments/assignment-2/best.pt")
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    
    detections = []
    print(f"Starting inference on {len(frame_files)} frames")

    for frame_path in frame_files:
        filename = os.path.basename(frame_path)
        frame_index = int(filename.replace("frame_", "").replace(".jpg", ""))
        timestamp_sec = frame_index * 5 
        results = model(frame_path, verbose=False) 
        for result in results:
            for box in result.boxes:
                
                coords = box.xyxy[0].tolist() 
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_label = model.names[class_id]
                
                if conf > 0.50:
                    detections.append({
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "class_label": class_label,
                        "bounding_box": coords,
                        "confidence_score": conf
                    })
                    
    print(f"Total valid detections found: {len(detections)}")
    
    df = pd.DataFrame(detections)
    df.to_parquet(output_parquet, engine="pyarrow")
    print(f"Index successfully saved to '{output_parquet}'!")
    
    return df

if __name__ == "__main__":
    # Run the indexer
    df_index = build_video_index(frame_dir="/workspaces/eng-ai-agents/frames")
    
    if not df_index.empty:
        print("\nHere is a peek at your generated index:")
        print(df_index.head())
    else:
        print("\nNo detections were found. (Check if your frames directory is populated).")
