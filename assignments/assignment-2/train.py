import os
import glob
import pandas as pd
from ultralytics import YOLO

def build_video_index(frame_dir="frames", video_id="YcvECxtXoxQ", output_parquet="video_index.parquet"):
    """
    Runs YOLOv8 on extracted frames and saves the detections to a Parquet file.
    """
    print("Loading YOLO model...")
    # NOTE: If you trained the custom car parts model from Step 2, change this to your 
    # 'best.pt' file path! Otherwise, 'yolov8n.pt' will download the standard pre-trained model.
    model = YOLO("/workspaces/eng-ai-agents/runs/segment/train6/weights/best.pt")
    
    # Grab all the images in the frames directory and sort them sequentially
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    
    detections = []
    print(f"Starting inference on {len(frame_files)} frames...")

    for frame_path in frame_files:
        # Extract frame index from the filename (e.g., 'frame_0001.jpg' -> 1)
        filename = os.path.basename(frame_path)
        frame_index = int(filename.replace("frame_", "").replace(".jpg", ""))
        
        # Calculate timestamp (if extracting 1 frame every 5 seconds: frame 1 is 5s)
        # Change the '5' if you used a different extraction fps!
        timestamp_sec = frame_index * 5 
        
        # Run YOLO inference (verbose=False keeps the console clean)
        results = model(frame_path, verbose=False) 
        
        # Loop through the YOLO results and extract the necessary data
        for result in results:
            for box in result.boxes:
                # Convert the tensor values to standard Python types
                coords = box.xyxy[0].tolist() # Returns [x_min, y_min, x_max, y_max]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_label = model.names[class_id]
                
                # Optional: Filter out low-confidence "junk" detections to keep your index clean
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
    
    # Convert our list of dictionaries into a structured Pandas DataFrame
    df = pd.DataFrame(detections)
    
    # Export to Parquet format using the pyarrow engine
    df.to_parquet(output_parquet, engine="pyarrow")
    print(f"Index successfully saved to '{output_parquet}'!")
    
    return df

if __name__ == "__main__":
    # Run the indexer
    df_index = build_video_index()
    
    # Print the first 5 rows to verify it worked
    if not df_index.empty:
        print("\nHere is a peek at your generated index:")
        print(df_index.head())
    else:
        print("\nNo detections were found. (Check if your frames directory is populated).")
