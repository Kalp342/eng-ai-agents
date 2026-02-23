import pandas as pd
from ultralytics import YOLO
from datasets import load_dataset

def search_video_by_image(query_image, model, index_path="video_index.parquet", sample_rate=5):
    """
    Analyzes a query image using your custom model and finds matching 
    contiguous timestamps in the video index.
    """
    print("Analyzing query image...")
    
    
    results = model(query_image, verbose=False)

   
    query_classes = set()
    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) > 0.50: 
                class_id = int(box.cls[0])
                query_classes.add(model.names[class_id])

    if not query_classes:
        print("No car parts were confidently detected in your query image. Try another image!")
        return

    print(f"Detected the following parts in your query: {list(query_classes)}")

  
    print(f"Searching index '{index_path}' for matches...")
    try:
        df = pd.read_parquet(index_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{index_path}'.")
        return

    matched_rows = df[df['class_label'].isin(query_classes)].copy()
    
    if matched_rows.empty:
        print("No matching frames found in the video.")
        return

    
    matched_rows = matched_rows.sort_values(by=['class_label', 'timestamp_sec'])
    
    results_list = []
    
    for class_label, group in matched_rows.groupby('class_label'):
        
        time_diff = group['timestamp_sec'].diff()
        
        
        block_id = (time_diff > (sample_rate + 1)).cumsum()
        
        
        segments = group.groupby(block_id).agg(
            start_timestamp=('timestamp_sec', 'min'),
            end_timestamp=('timestamp_sec', 'max'),
            number_of_supporting_detections=('timestamp_sec', 'count')
        ).reset_index(drop=True)
        
        
        segments['class_label'] = class_label
        
        results_list.append(segments)
        
    final_segments = pd.concat(results_list, ignore_index=True)
    
    # Sort by the most robust segments (the ones with the most supporting frames)
    final_segments = final_segments.sort_values(by='number_of_supporting_detections', ascending=False)
    
    print("\n Retrieved Contiguous Video Segments:")
    for _, row in final_segments.iterrows():
        start = int(row['start_timestamp'])
        end = int(row['end_timestamp'])
        label = row['class_label']
        count = int(row['number_of_supporting_detections'])
        
        start_fmt = f"{start//60:02d}:{start%60:02d}"
        end_fmt = f"{end//60:02d}:{end%60:02d}"
        
        print(f" ➔ {label}: [{start_fmt} - {end_fmt}] ({count} supporting frames)")

if __name__ == "__main__":
    
    CUSTOM_MODEL_PATH = "/workspaces/eng-ai-agents/assignments/assignment-2/best.pt"
    INDEX_FILE = "/workspaces/eng-ai-agents/video_index.parquet"
    
    print("Loading YOLO model...")
    custom_model = YOLO(CUSTOM_MODEL_PATH)

    print("Loading Hugging Face query dataset...")
    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")
    query_idx = 0
    
    print(f"\nTesting with query image at index {query_idx} from the dataset...")
    test_image = ds[query_idx]["image"]
    
    search_video_by_image(test_image, custom_model, index_path=INDEX_FILE, sample_rate=5)