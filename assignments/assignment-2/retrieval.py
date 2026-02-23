import pandas as pd
from ultralytics import YOLO

def search_video_by_image(query_image_path, model_path, index_path="video_index.parquet"):
    """
    Analyzes a query image using your custom model and finds matching timestamps in the video index.
    """
    print(f"Analyzing query image: '{query_image_path}'...")
    
    # 1. Load YOUR custom-trained model (Make sure the path matches your train folder!)
    model = YOLO(model_path)
    
    # Run inference on the query image
    results = model(query_image_path, verbose=False)

    # 2. Extract unique car parts detected in the query image
    query_classes = set()
    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) > 0.50: # Use standard confidence threshold
                class_id = int(box.cls[0])
                query_classes.add(model.names[class_id])

    if not query_classes:
        print("No car parts were confidently detected in your query image. Try another image!")
        return

    print(f"✅ Detected the following parts in your query: {list(query_classes)}")

    # 3. Load your Parquet index
    print(f"Searching index '{index_path}' for matches...")
    try:
        df = pd.read_parquet(index_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{index_path}'.")
        return

    # 4. Filter the index for matching frames
    matched_rows = df[df['class_label'].isin(query_classes)]
    
    if matched_rows.empty:
        print("No matching frames found in the video.")
        return

    # Count how many matching objects appear in each frame
    match_counts = matched_rows.groupby(['timestamp_sec', 'frame_index'])['class_label'].nunique().reset_index()
    match_counts = match_counts.sort_values(by='class_label', ascending=False)
    
    # 5. Print the top 3 results
    print("\n🎬 Top Matching Video Timestamps:")
    for _, row in match_counts.head(3).iterrows():
        minutes, seconds = divmod(int(row['timestamp_sec']), 60)
        timestamp_formatted = f"{minutes:02d}:{seconds:02d}"
        frame_num = int(row['frame_index'])
        
        print(f" ➔ Timestamp {timestamp_formatted} (Frame {frame_num})")

if __name__ == "__main__":
    # UPDATE THIS PATH to your actual best.pt file (e.g., train6, train4, etc.)
    CUSTOM_MODEL_PATH = "/workspaces/eng-ai-agents/runs/segment/train6/weights/best.pt"

    INDEX_FILE = "/workspaces/eng-ai-agents/video_index.parquet"
    
    # Let's test it on one of the frames from your video!
    TEST_QUERY_IMAGE = "/workspaces/eng-ai-agents/frames/frame_0002.jpg"
    
    search_video_by_image(TEST_QUERY_IMAGE, CUSTOM_MODEL_PATH, index_path=INDEX_FILE)