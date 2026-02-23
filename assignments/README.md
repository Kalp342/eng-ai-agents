# Image-to-Video Semantic Retrieval via Object Detection

This repository contains a complete pipeline for semantic video retrieval based on object detection. Given a query image of a car exterior component, the system retrieves contiguous video clips where that exact component is visible. 


## How to Run the Pipeline

The project is broken down into four distinct scripts that should be executed sequentially.

### Step 1: Train the Object Detector
Run the fine-tuning script to train the YOLOv8 segmentation model on the car parts dataset.

```
python finetune.py
```
* **What it does:** Loads the base `yolov8n-seg.pt` model and trains it for 30 epochs. 
* **Output:** A customized weights file (typically saved in `runs/segment/train/weights/best.pt`) that will be used for indexing and retrieval. *Note: Ensure the path to `best.pt` in `save.py` and `retrieval.py` matches your local output path.*

### Step 2: Download and Extract Video Frames
Prepare the searchable video corpus.

```
python video.py
```
* **What it does:** Downloads the target YouTube video using `yt-dlp` and uses `ffmpeg` to extract frames.
* **Output:** A new directory called `frames/` containing sampled images (1 frame every 5 seconds).

### Step 3: Build the Semantic Index
Analyze the extracted frames and build a searchable database.

```bash
python save.py
```
* **What it does:** Runs the fine-tuned YOLO model across all images in the `frames/` directory. It calculates bounding boxes, extracts class labels, maps frames to 1-indexed video timestamps, and filters out low-confidence detections (below 0.50).
* **Output:** A structured `video_index.parquet` file. This file serves as the core database for the retrieval step.

### Step 4: Run Semantic Retrieval
Query the system using images to find matching video clips.

```
python retrieval.py
```
* **What it does:** Loads query images from the `aegean-ai/rav4-exterior-images` dataset on Hugging Face. It runs the same YOLO model to identify car parts in the image, searches `video_index.parquet` for matching labels, and groups the results into contiguous temporal blocks.
* **Output:** Prints the start and end timestamps, the targeted class label, and the number of supporting frames for each matching video segment directly to the console.


* **Hugging Face Link:**
https://huggingface.co/datasets/KalpPatel342/Assignment-2

