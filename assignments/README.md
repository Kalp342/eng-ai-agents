# Assignment 3: UAV Drone Detection and Tracking

## Deliverables Links
* **Hugging Face Dataset:** https://huggingface.co/datasets/KalpPatel342/Assignment3
* **Tracking Video 1:** https://www.youtube.com/watch?v=-delECRLarM
* **Tracking Video 2:** https://www.youtube.com/watch?v=58eHiAPmkL8

---

## 1. Dataset Choice and Detector Configuration
**Dataset:** I utilized a drone-specific detection dataset sourced from Roboflow Universe. https://universe.roboflow.com/yolo-bv4k3/drone-detection-rmyxm/dataset/46 . It has a total of 1094 images and has 4 classes

**Detector:** I chose the Ultralytics YOLOv8 Nano (`yolov8n`) architecture due to its strong balance of inference speed and accuracy, making it ideal for video tracking. I fine-tuned the base model on my custom dataset for 15 epochs with an image size of 640. 

## 2. Kalman Filter State Design and Noise Parameters
 I implemented a Kalman filter using the `filterpy` library.
* **State Vector:** The state is represented as a 4D vector: $[x, y, dx, dy]$, where $x$ and $y$ are the 2D pixel coordinates of the bounding box center, and $dx$ and $dy$ represent the velocity of the drone.
* **Measurement Vector:** The YOLO detector only provides position, so the measurement is a 2D vector: $[x, y]$.
* **Noise Parameters:** * The measurement noise matrix ($R$) was set relatively low to reflect high confidence in the YOLO bounding box detections when they occur.
  * The process noise matrix ($Q$) was scaled down slightly to allow for the drone's occasionally erratic changes in direction and velocity, ensuring the tracker wouldn't rigidly overshoot when the drone hovered or turned.

## 3. Failure Cases and Missed Detections
**Handling Missed Detections:** The tracking loop features built-in resilience for missed frames. If the YOLO model fails to detect the drone in a given frame, the Kalman filter relies entirely on its `predict()` step. It uses the last known velocity ($dx$, $dy$) to blindly estimate the drone's continuing path. The tracker is configured to tolerate up to 10 consecutive missed frames before assuming the drone has left the scene and resetting the track.

**Observed Failure Cases:**
* **Microscopic Scale:** When the drone flew extremely far away, its pixel footprint became too small for the YOLO model to confidently differentiate from background noise, leading to dropped tracks.
* **Background Blending/Motion Blur:** Rapid camera movement or instances where the drone crossed over visually complex, similarly-colored backgrounds occasionally caused the detector to miss a frame. The Kalman filter successfully bridged small gaps in these instances.