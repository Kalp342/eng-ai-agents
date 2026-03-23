import cv2
import numpy as np
import os
import glob
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
                     
    kf.P *= 1000.  # High initial uncertainty
    kf.R = np.array([[50., 0.], [0., 50.]])
    kf.Q *= 0.1    # Process noise (Accounts for sudden changes in drone speed/direction)
    
    return kf

def process_tracking(video_directory='./', output_directory='tracking_outputs'):
    model_path = '/workspaces/eng-ai-agents/assignments/assignment-3/best.pt'
    model = YOLO(model_path)
    os.makedirs(output_directory, exist_ok=True)
    
    video_files = glob.glob(os.path.join(video_directory, '*.mp4'))
    for video_idx, video_path in enumerate(video_files, start=1):
        video_name = os.path.basename(video_path)
        print(f"Tracking Video {video_idx}: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup the output video file
        out_path = os.path.join(output_directory, f"tracked_vid_{video_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        kf = initialize_kalman_filter()
        trajectory = []
        is_tracking = False
        missed_frames = 0
        max_missed = 10  # Max frames to predict blindly before dropping the track
        
        frame_idx = 1
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            results = model(frame, verbose=False)
            boxes = results[0].boxes
            drone_detected = len(boxes) > 0
            
            if drone_detected:
                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                
                if not is_tracking:
                    kf.x = np.array([[cx], [cy], [0.], [0.]])
                    is_tracking = True
                    trajectory.clear()
                
                
                kf.predict()
                kf.update(np.array([cx, cy]))
                missed_frames = 0
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, 'Drone', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            elif is_tracking:
                # Drone lost, rely entirely on Kalman filter prediction
                kf.predict()
                missed_frames += 1
                
                if missed_frames > max_missed:
                    # Lost for too long, reset
                    is_tracking = False 
            if is_tracking:

                est_x, est_y = int(kf.x[0, 0]), int(kf.x[1, 0])
                trajectory.append((est_x, est_y))
                cv2.circle(frame, (est_x, est_y), 5, (0, 0, 255), -1)

                if len(trajectory) > 1:
                    for i in range(2, len(trajectory) + 1):
                        cv2.line(frame, trajectory[i-2], trajectory[i-1], (255, 0, 0), 2)
                        

                out.write(frame)
                
            frame_idx += 1
                
        cap.release()
        out.release()
        print(f"Finished tracking! Saved to {out_path}\n")

if __name__ == "__main__":
    process_tracking()