import subprocess
import os

def download_and_extract_frames(video_url, output_video="input_video.mp4", frame_dir="frames", fps="1/5"):
    """
    Downloads a YouTube video and extracts frames at a specified FPS.
    """
    # Clean up any broken leftover files from previous failed runs
    for file in [output_video, output_video + ".temp", output_video + ".part"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed leftover file: {file}")

    # 1. Download the video using yt-dlp
    print(f"Downloading video from: {video_url}...")
    download_cmd = [
        "yt-dlp",
        # Changed: Only download the video stream (no audio merge required)
        "-f", "bestvideo[ext=mp4]/best[ext=mp4]", 
        # Changed: Prevent yt-dlp from using temporary .part files
        "--no-part", 
        "-o", output_video,
        video_url
    ]
    
    try:
        subprocess.run(download_cmd, check=True)
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        return

    # 2. Extract frames using ffmpeg
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
        
    print(f"Extracting frames at {fps} frames per second...")
    extract_cmd = [
        "ffmpeg",
        "-i", output_video,
        "-vf", f"fps={fps}",
        os.path.join(frame_dir, "frame_%04d.jpg")
    ]
    
    try:
        subprocess.run(extract_cmd, check=True)
        print(f"Frames successfully extracted to the '{frame_dir}' directory.")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")

# Run the function using the assignment's URL
video_url = "https://www.youtube.com/watch?v=YcvECxtXoxQ"
# Start with sampling 1 frame every 5 seconds as a good baseline
download_and_extract_frames(video_url, fps="1/5")