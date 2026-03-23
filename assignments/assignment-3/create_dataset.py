import os
from datasets import Dataset, Image
from huggingface_hub import login

def upload_proper_image_dataset():

    hf_token = "PASTE_YOUR_TOKEN_HERE"
    repo_id = "KalpPatel342/Assignment3"
    image_folder = "detections"
    
    print("Logging in to Hugging Face...")
    login(token=hf_token, add_to_git_credential=False)
    
    print("Scanning detections folder...")
    # Grab all the images
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    full_paths = [os.path.join(image_folder, f) for f in image_files]
    
    if not full_paths:
        print("No images found in the detections folder!")
        return
    ids = list(range(1, len(image_files) + 1))
    labels = ["drone"] * len(image_files)
    
    data = {
        "id": ids,
        "image": full_paths, 
        "label": labels
    }
    
    print(f"Building dataset with {len(ids)} images...")
 
    hf_dataset = Dataset.from_dict(data)

    hf_dataset = hf_dataset.cast_column("image", Image())
    
    print("Uploading to Hub (this handles formatting, chunking, and uploading automatically)...")
    hf_dataset.push_to_hub(repo_id)
    
    print("\nSuccess! Deliverable 1 is officially perfect.")
    print(f"Check your viewer at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    upload_proper_image_dataset()