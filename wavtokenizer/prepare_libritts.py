import os
import subprocess

def download_and_extract(url, target_dir):
    """Downloads and extracts a tar.gz file from a URL into the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    
    # The tar file should be downloaded into the top-level directory (e.g., infer or train)
    tar_filename = os.path.join(target_dir, os.path.basename(url))
    
    # Check if the dataset is already extracted
    if os.path.exists(tar_filename.replace(".tar.gz", "")) and os.listdir(tar_filename.replace(".tar.gz", "")):
        print(f"{tar_filename.replace('.tar.gz', '')} already exists and is not empty, skipping download and extraction.")
        return
    
    # Download the tar file if it doesn't exist
    if not os.path.exists(tar_filename):
        print(f"Downloading {tar_filename}...")
        subprocess.run(['wget', url, '-O', tar_filename], check=True)
    else:
        print(f"{tar_filename} already exists, skipping download.")
    
    # Extract the tar file into the parent directory (not creating nested dirs)
    print(f"Extracting {tar_filename}...")
    subprocess.run(['tar', '-xzvf', tar_filename, '-C', target_dir], check=True)

def generate_filelist(extraction_subdir, output_file):
    """Generates a file list from a directory of audio files."""
    if os.path.exists(output_file):
        print(f"File list {output_file} already exists, skipping generation.")
        return

    with open(output_file, 'w') as f:
        for root, _, files in os.walk(extraction_subdir):
            for file in files:
                if file.endswith('.wav'):
                    f.write(os.path.join(root, file) + '\n')
    print(f"File list generated at {output_file}")

def main():
    # Define download URLs
    train_url = "http://www.openslr.org/resources/60/train-clean-100.tar.gz"
    val_url = "http://www.openslr.org/resources/60/dev-clean.tar.gz"

    # Define target directories and extraction subdirectories
    train_dir = "./data/train"
    val_dir = "./data/infer"
    train_extraction_subdir = os.path.join(train_dir, "LibriTTS", "train-clean-100")
    val_extraction_subdir = os.path.join(val_dir, "LibriTTS", "dev-clean")

    # Skip processing if the train or infer directory already exists
    if os.path.exists(train_extraction_subdir) and os.listdir(train_extraction_subdir):
        print(f"{train_extraction_subdir} already exists, skipping training data preparation.")
    else:
        print(f"Checking training data in {train_extraction_subdir}...")
        download_and_extract(train_url, train_dir)
        generate_filelist(train_extraction_subdir, os.path.join(train_dir, "libritts_train"))

    if os.path.exists(val_extraction_subdir) and os.listdir(val_extraction_subdir):
        print(f"{val_extraction_subdir} already exists, skipping validation data preparation.")
    else:
        print(f"Checking validation data in {val_extraction_subdir}...")
        download_and_extract(val_url, val_dir)
        generate_filelist(val_extraction_subdir, os.path.join(val_dir, "libritts_val"))

    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()


"""
wavtokenizer/
├── data/
│   ├── train/
│   │   ├── LibriTTS/
│   │   │   └── train-clean-100/
│   │   │       ├── <speaker_id>/
│   │   │       │   ├── <chapter_id>/
│   │   │       │   │   ├── <audio_files>.wav
│   │   │       │   │   └── ...
│   │   │       └── ...
│   │   └── libritts_train  # File list generated here
│   └── infer/
│       ├── LibriTTS/
│       │   └── dev-clean/
│       │       ├── <speaker_id>/
│       │       │   ├── <chapter_id>/
│       │       │   │   ├── <audio_files>.wav
│       │       │   │   └── ...
│       │       └── ...
│       └── libritts_val  # File list generated here
"""