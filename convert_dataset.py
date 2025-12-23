import os
import glob
from tqdm import tqdm
from convert_episode import *

def convert_dataset(input_dir, output_dir="converted_dataset", H=800, W=800):
    """
    Convert all TFRecord files in a dataset folder into rasterized images.
    
    Args:
        input_dir (str): Path to folder containing .tfrecord files.
        output_dir (str): Base directory where results will be saved.
        H, W (int): Output raster size.
    """
    os.makedirs(output_dir, exist_ok=True)

    tfrecord_files = sorted(glob.glob(os.path.join(input_dir, "*.tfrecord*")))
    if not tfrecord_files:
        print(f"No TFRecord files found in {input_dir}")
        return

    print(f"Found {len(tfrecord_files)} files in {input_dir}")

    for tfrecord_path in tqdm(tfrecord_files, desc="Converting TFRecords", unit="file"):
        try:
            tfrecord_to_images(tfrecord_path, output_base_dir=output_dir, H=H, W=W)
        except Exception as e:
            print(f"\n Error processing {tfrecord_path}: {e}")

def main():
    input_dir = "scenario/training"  
    output_dir = "converted_dataset/train"
    convert_dataset(input_dir, output_dir, H=800, W=800)

    input_dir = "scenario/validation"  
    output_dir = "converted_dataset/validation"
    convert_dataset(input_dir, output_dir, H=800, W=800)

    input_dir = "scenario/testing"  
    output_dir = "converted_dataset/test"
    convert_dataset(input_dir, output_dir, H=800, W=800)

if __name__ == "__main__":
    main()