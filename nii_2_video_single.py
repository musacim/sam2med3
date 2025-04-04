import os
from os.path import expanduser
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt

import nibabel as nib
import SimpleITK as sitk

def save_label_slice_as_npy(label_slice, output_file_path):
    np.save(output_file_path, label_slice)

def save_label_slice(label_slice, output_file_path):
    plt.imshow(label_slice, cmap='gray')
    plt.colorbar()
    plt.title('Label Slice')
    plt.savefig(output_file_path)
    plt.close()

def process_slice(img_slice, label_slice, subfolder_path, file_counter):
    # Binarize the label slice (convert any nonzero value to 1)
    label_slice_binary = np.where(label_slice != 0, 1, 0)
    label_output_file_path = os.path.join(subfolder_path, f"{file_counter:05d}.npy")
    save_label_slice_as_npy(label_slice_binary, label_output_file_path)
    
    # Normalize the image slice and convert to 8-bit for JPG saving.
    img_slice_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
    img_slice_8bit = (img_slice_normalized * 255).astype(np.uint8)
    output_file_path = os.path.join(subfolder_path, f"{file_counter:05d}.jpg")
    Image.fromarray(img_slice_8bit).save(output_file_path)
    print(f"Saved slice {file_counter:05d} to {output_file_path}")

def load_image(path):
    """
    Load a medical image from a .nii.gz or .mha file and return a numpy array with shape (H, W, D).
    """
    if path.endswith('.nii') or path.endswith('.nii.gz'):
        img = nib.load(path)
        img_data = img.get_fdata()
        # nibabel typically returns (D, H, W); transpose to (H, W, D)
        img_data = np.transpose(img_data, (1, 2, 0))
        return img_data
    elif path.endswith('.mha'):
        img = sitk.ReadImage(path)
        img_data = sitk.GetArrayFromImage(img)  # shape: (D, H, W)
        # Transpose to (H, W, D)
        img_data = np.transpose(img_data, (1, 2, 0))
        return img_data
    else:
        raise ValueError(f"Unsupported file format: {path}")

def process_single_image_label(image_file_path, label_file_path, output_folder_path, threshold):
    """
    Load a 3D image and its corresponding label, reorient the label so that both have shape (H, W, D),
    and then slice along the D dimension saving each slice as a JPG (for the image) and a NPY (for the label).
    """
    subfolder_name = os.path.splitext(os.path.basename(image_file_path))[0]
    subfolder_path = os.path.join(output_folder_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Load image and label volumes.
    img_data = load_image(image_file_path)    # Expected shape: (H, W, D)
    label_data = load_image(label_file_path)    # May have a different orientation.

    # Fix label orientation.
    # For example, if img_data.shape is (2048, 1024, 128) and label_data.shape is (2048, 128, 1024),
    # then we transpose the last two dimensions of the label so it becomes (2048, 1024, 128).
    label_data = np.transpose(label_data, (0, 2, 1))

    print(f"Image shape: {img_data.shape}, Label shape: {label_data.shape}")

    # Analyze positive pixels per slice (summing over height and width).
    foreground_counts = np.sum(label_data > 0, axis=(0, 1))
    positive_indices = np.where(foreground_counts > threshold)[0]

    if len(positive_indices) == 0:
        print("No slices with enough positive pixels found. Try lowering the threshold.")
        return

    # Process only slices that are within the image depth.
    min_slice = max(0, np.min(positive_indices))
    max_slice = min(img_data.shape[2], np.max(positive_indices) + 1)

    print(f"Processing slices from {min_slice} to {max_slice} (threshold: {threshold})")

    for i in range(min_slice, max_slice):
        process_slice(img_data[:, :, i], label_data[:, :, i], subfolder_path, i - min_slice)

    print(f"✅ Processing complete. Output saved in: {subfolder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert one 3D image+label to 2D slices (.jpg + .npy).')
    parser.add_argument('--image_file', type=str, required=True, help='Path to the image file (.nii.gz or .mha)')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file (.nii.gz or .mha)')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--threshold', type=int, default=0, help='Threshold for number of positive pixels in a slice')

    args = parser.parse_args()

    process_single_image_label(
        expanduser(args.image_file),
        expanduser(args.label_file),
        expanduser(args.output_folder),
        args.threshold
    )
