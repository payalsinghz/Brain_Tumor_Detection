import os
import nibabel as nib

TRAIN_DATASET_PATH = r'Dataset\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
patient_id = '005'
flair_path = os.path.join(TRAIN_DATASET_PATH, f'BraTS20_Training_{patient_id}', f'BraTS20_Training_{patient_id}_flair.nii')
print(f"Checking path: {flair_path}")
print(f"Path exists: {os.path.exists(flair_path)}")
if os.path.exists(flair_path):
    image_flair = nib.load(flair_path).get_fdata()
    print(f"Image shape: {image_flair.shape}")
else:
    print("File not found. Check dataset path and structure.")