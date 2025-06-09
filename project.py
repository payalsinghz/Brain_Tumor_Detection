import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data, transform
from skimage.util import montage
from skimage.transform import rotate
from PIL import Image, ImageOps
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
import gif_your_nifti.core as gif2nif
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers import RandomFlip, RandomRotation, Rescaling


# In[2]:


import os
import nibabel as nib

def check_image_size(patient_id, dataset_path):
    # Load the image using nibabel
    image_flair = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_flair.nii')).get_fdata()
    image_t1 = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t1.nii')).get_fdata()
    image_t1ce = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t1ce.nii')).get_fdata()
    image_t2 = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t2.nii')).get_fdata()
    mask = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_seg.nii')).get_fdata()

    # Get sizes of each image
    size_flair = image_flair.shape
    size_t1 = image_t1.shape
    size_t1ce = image_t1ce.shape
    size_t2 = image_t2.shape
    size_mask = mask.shape

    # Print sizes
    print(f"Image flair size: {size_flair}")
    print(f"Image t1 size: {size_t1}")
    print(f"Image t1ce size: {size_t1ce}")
    print(f"Image t2 size: {size_t2}")
    print(f"Mask size: {size_mask}")

# Example usage
TRAIN_DATASET_PATH = r'Dataset\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
patient_id = 5  # Example patient ID

check_image_size(patient_id, TRAIN_DATASET_PATH)


# In[3]:


# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
# VOLUME_SLICES = 155 
# VOLUME_START_AT = 0 # first slice of volume that we will include
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include


# In[4]:


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def load_data(patient_id, dataset_path):
    image_flair = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_flair.nii')).get_fdata()
    image_t1 = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t1.nii')).get_fdata()
    image_t1ce = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t1ce.nii')).get_fdata()
    image_t2 = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t2.nii')).get_fdata()
    mask = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_seg.nii')).get_fdata()
    
    return image_flair, image_t1, image_t1ce, image_t2, mask

def plot_images(images, titles, slice_w):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image[:, :, image.shape[0] // 2 - slice_w], cmap='gray')
        axes[i].set_title(title)

# Example usage
TRAIN_DATASET_PATH = r'Dataset\BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
patient_id = 5  # Example patient ID
slice_w = 25

images = load_data(patient_id, TRAIN_DATASET_PATH)
image_titles = ['Image flair', 'Image t1', 'Image t1ce', 'Image t2', 'Mask']

plot_images(images, image_titles, slice_w)
plt.show()


# In[5]:


def plot_montage(image, slice_skip):
    montaged_image = montage(image[slice_skip:-slice_skip, :, :])
    rotated_image = rotate(montaged_image, 90, resize=True)
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(rotated_image, cmap='gray')
    ax.set_title('Montaged and Rotated Image')

slice_skip = 50
plot_montage(images[0], slice_skip=slice_skip)  # Plotting montaged and rotated t1 image
plt.show()


# In[6]:


import os
import numpy as np
import nibabel as nib
import imageio

def load_data(patient_id, dataset_path):
    image_flair = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_flair.nii')).get_fdata()
    image_t1 = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t1.nii')).get_fdata()
    image_t1ce = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t1ce.nii')).get_fdata()
    image_t2 = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_t2.nii')).get_fdata()
    mask = nib.load(os.path.join(dataset_path, f'BraTS20_Training_{patient_id:03d}', f'BraTS20_Training_{patient_id:03d}_seg.nii')).get_fdata()
    
    return image_flair, image_t1, image_t1ce, image_t2, mask

def create_and_save_gif(patient_id, dataset_path):
    # Load data for the specified patient using the load_data function
    image_flair, _, _, _, mask = load_data(patient_id, dataset_path)

    # Prepare to create and save the GIF
    label = f'BraTS20_Training_{patient_id:03d}'  # This will be the label for this patient
    filename = f'{label}_3d_2d.gif'

    # Initialize a list to collect frames for the GIF
    frames = []

    # Loop through the slices of the FLAIR image (assuming all have the same shape)
    for i in range(image_flair.shape[2]):
        # Extract the 2D slice from FLAIR and the corresponding mask
        image = np.rot90(image_flair[:, :, i])
        mask_slice = np.clip(np.rot90(mask[:, :, i]), 0, 255).astype(np.uint8) * 255  # Adjust mask intensity if needed

        # Combine image and mask for visualization (e.g., overlay mask on image)
        # Here we use a simple overlay approach, adjust as per your visualization needs
        combined_image = np.stack([image, image, image], axis=-1)  # Convert to RGB image
        combined_image[..., 0] += mask_slice  # Add mask to the red channel

        # Append the combined image as a frame for the GIF
        frames.append(combined_image)

    # Save the collected frames as a GIF using imageio
    imageio.mimsave(filename, frames, fps=15)

    print(f'GIF saved successfully as: {filename}')

# Example usage:
patient_id = 1  # Specify the patient ID you want to process
dataset_path = 'archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'  # Specify the path to your dataset
create_and_save_gif(patient_id, dataset_path)


# In[7]:


# import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
# from IPython.display import display, HTML
# import base64

# def create_and_save_gif(sample_img_filename, sample_mask_filename, output_filename):
#     sample_img = nib.load(sample_img_filename)
#     sample_img_data = np.asanyarray(sample_img.dataobj)
    
#     sample_mask = nib.load(sample_mask_filename)
#     sample_mask_data = np.asanyarray(sample_mask.dataobj)
    
#     # Determine number of slices
#     num_slices = sample_img_data.shape[2]
    
#     # Initialize plot
#     fig, ax = plt.subplots()
#     ax.axis('off')
    
#     # Function to update each frame
#     def update(frame):
#         ax.clear()
#         image = sample_img_data[:, :, frame]
#         mask = sample_mask_data[:, :, frame]
        
#         ax.imshow(image, cmap='gray')
#         ax.imshow(mask, cmap='viridis', alpha=0.5)
#         ax.set_title(f"Slice {frame}")
        
#     # Create animation
#     animation = FuncAnimation(fig, update, frames=num_slices, blit=False)
#     animation.save(output_filename, writer='pillow', fps=10)  # Save animation as GIF
    
#     plt.close(fig)  # Close plot after saving

# # Example usage
# sample_img_filename = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
# sample_mask_filename = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'
# output_filename = 'sample_output.gif'

# create_and_save_gif(sample_img_filename, sample_mask_filename, output_filename)


# # Function to display GIF
# def show_gif(filename):
#     with open(filename, 'rb') as file:
#         # Read binary data
#         gif_binary = file.read()
#         # Encode binary data as base64
#         gif_base64 = base64.b64encode(gif_binary).decode()
#         # Create HTML string with embedded GIF
#         html_str = f'<img src="data:image/gif;base64,{gif_base64}">'
#         # Display HTML
#         display(HTML(html_str))

# # Display the saved GIF
# show_gif(output_filename)


# In[8]:


# shutil.copy2(TRAIN_DATASET_PATH + '\BraTS20_Training_001/BraTS20_Training_001_flair.nii', './test_gif_BraTS20_Training_001_flair.nii')
# gif2nif.write_gif_normal('./test_gif_BraTS20_Training_001_flair.nii')


# In[9]:


# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
#    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss


 
# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# In[10]:


import os
from sklearn.model_selection import train_test_split

# List directories containing studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

# Remove directory with ill-formatted segmentation file name
train_and_val_directories = [dir_path for dir_path in train_and_val_directories if not dir_path.endswith('BraTS20_Training_355')]

# Extract study IDs from directory paths
def pathListIntoIds(dirList):
    return [dir_path.split(os.path.sep)[-1] for dir_path in dirList]

train_and_test_ids = pathListIntoIds(train_and_val_directories)

# Split IDs into train, and test sets

train_ids, test_ids = train_test_split(train_and_test_ids, test_size=0.15, random_state=42)

# Display the lengths of each set
print(f"Number of training IDs: {len(train_ids)}")
print(f"Number of test IDs: {len(test_ids)}")


# In[11]:


# import os
# import numpy as np
# import nibabel as nib
# import cv2
# import tensorflow as tf
# from tensorflow import keras

# IMG_SIZE = 240

# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=4, shuffle=False):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
#         Batch_ids = [self.list_IDs[k] for k in indexes]
#         X, Y = self.__data_generation(Batch_ids)
#         return X, Y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, Batch_ids):
#         'Generates data containing batch_size samples'
#         X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
#         Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

#         for c, i in enumerate(Batch_ids):
#             case_path = os.path.join(TRAIN_DATASET_PATH, i)

#         # Load FLAIR, T1CE, T1, T2, and segmentation
#             flair_path = os.path.join(case_path, f'{i}_flair.nii')
#             ce_path = os.path.join(case_path, f'{i}_t1ce.nii')
#             t1_path = os.path.join(case_path, f'{i}_t1.nii')
#             t2_path = os.path.join(case_path, f'{i}_t2.nii')
#             seg_path = os.path.join(case_path, f'{i}_seg.nii')

#             flair = nib.load(flair_path).get_fdata()
#             ce = nib.load(ce_path).get_fdata()
#             t1 = nib.load(t1_path).get_fdata()
#             t2 = nib.load(t2_path).get_fdata()
#             seg = nib.load(seg_path).get_fdata()

#             for j in range(VOLUME_SLICES):
#                 X[j + VOLUME_SLICES * c, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
#                 X[j + VOLUME_SLICES * c, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
#                 X[j + VOLUME_SLICES * c, :, :, 2] = cv2.resize(t1[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
#                 X[j + VOLUME_SLICES * c, :, :, 3] = cv2.resize(t2[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

#                 # Resize segmentation and convert to one-hot encoding
#                 resized_seg = cv2.resize(seg[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
#                 # Ensure segmentation values are within valid range (0 to 3)
#                 resized_seg[resized_seg == 4] = 3
#                 # Convert to one-hot encoding
#                 one_hot_seg = tf.one_hot(resized_seg.astype(np.uint8), 4)
#                 # Store in Y with appropriate dimensions
#                 Y[j + VOLUME_SLICES * c, :, :, :] = one_hot_seg

#         return X / np.max(X), Y


# # Usage example:
# train_generator = DataGenerator(train_ids)
# test_generator = DataGenerator(test_ids)


# In[12]:


import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 240

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=4, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(Batch_ids)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            flair_path = os.path.join(case_path, f'{i}_flair.nii')
            ce_path = os.path.join(case_path, f'{i}_t1ce.nii')
            t1_path = os.path.join(case_path, f'{i}_t1.nii')
            t2_path = os.path.join(case_path, f'{i}_t2.nii')
            seg_path = os.path.join(case_path, f'{i}_seg.nii')

            flair = nib.load(flair_path).get_fdata().astype(np.float32)  # Load and cast to float32
            ce = nib.load(ce_path).get_fdata().astype(np.float32)
            t1 = nib.load(t1_path).get_fdata().astype(np.float32)
            t2 = nib.load(t2_path).get_fdata().astype(np.float32)
            seg = nib.load(seg_path).get_fdata()

            for j in range(VOLUME_SLICES):
                # Gaussian smoothing and adaptive Gaussian thresholding
                flair_smooth = cv2.GaussianBlur(flair[:, :, j + VOLUME_START_AT], (3, 3), 0).astype(np.uint8)
                ce_smooth = cv2.GaussianBlur(ce[:, :, j + VOLUME_START_AT], (3, 3), 0).astype(np.uint8)
                t1_smooth = cv2.GaussianBlur(t1[:, :, j + VOLUME_START_AT], (3, 3), 0).astype(np.uint8)
                t2_smooth = cv2.GaussianBlur(t2[:, :, j + VOLUME_START_AT], (3, 3), 0).astype(np.uint8)

                # Adaptive Gaussian thresholding
                _, flair_thresh = cv2.threshold(flair_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, ce_thresh = cv2.threshold(ce_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, t1_thresh = cv2.threshold(t1_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, t2_thresh = cv2.threshold(t2_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

               # Resize to desired dimensions
                flair_resized = cv2.resize(flair_thresh, (IMG_SIZE, IMG_SIZE))
                ce_resized = cv2.resize(ce_thresh, (IMG_SIZE, IMG_SIZE))
                t1_resized = cv2.resize(t1_thresh, (IMG_SIZE, IMG_SIZE))
                t2_resized = cv2.resize(t2_thresh, (IMG_SIZE, IMG_SIZE))

                # Store in X
                X[j + VOLUME_SLICES * c, :, :, 0] = flair_resized
                X[j + VOLUME_SLICES * c, :, :, 1] = ce_resized
                X[j + VOLUME_SLICES * c, :, :, 2] = t1_resized
                X[j + VOLUME_SLICES * c, :, :, 3] = t2_resized

                # Resize segmentation and convert to one-hot encoding
                resized_seg = cv2.resize(seg[:, :, j + VOLUME_START_AT].astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                resized_seg[resized_seg == 4] = 3
                one_hot_seg = tf.one_hot(resized_seg, 4)
                Y[j + VOLUME_SLICES * c, :, :, :] = one_hot_seg

        return X / 255.0, Y


# Usage example:
train_generator = DataGenerator(train_ids)
test_generator = DataGenerator(test_ids)


# In[13]:


# Example: Check the shape of the first batch from test_generator
batch_X, batch_Y = test_generator[0]  # Assuming batch_size=1 for simplicity
print(f"Shape of X (input images): {batch_X.shape}, Shape of Y (one-hot encoded segmentation): {batch_Y.shape}")


# In[14]:


import matplotlib.pyplot as plt

# Define titles for image channels
image_titles = ['FLAIR', 'T1CE', 'T1', 'T2', 'Segmentation Mask']

# Set figure size and axis properties
plt.figure(figsize=(25, 5))
plt.axis('off')

# Display preprocessed images and segmentation mask for a single slice
slice_idx = 50 # Choose the slice index to visualize (e.g., first slice in the batch)

# Plot preprocessed FLAIR, T1CE, T1, T2 images
for channel_idx in range(batch_X.shape[-1]):
    plt.subplot(1, batch_X.shape[-1] + 1, channel_idx + 1)
    plt.imshow(batch_X[slice_idx, :, :, channel_idx], cmap='gray')
    plt.title(image_titles[channel_idx])
    plt.axis('off')

# Plot the segmentation mask
seg_mask = tf.argmax(batch_Y[slice_idx, :, :, :], axis=-1)
plt.subplot(1, batch_X.shape[-1] + 1, batch_X.shape[-1] + 1)
plt.imshow(seg_mask, cmap='jet', vmin=0, vmax=3)  # Assuming 4 classes (0 to 3)
plt.title(image_titles[-1])
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()


# In[15]:


def showDataLayout(train_ids, test_ids):
    # Calculate the number of images in each dataset split
    train_count = len(train_ids)
    
    test_count = len(test_ids)

    # Create a bar plot to show data distribution
    categories = ["Train",  "Test"]
    counts = [train_count, test_count]
    colors = ['red', 'blue']
    
    plt.bar(categories, counts, align='center', color=colors)
    plt.ylabel('Number of images')
    plt.title('Data distribution')
    
    plt.show()

# Example usage
showDataLayout(train_ids, test_ids)


# In[16]:


from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau

# Define the CSV logger to save training logs to a file
csv_logger = CSVLogger('training.log', separator=',', append=False)

# Define the callbacks list
callbacks = [
    # Reduce learning rate on plateau based on training loss
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
    # CSV logger to save training logs
    #csv_logger
]


# In[25]:


import tensorflow as tf
from tensorflow.keras import layers

# Transformer encoder layer
batch_size = train_generator.batch_size
width, height = IMG_SIZE, IMG_SIZE  # Assuming the input image size is IMG_SIZE x IMG_SIZE

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Transformer encoder
class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
             maximum_position_encoding, batch_size = train_generator.batch_size, width= IMG_SIZE, height= IMG_SIZE, rate=0.1):
            super(TransformerEncoder, self).__init__()

            self.d_model = d_model
            self.num_layers = num_layers

            self.embedding = layers.Embedding(input_vocab_size, d_model)
            self.pos_encoding = positional_encoding(maximum_position_encoding, d_model, batch_size, width, height)

            self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

            self.dropout = layers.Dropout(rate)


    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

# Positional encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model, batch_size, width, height):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    # Convert batch_size to integer
    batch_size = int(batch_size)

    # Repeat the positional encoding for each image in the batch
    pos_encoding = tf.repeat(pos_encoding, repeats=batch_size, axis=0)

    # Reshape the positional encoding to match the input tensor shape
    pos_encoding = tf.reshape(pos_encoding, [batch_size, width, height, d_model])

    return tf.cast(pos_encoding, dtype=tf.float32)




# Transformer model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, output_dim, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff,
                                           input_vocab_size, maximum_position_encoding, rate)

        self.final_layer = layers.Dense(output_dim)

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        final_output = self.final_layer(enc_output)

        return final_output
# Hyperparameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
input_vocab_size = 256  # Define the vocabulary size based on your data
maximum_position_encoding = 240  # Define the maximum position encoding based on your data
output_dim = 4  # Number of classes

# Create an instance of the Transformer model
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, maximum_position_encoding,
                          output_dim, dropout_rate)

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Define metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = transformer(inputs, training=True, enc_padding_mask=None)
        loss = loss_object(targets, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(targets, predictions)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch in train_generator:
        inputs, targets = batch
        train_step(inputs, targets)

    print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}')


# In[94]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import MeanIoU

def build_unet(input_shape, num_classes=4, kernel_initializer='he_normal', dropout=0.2):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom/central layer
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(dropout)(conv5)

    # Expansive path
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)

    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)  # Output with softmax activation for multi-class

    return Model(inputs=inputs, outputs=conv10)

# Define input shape
input_shape = (240, 240, 4)  # Input shape for (height, width, channels)

# Build the U-Net model
model1 = build_unet(input_shape, num_classes=4)  # Assuming 4 classes for segmentation

# Compile the model
#optimizer = Adam(learning_rate=0.001)  optimizer=optimizer, 
model1.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),  metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing])
# Display model summary
model1.summary()



# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU, Precision, Recall

def build_ca_cnn(input_shape, num_classes=4, kernel_initializer='he_normal', dropout=0.2):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom/central layer
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(dropout)(conv4)

    # Expansive path
    up5 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop4))
    merge5 = Concatenate(axis=3)([conv3, up5])
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)

    up6 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=3)([conv2, up6])
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2D(32, (2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv1, up7])
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    # Additional convolutional layers for enhanced feature extraction (anisotropic)
    conv8 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)
    conv9 = Conv2D(64, (1, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    # Output layer
    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    return Model(inputs=inputs, outputs=conv10)

# Define input shape
input_shape = (240, 240, 4)   # Adjusted input shape for (height, width, channels)

# Build the CA-CNN model
model2 = build_ca_cnn(input_shape, num_classes=4)  # Assuming 4 classes for segmentation

# Compile the model
# optimizer = Adam(learning_rate=0.001)
model2.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),  metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing])

# Display model summary
model2.summary()


# In[96]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

def build_vnet(input_shape, num_classes=4, kernel_initializer='he_normal', dropout=0.2):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom/central layer
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(dropout)(conv5)

    # Expansive path
    up6 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(drop5)
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = Conv2DTranspose(16, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)

    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)  # Output with softmax activation for multi-class

    return Model(inputs=inputs, outputs=conv10)

# Define input shape
input_shape = (240, 240, 4)   # Input shape for (height, width, channels)

# Build the V-Net model
model3 = build_vnet(input_shape, num_classes=4)  # Assuming 4 classes for segmentation

# Compile the model
#optimizer = Adam(learning_rate=0.001)
model3.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),  metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing])

# Display model summary
model3.summary()


# In[97]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Activation, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

def build_attention_unet(input_shape, num_classes=4, kernel_initializer='he_normal', dropout=0.2):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom/central layer
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(dropout)(conv5)

    # Expansive path with attention gates
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop5))
    up6 = AttentionGate(conv4, up6, 256)  # Attention gate between conv4 and up6
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    up7 = AttentionGate(conv3, up7, 128)  # Attention gate between conv3 and up7
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    up8 = AttentionGate(conv2, up8, 64)  # Attention gate between conv2 and up8
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    up9 = AttentionGate(conv1, up9, 32)  # Attention gate between conv1 and up9
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)

    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)  # Output with softmax activation for multi-class

    return Model(inputs=inputs, outputs=conv10)

def AttentionGate(g, x, out_channels):
    g_conv = Conv2D(out_channels, 1, activation='relu', padding='same')(g)
    x_conv = Conv2D(out_channels, 1, activation='relu', padding='same')(x)

    psi = tf.keras.activations.sigmoid(g_conv + x_conv)
    out = Multiply()([x, psi])
    return out

# Define input shape
input_shape = (240, 240, 4)   # Input shape for (height, width, channels)

# Build the Attention U-Net model
model4 = build_attention_unet(input_shape, num_classes=4)  # Assuming 4 classes for segmentation

# Compile the model
model4.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),  metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing])

# Display model summary
model4.summary()


# In[98]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import MeanIoU

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    possible_negatives = tf.reduce_sum(1 - y_true)
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def build_nnunet(input_shape, num_classes=4, kernel_initializer='he_normal', dropout=0.2):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom/central layer
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(dropout)(conv5)

    # Expansive path
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)

    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)  # Output with softmax activation for multi-class

    return Model(inputs=inputs, outputs=conv10)

# Define input shape
input_shape = (240, 240, 4)   # Input shape for (height, width, channels)

# Build the U-Net model
model5 = build_nnunet(input_shape, num_classes=4)  # Assuming 4 classes for segmentation

# Compile the model
#optimizer = Adam(learning_rate=0.001)
model5.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001),  metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing])
# Display model summary
model5.summary()


# In[99]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

def residual_block(x, filters, kernel_size=3, dropout=0.2):
    # Save the input tensor for the residual connection
    input_tensor = x

    # First convolutional layer
    conv1 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)

    # Second convolutional layer
    conv2 = Conv2D(filters, kernel_size, activation='relu', padding='same')(conv1)

    # Apply dropout if specified
    if dropout > 0:
        conv2 = Dropout(dropout)(conv2)

    # Check if the number of channels matches
    if input_tensor.shape[-1] != conv2.shape[-1]:
        # Adjust the number of channels using 1x1 convolution if needed
        input_tensor = Conv2D(filters, (1, 1), padding='same')(input_tensor)

    # Add the input tensor (with appropriate adjustment) to the output of the residual block
    residual = Add()([input_tensor, conv2])

    return residual

def build_unetr(input_shape, num_classes=4, kernel_initializer='he_normal', dropout=0.2):
    inputs = Input(input_shape)

    # Contracting path with residual blocks
    conv1 = residual_block(inputs, 32)
    conv1 = residual_block(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, 64)
    conv2 = residual_block(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_block(pool2, 128)
    conv3 = residual_block(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_block(pool3, 256)
    conv4 = residual_block(conv4, 256)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom/central layer
    conv5 = residual_block(pool4, 512)
    conv5 = residual_block(conv5, 512)
    drop5 = Dropout(dropout)(conv5)

    # Expansive path
    up6 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = residual_block(merge6, 256)
    conv6 = residual_block(conv6, 256)

    up7 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = residual_block(merge7, 128)
    conv7 = residual_block(conv7, 128)

    up8 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = residual_block(merge8, 64)
    conv8 = residual_block(conv8, 64)

    up9 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = residual_block(merge9, 32)
    conv9 = residual_block(conv9, 32)

    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)  # Output with softmax activation for multi-class

    return Model(inputs=inputs, outputs=conv10)

# Define input shape
input_shape = (240, 240, 4)   # Input shape for (height, width, channels)

# Build the UNetR model
model6 = build_unetr(input_shape, num_classes=4)  # Assuming 4 classes for segmentation

# Compile the model
optimizer = Adam(learning_rate=0.001)
model6.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),  metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing])

# Display model summary
model6.summary()


# In[100]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

def build_nnformer(input_shape, num_classes=4, kernel_initializer='he_normal', dropout=0.2):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom/central layer
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)

    # Expansive path
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)

    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)  # Output with softmax activation for multi-class

    return Model(inputs=inputs, outputs=conv10)

# Define input shape
input_shape = (240, 240, 4)  # Input shape for (height, width, channels)

# Build the nnformer model
model7 = build_nnformer(input_shape, num_classes=4)  # Assuming 4 classes for segmentation

# Compile the model
model7.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),  metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing])

# Display model summary
model7.summary()


# In[103]:


# Instantiate your DataGenerator for training and testing
training_generator = DataGenerator(train_ids)
test_generator = DataGenerator(test_ids)


# In[104]:


history= model1.fit(training_generator, epochs=1,steps_per_epoch=len(train_ids),callbacks= callbacks, validation_data=test_generator)

# Save the trained model
model1.save('unet_model.h5')


# In[105]:


history2 = model2.fit(training_generator, epochs=1,steps_per_epoch=len(train_ids),callbacks= callbacks, validation_data=test_generator)

# Save the trained model
model2.save('ca_cnn_model.h5')


# In[106]:


history3 = model3.fit(training_generator, epochs=1,steps_per_epoch=len(train_ids),callbacks= callbacks, validation_data=test_generator)

# Save the trained model
model3.save('vnet_model.h5')


# In[107]:


history4 = model4.fit(training_generator, epochs=1,steps_per_epoch=len(train_ids),callbacks= callbacks, validation_data=test_generator)

# Save the trained model
model4.save('attention_unet_model.h5')


# In[108]:


history5 = model5.fit(training_generator, epochs=1,steps_per_epoch=len(train_ids),callbacks= callbacks, validation_data=test_generator)

# Save the trained model
model5.save('nnunet_model.h5')


# In[109]:


history6 = model6.fit(training_generator, epochs=5,steps_per_epoch=len(train_ids),callbacks= callbacks, validation_data=test_generator)

# Save the trained model
model6.save('unetr_model.h5')


# In[110]:


import matplotlib.pyplot as plt

# Access training history
history = history6.history

# Plotting training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting training and validation Dice Coefficient
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['dice_coef'], label='Training Dice Coefficient')
plt.plot(history['val_dice_coef'], label='Validation Dice Coefficient')
plt.title('Training and Validation Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()

# Plotting training and validation Mean IOU
plt.subplot(1, 2, 2)
plt.plot(history['mean_io_u_5'], label='Training Mean IOU')
plt.plot(history['val_mean_io_u_5'], label='Validation Mean IOU')
plt.title('Training and Validation Mean IOU')
plt.xlabel('Epoch')
plt.ylabel('Mean IOU')
plt.legend()
plt.show()


# In[ ]:


history7 = model7.fit(training_generator, epochs=1,steps_per_epoch=len(train_ids),callbacks= callbacks, validation_data=test_generator)

# Save the trained model
model7.save('nnformer_model.h5')


# In[111]:


import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import tensorflow as tf

# Constants and configurations
IMG_SIZE = 240
SEGMENT_CLASSES = ["Background", "Edema", "Core", "Enhancing"]

# Function to load and preprocess MRI images
def imageLoader(path):
    image = nib.load(path).get_fdata()
    return image

# Function to load data from directory for a specified MRI type
def loadDataFromDir(path, list_of_files, mriType, n_images):
    scans = []
    masks = []
    for i in list_of_files[:n_images]:
        fullPath = glob.glob(i + f'/*{mriType}*')[0]
        currentScanVolume = imageLoader(fullPath)
        currentMaskVolume = imageLoader(glob.glob(i + '/*seg*')[0])
        for j in range(0, currentScanVolume.shape[2]):
            scan_img = cv2.resize(currentScanVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
            mask_img = cv2.resize(currentMaskVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
            scans.append(scan_img[..., np.newaxis])
            masks.append(mask_img[..., np.newaxis])
    return np.array(scans, dtype='float32'), np.array(masks, dtype='float32')

# Function to predict using a loaded model for a specific case
# Function to predict using a loaded model for a specific case
def predictByPath(case_path, case, model):
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 4))  # Four channels: flair, t1ce, t1, t2
    
    # Load and resize MRI modalities
    modalities = ['flair', 't1ce', 't1', 't2']
    for idx, modality in enumerate(modalities):
        vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_{modality}.nii')
        modality_volume = nib.load(vol_path).get_fdata()
        for j in range(VOLUME_SLICES):
            # Resize the modality volume to match the model's input size
            resized_modality = cv2.resize(modality_volume[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
            X[j,:,:,idx] = resized_modality
    
    # Make predictions using the loaded model
    predictions = model.predict(X / np.max(X), verbose=1)
    return predictions


# Function to visualize predictions for a specific case and save as PDF
def showPredictsById(case, start_slice=60, model=None, pdf_filename="output_visualizations.pdf"):
    path = f"archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_t1.nii')).get_fdata()  # Using T1 for visualization
    
    # Predict using the specified model
    if model is not None:
        p = predictByPath(path, case, model)
        core = p[:,:,:,1]
        edema = p[:,:,:,2]
        enhancing = p[:,:,:,3]

        # Create PDF file to save visualizations
        with PdfPages(pdf_filename) as pdf:
            plt.figure(figsize=(18, 50))
            f, axarr = plt.subplots(1, 6, figsize=(18, 50)) 

            for i in range(6):  # Display original image in grayscale
                axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
            
            axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
            axarr[0].title.set_text('Original image T1')
            curr_gt = cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
            axarr[1].title.set_text('Ground truth')
            axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
            axarr[2].title.set_text('All classes predicted')
            axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
            axarr[4].imshow(core[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
            axarr[5].imshow(enhancing[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')

            pdf.savefig()
            plt.close()

# Load the trained model
model_path = "unet_model.h5"
custom_objects = {'dice_coef': dice_coef, 'dice_coef_necrotic': dice_coef_necrotic, 'dice_coef_edema': dice_coef_edema, 'dice_coef_enhancing': dice_coef_enhancing, 'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity}
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Test the prediction visualization for multiple test cases and save as PDF
test_ids = ["001", "002", "003", "004", "005", "006", "007"]  # Assuming these are the test case IDs
for idx, test_case in enumerate(test_ids):
    pdf_filename = f"prediction_visualizations_{idx+1}.pdf"
    showPredictsById(test_case, model=loaded_model, pdf_filename=pdf_filename)

