import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initializes the FaceDataset class.

        Args:
            image_dir (str): Path to the image directory.
            mask_dir (str): Path to the mask directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # List of image names
        
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
            """
            Retrieves the image and mask at the given index.

            Parameters:
                idx (int): Index of the image and mask to retrieve.

            Returns:
                tuple: A tuple containing the image and mask.
            """
            image_filename = self.images[idx]
            # Split the filename and extension
            base, extension = os.path.splitext(image_filename)
            # Create the mask filename by adding '_mask.PNG' to the base
            mask_filename = base + "_mask.PNG"
            # Now create the full paths
            image_path = os.path.join(self.image_dir, image_filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)
                       
            image = np.array(Image.open(image_path).convert("RGB")) # Convert to RGB
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # Convert to grayscale
            
            # Some preprocessing
            mask[mask == 255.0] = 1.0 # since using sigmoid activation
            
            # Perform data augmentation
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            
            return image, mask
        