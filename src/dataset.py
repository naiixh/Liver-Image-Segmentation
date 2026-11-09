"""
=========================================================
DATA LOADING IN A PYTORCH DATASET
Includes: Custom Dataset class for loading liver images 
and their corresponding segmentation masks.
=========================================================

"""
import os
from torch.utils.data import Dataset 
from PIL import Image
from torchvision.transforms import functional as TF

class LiverDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):        
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
                
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Found {len(self.images)} images but {len(self.labels)} labels."
            )

    def __len__(self):
        return (len(self.images))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])
        image = Image.open(img_path).convert('L') 
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        else:
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)

        label = (label > 0.5).long().squeeze(0)  
        return image, label
