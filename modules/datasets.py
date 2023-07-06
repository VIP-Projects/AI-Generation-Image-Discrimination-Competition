from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from glob import glob
from sklearn.model_selection import train_test_split
import cv2

def SplitDataset(img_dir:str, val_size:float=0.1, seed=42):
    fake_images = glob(f'{img_dir}/fake_images/*.png')
    real_images = glob(f'{img_dir}/real_images/*.png')
    labels = [1] * len(fake_images) + [0] * len(real_images)

    X_train, X_val, y_train, y_val = train_test_split(fake_images + real_images, labels, test_size=val_size, random_state=seed, shuffle=True)

    return X_train, X_val, y_train, y_val

class CustomDataset(Dataset):
    def __init__(self, X, y):
        '''
        X: list of image path
        y: list of label (0->real, 1->fake)
        '''
        
        self.X = X
        self.y = y
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)
        target = self.y[index]

        return img, target, fname
    
class TestDataset(Dataset):
    def __init__(self, X):
        '''
        X: list of image path
        '''
        self.X = X
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        img = self.transforms(img)

        return img, fname

if __name__ == '__main__':
    pass

        