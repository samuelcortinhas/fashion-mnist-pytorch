import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd



# Training data
train_data=pd.read_csv('../input/digit-recognizer/train.csv')

# Test data scaled to lie in [0,1]
test_data=pd.read_csv('../input/digit-recognizer/test.csv')/255

y = train_data.label

# Features scaled to lie in [0,1]
X = train_data.drop('label', axis=1)/255

X = X.values.reshape(-1, 1, 28, 28)

data_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=(28,28), scale=(0.8, 1.2),ratio=(1,1)),
        ])

class MNIST(Dataset):
    # Initialise
    def __init__(self, subset='train', transform=None):
        super().__init__()
        self.subset = subset
        self.transform = transform
        
        # Store data
        if self.subset=='train':
            self.X = torch.from_numpy(X_train.astype(np.float32))
            self.y = torch.from_numpy(y_train.values)
        elif self.subset=='valid':
            self.X = torch.from_numpy(X_valid.astype(np.float32))
            self.y = torch.from_numpy(y_valid.values)
        elif self.subset=='test':
            self.X = torch.from_numpy(test_data.astype(np.float32))
        else:
            raise Exception("subset must be train, valid or test")
        
    # Get item in position given by index
    def __getitem__(self, index):
        if self.subset=='test':
            return self.X[index]
        else:
            sample = self.X[index]
            
        # Data augmentations
        if self.transform:
            sample = self.transform(sample)
    
        return sample, self.y[index]
        
    # Length of dataset 
    def __len__(self):
        return self.X.shape[0]

train_dataset = MNIST(subset='train', transform=data_transforms)
valid_dataset = MNIST(subset='valid')
test_dataset = MNIST(subset='test')

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)