import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, conv1_channels=64, conv2_channels=128, linear1_size=256, linear2_size=128, dropout_rate=0.25):
        super().__init__()
        
        # Layers
        self.conv1=nn.Conv2d(in_channels=1, out_channels=conv1_channels, kernel_size=5, stride=1, padding=2)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2=nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3, stride=1, padding=1)
        self.flat=nn.Flatten()
        self.fc1=nn.Linear(in_features=conv2_channels*7*7, out_features=linear1_size)
        self.fc2=nn.Linear(in_features=linear1_size, out_features=linear2_size)
        self.fc3=nn.Linear(in_features=linear2_size, out_features=10)
        self.relu=nn.ReLU()
        self.drop = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        
        # Conv block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        
        # Conv block 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        
        # Fully connected layer 1
        out = self.flat(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        
        # Fully connected layer 2
        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)
        
        # Output layer (no softmax needed)
        out = self.fc3(out)
        
        return out
