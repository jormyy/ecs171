import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

class QuickDrawDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load and process data from each file
        for label, file_path in enumerate(file_paths):
            with open(file_path, 'r') as f:
                for line in f:
                    drawing_data = json.loads(line)
                    # Convert strokes to image
                    image = self._drawing_to_image(drawing_data['drawing'])
                    self.data.append(image)
                    self.labels.append(label)
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels)

    def _drawing_to_image(self, strokes, image_size=28):
        """Convert drawing strokes to a numpy array."""
        image = np.zeros((image_size, image_size), dtype=np.float32)
        for stroke in strokes:
            x, y = stroke[0], stroke[1]
            for i in range(len(x)):
                # Scale coordinates to image size
                px = int((x[i] / 256) * image_size)
                py = int((y[i] / 256) * image_size)
                if 0 <= px < image_size and 0 <= py < image_size:
                    image[py, px] = 255
        return image / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes):
        super(QuickDrawCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        accuracy = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    data_dir = Path('raw')
    file_paths = list(data_dir.glob('*.ndjson'))
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create dataset
    dataset = QuickDrawDataset(file_paths, transform=transform)
    
    # Split dataset
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx)
    )
    
    # Initialize model
    model = QuickDrawCNN(num_classes=len(file_paths)).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=20,
        device=device
    )
    
    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_plot.png')
    plt.close()

if __name__ == '__main__':
    main()
