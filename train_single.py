# train_single.py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import time
import random
from sklearn.model_selection import train_test_split

class SingleClassQuickDrawDataset(Dataset):
    def __init__(self, file_path, max_samples=1000, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_name = Path(file_path).stem
        
        print("\n" + "="*50)
        print(f"LOADING DATA FOR: {self.class_name.upper()}")
        print("="*50)
        print(f"Reading from: {file_path}")
        print(f"Maximum samples: {max_samples}")
        
        start_time = time.time()
        
        # Load data from the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Take random subset if max_samples specified
            if max_samples and max_samples < len(lines):
                lines = random.sample(lines, max_samples)
            
            print(f"\nProcessing {len(lines)} drawings...")
            for idx, line in enumerate(lines):
                if idx % 100 == 0:  # Progress update every 100 drawings
                    print(f"Progress: {idx}/{len(lines)} drawings processed", end='\r')
                    
                drawing_data = json.loads(line)
                image = self._drawing_to_image(drawing_data['drawing'])
                self.data.append(image)
                self.labels.append(1)  # Positive class
                
                # Create a negative sample (blank or random noise)
                if random.random() < 0.5:
                    # Blank image
                    self.data.append(np.zeros((28, 28), dtype=np.float32))
                else:
                    # Random noise
                    self.data.append(np.random.rand(28, 28).astype(np.float32) * 0.1)
                self.labels.append(0)  # Negative class
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels)
        
        load_time = time.time() - start_time
        print(f"\nDataset loaded in {load_time:.2f} seconds")
        print(f"Total samples: {len(self.data)} ({len(self.data)//2} real, {len(self.data)//2} negative)")
        print("="*50 + "\n")

    def _drawing_to_image(self, strokes, image_size=28):
        image = np.zeros((image_size, image_size), dtype=np.float32)
        for stroke in strokes:
            x, y = stroke[0], stroke[1]
            for i in range(len(x)):
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

class SimpleQuickDrawCNN(nn.Module):
    def __init__(self):
        super(SimpleQuickDrawCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # Binary classification: is/isn't the class
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, class_name):
    print("\n" + "="*50)
    print(f"TRAINING MODEL FOR: {class_name.upper()}")
    print("="*50)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(val_loader.dataset)}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Batch size: {train_loader.batch_size}")
    print("="*50 + "\n")
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss: {loss.item():.4f}, "
                      f"Accuracy: {100.*train_correct/train_total:.2f}%")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {epoch_val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), f'best_model_{class_name}.pth')
            print(f"New best model saved! (Validation Accuracy: {val_accuracy:.2f}%)")

def main():
    # Configuration
    config = {
        'learning_rate': 0.001,  # Adjustable learning rate
        'batch_size': 32,        # Batch size
        'num_epochs': 10,        # Number of epochs
        'max_samples': 1000,     # Maximum number of samples to use
        'class_file': 'raw/house.ndjson'  # Change this to the file you want to train on
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = SingleClassQuickDrawDataset(
        file_path=config['class_file'],
        max_samples=config['max_samples'],
        transform=transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = SimpleQuickDrawCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config['num_epochs'],
        device,
        dataset.class_name
    )

if __name__ == '__main__':
    main()
