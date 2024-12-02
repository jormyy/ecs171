import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from time import time

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

class QuickDrawDataset(Dataset):
    def __init__(self, root_dir, transform=None, samples_per_class=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.data = []
        self.labels = []
        
        print("Loading dataset...")
        total_samples = 0
        class_counts = {}
        
        for class_name in tqdm(self.classes, desc="Loading classes"):
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get list of all files for this class
            class_files = os.listdir(class_dir)
            selected_files = class_files[:samples_per_class]
            class_counts[class_name] = len(selected_files)
            
            for img_name in selected_files:
                img_path = os.path.join(class_dir, img_name)
                self.data.append(img_path)
                self.labels.append(class_idx)
                total_samples += 1
        
        print(f"\nDataset loaded with {total_samples} total samples")
        print("\nSamples per class:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")
        print()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0
    training_start = time()
    total_batches = len(train_loader)
    
    print("\nStarting training...")
    print(f"Total batches per epoch: {total_batches}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total samples per epoch: {total_batches * train_loader.batch_size}\n")
    
    for epoch in range(num_epochs):
        epoch_start = time()
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', 
                         unit='batch', leave=True)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with detailed metrics
            train_pbar.set_postfix({
                'batch': f'{batch_idx+1}/{total_batches}',
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', 
                       unit='batch', leave=True)
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        epoch_time = time() - epoch_start
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s')
        print(f'Training:   Loss: {train_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}%')
        print(f'Validation: Loss: {val_loss/len(val_loader):.4f} | Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'models/best_model_MLP.pth')
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
        print() # Empty line for readability
    
    total_time = time() - training_start
    print(f'\nTraining completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

def main():
    # Hyperparameters
    input_size = 28 * 28  # Flattened image size
    hidden_sizes = [512, 256, 128]  # Multiple hidden layers
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    samples_per_class = None  # Limit samples for testing
    
    print("\nHyperparameters:")
    print(f"Input size: {input_size}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Samples per class: {samples_per_class}")
    print()
    
    # Load and split data
    print("Initializing dataset...")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    data_dir = 'processed_data/images/inliers'
    print(f"Loading data from: {data_dir}")
    dataset = QuickDrawDataset(data_dir, transform=transform, samples_per_class=samples_per_class)
    
    train_size = 0.8
    print(f"\nSplitting dataset into {train_size*100}% train, {(1-train_size)*100}% validation...")
    train_dataset, val_dataset = train_test_split(
        dataset, train_size=train_size, random_state=42)
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    print("\nInitializing model...")
    model = MLP(input_size, hidden_sizes, num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()
