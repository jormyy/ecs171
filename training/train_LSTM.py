import torch 
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
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
    def __init__(self, data_dir, samples_per_class=None):
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        
        # Get list of classes from directory
        classes = [f.replace('.ndjson', '') for f in os.listdir(data_dir) 
                  if f.endswith('.ndjson')]
        classes = sorted(classes)
        
        print("Loading dataset...")
        total_samples = 0
        class_counts = {}
        
        for idx, class_name in enumerate(tqdm(classes, desc="Loading classes")):
            class_counts[class_name] = 0
            self.class_to_idx[class_name] = idx
            file_path = os.path.join(data_dir, f"{class_name}.ndjson")
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                # If samples_per_class is specified, limit the number of samples
                if samples_per_class is not None:
                    lines = lines[:samples_per_class]
                
                for line in lines:
                    drawing_data = json.loads(line)
                    # Extract stroke data and normalize
                    strokes = []
                    for stroke in drawing_data['drawing']:
                        points = np.array(stroke[0:2]).T  # Only x,y coordinates
                        # Normalize to [0, 1]
                        points = points.astype(np.float32)
                        points[:, 0] = points[:, 0] / 1000.0  # assuming max width is 1000
                        points[:, 1] = points[:, 1] / 1000.0  # assuming max height is 1000
                        strokes.append(points)
                    
                    # Pad or truncate to fixed sequence length
                    max_seq_length = 200
                    padded_stroke = np.zeros((max_seq_length, 2), dtype=np.float32)
                    total_points = 0
                    
                    for stroke in strokes:
                        points_to_copy = min(stroke.shape[0], max_seq_length - total_points)
                        if points_to_copy <= 0:
                            break
                        padded_stroke[total_points:total_points + points_to_copy] = \
                            stroke[:points_to_copy]
                        total_points += points_to_copy
                    
                    self.data.append(padded_stroke)
                    self.labels.append(idx)
                    class_counts[class_name] += 1
                    total_samples += 1
        
        print(f"\nDataset loaded with {total_samples} total samples")
        print("\nSamples per class:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")
        print()

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        self.data = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class QuickDrawLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(QuickDrawLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

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
        
        # Training loop with progress bar
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
        
        # Validation loop with progress bar
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
            }, 'models/best_model_LSTM_2.pth')
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
        print() # Empty line for readability
    
    total_time = time() - training_start
    print(f'\nTraining completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

def main():
    # Hyperparameters
    input_size = 2  # x,y coordinates
    hidden_size = 256
    num_layers = 2
    num_classes = 10  # number of drawing classes
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    samples_per_class = None  # Limit samples for testing
    
    print("\nHyperparameters:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Samples per class: {samples_per_class}")
    print()
    
    # Load and split data
    print("Initializing dataset...")
    data_dir = 'processed_data/inliers'
    print(f"Loading data from: {data_dir}")
    dataset = QuickDrawDataset(data_dir, samples_per_class=samples_per_class)
    
    train_size = 0.8
    print(f"\nSplitting dataset into {train_size*100:.0f}% train, {(1-train_size)*100:.0f}% validation...")
    split_start = time()
    train_dataset, val_dataset = train_test_split(
        dataset, train_size=train_size, random_state=42)
    split_time = time() - split_start
    print(f"\nSplit completed in {split_time:.2f}s")
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    print("\nInitializing model...")
    model = QuickDrawLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()
