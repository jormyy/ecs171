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
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta

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
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def calculate_metrics(outputs, labels, loss):
    """Calculate multiple evaluation metrics"""
    _, predicted = outputs.max(1)
    accuracy = predicted.eq(labels).float().mean().item() * 100
    mse = nn.MSELoss()(torch.softmax(outputs, dim=1), 
                       nn.functional.one_hot(labels, outputs.size(1)).float()).item()
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'mse': mse
    }

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                      num_epochs, device, combination_num, total_combinations,
                      early_stopping_patience=5, min_epochs=10, min_delta=0.001):
    best_val_acc = 0
    patience_counter = 0
    training_history = []
    start_time = time()
    total_steps = num_epochs * (len(train_loader) + len(val_loader))
    
    # Single progress bar for all epochs
    pbar = tqdm(total=total_steps, 
                desc=f'Combination {combination_num}/{total_combinations}',
                unit='batch')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        mse_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            metrics = calculate_metrics(outputs, labels, loss)
            train_loss += metrics['loss']
            train_correct += metrics['accuracy'] * labels.size(0) / 100
            train_total += labels.size(0)
            mse_total += metrics['mse']
            
            # Update progress bar
            pbar.set_postfix({
                'epoch': f'{epoch+1}/{num_epochs}',
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
            pbar.update(1)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_mse_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                metrics = calculate_metrics(outputs, labels, loss)
                val_loss += metrics['loss']
                val_correct += metrics['accuracy'] * labels.size(0) / 100
                val_total += labels.size(0)
                val_mse_total += metrics['mse']
                pbar.update(1)
        
        # Calculate epoch metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        epoch_summary = {
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader),
            'train_accuracy': train_acc,
            'train_mse': mse_total/len(train_loader),
            'val_loss': val_loss/len(val_loader),
            'val_accuracy': val_acc,
            'val_mse': val_mse_total/len(val_loader)
        }
        training_history.append(epoch_summary)
        
        # Early stopping
        if epoch >= min_epochs:
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                    print(f"Validation accuracy plateaued at {best_val_acc:.2f}%")
                    pbar.close()
                    break
        elif val_acc > best_val_acc:
            best_val_acc = val_acc
    
    pbar.close()
    total_time = time() - start_time
    return best_val_acc, training_history, total_time

def plot_grid_search_results(results_df, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Heatmaps for parameter combinations
    param_pairs = list(itertools.combinations([col for col in results_df.columns 
                                             if col not in ['validation_accuracy', 'mse']], 2))
    
    for param1, param2 in param_pairs:
        plt.figure(figsize=(10, 8))
        pivot_table = results_df.pivot_table(
            values='validation_accuracy', 
            index=param1, 
            columns=param2, 
            aggfunc='max'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis')
        plt.title(f'Validation Accuracy: {param1} vs {param2}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{param1}_{param2}_{timestamp}.png'))
        plt.close()
    
    # Box plots for each hyperparameter
    metrics = ['validation_accuracy', 'mse']
    params = [col for col in results_df.columns if col not in metrics]
    
    for param in params:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy distribution
        sns.boxplot(x=param, y='validation_accuracy', data=results_df, ax=axes[0])
        axes[0].set_title(f'Validation Accuracy Distribution by {param}')
        axes[0].set_ylabel('Validation Accuracy (%)')
        
        # MSE distribution
        sns.boxplot(x=param, y='mse', data=results_df, ax=axes[1])
        axes[1].set_title(f'MSE Distribution by {param}')
        axes[1].set_ylabel('Mean Squared Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{param}_{timestamp}.png'))
        plt.close()

def main():
    # Define hyperparameter grid
    param_grid = {
        'batch_size': [32, 64, 128],
        'learning_rate': [0.001, 0.0001],
        'hidden_layers': [
            [512, 256],
            [1024, 512, 256],
            [512, 256, 128]
        ],
        'dropout_rate': [0.2, 0.3, 0.4]
    }
    
    # Create output directories
    plots_dir = 'tuning/output/plots'
    stats_dir = 'tuning/output/stats'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print("Loading dataset...")
    dataset = QuickDrawDataset('processed_data/images/inliers', 
                              transform=transform, 
                              samples_per_class=1000)
    
    results = []
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in itertools.product(*param_grid.values())]
    total_combinations = len(param_combinations)
    overall_start = time()
    
    print(f"\nStarting grid search with {total_combinations} combinations")
    print("Hyperparameter space:")
    for param, values in param_grid.items():
        print(f"{param}: {values}")
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\nStarting combination {i}/{total_combinations}")
        print("Parameters:", params)
        combination_start = time()
        
        # Split data and create loaders
        train_dataset, val_dataset = train_test_split(dataset, train_size=0.8, 
                                                     random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Initialize model
        model = MLP(
            input_size=28*28,
            hidden_layers=params['hidden_layers'],
            num_classes=10,
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train and evaluate
        best_val_acc, history, training_time = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=20, device=device,
            combination_num=i, total_combinations=total_combinations
        )
        
        # Store results
        combination_time = time() - combination_start
        results.append({
            **params,
            'validation_accuracy': best_val_acc,
            'mse': history[-1]['val_mse'],
            'training_time': training_time,
            'total_combination_time': combination_time
        })
        
        # Save training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(
            stats_dir, 
            f'training_history_combination_{i}_{timestamp}.csv'
        ), index=False)
        
        # Print combination summary
        avg_time = (time() - overall_start) / i
        remaining_time = avg_time * (total_combinations - i)
        print(f"\nCompleted in {combination_time:.1f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Estimated remaining time: {remaining_time/3600:.1f}h")
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(
        stats_dir, 
        f'grid_search_results_{timestamp}.csv'
    ), index=False)
    
    # Plot results
    print("\nGenerating plots...")
    plot_grid_search_results(results_df, plots_dir)
    
    # Find and print best configuration
    best_idx = results_df['validation_accuracy'].idxmax()
    best_config = results_df.iloc[best_idx]
    
    print("\nBest configuration found:")
    for param in param_grid.keys():
        print(f"{param}: {best_config[param]}")
    print(f"Validation accuracy: {best_config['validation_accuracy']:.2f}%")
    print(f"MSE: {best_config['mse']:.6f}")
    
    # Train final model with best parameters for 50 epochs
    print("\nTraining final model with best parameters for 50 epochs...")
    
    # Split data
    train_dataset, val_dataset = train_test_split(dataset, train_size=0.8, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_config['batch_size'])
    
# Initialize model with best parameters
    final_model = MLP(
        input_size=28*28,
        hidden_layers=best_config['hidden_layers'],
        num_classes=10,
        dropout_rate=best_config['dropout_rate']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best_config['learning_rate'])
    
    # Train final model
    _, history, _ = train_and_evaluate(
        final_model, train_loader, val_loader, criterion, optimizer,
        num_epochs=50, device=device,
        combination_num=-1, total_combinations=-1,
        early_stopping_patience=float('inf')  # Disable early stopping for final training
    )
    
    # Plot convergence for final model
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy convergence
    plt.subplot(2, 1, 1)
    plt.plot(history_df['epoch'], history_df['train_accuracy'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy over 50 Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot loss convergence
    plt.subplot(2, 1, 2)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss over 50 Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'final_model_convergence_{timestamp}.png'))
    plt.close()
    
    # Save final model history
    history_df.to_csv(os.path.join(
        stats_dir, 
        f'final_model_history_{timestamp}.csv'
    ), index=False)

if __name__ == '__main__':
    main()
