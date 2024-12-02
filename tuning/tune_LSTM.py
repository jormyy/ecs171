import torch 
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from time import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# Reuse QuickDrawDataset class from original script
class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, samples_per_class=None):
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        
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
                if samples_per_class is not None:
                    lines = lines[:samples_per_class]
                
                for line in lines:
                    drawing_data = json.loads(line)
                    strokes = []
                    for stroke in drawing_data['drawing']:
                        points = np.array(stroke[0:2]).T
                        points = points.astype(np.float32)
                        points[:, 0] = points[:, 0] / 1000.0
                        points[:, 1] = points[:, 1] / 1000.0
                        strokes.append(points)
                    
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
        
        self.data = torch.from_numpy(np.array(self.data))
        self.labels = torch.tensor(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class QuickDrawLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super(QuickDrawLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

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
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            mse_total += nn.MSELoss()(torch.softmax(outputs, dim=1), 
                                     nn.functional.one_hot(labels, outputs.size(1)).float()).item()
            
            # Update progress bar
            train_acc = 100. * train_correct / train_total
            pbar.set_postfix({
                'epoch': f'{epoch+1}/{num_epochs}',
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{train_acc:.2f}%'
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
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_mse_total += nn.MSELoss()(torch.softmax(outputs, dim=1), 
                                             nn.functional.one_hot(labels, outputs.size(1)).float()).item()
                pbar.update(1)
        
        val_acc = 100. * val_correct / val_total
        
        epoch_summary = {
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader),
            'train_accuracy': 100. * train_correct / train_total,
            'train_mse': mse_total/len(train_loader),
            'val_loss': val_loss/len(val_loader),
            'val_accuracy': val_acc,
            'val_mse': val_mse_total/len(val_loader)
        }
        training_history.append(epoch_summary)
        
        # Early stopping
        if epoch >= min_epochs:  # Only consider early stopping after minimum epochs
            if val_acc > best_val_acc + min_delta:  # Require minimum improvement
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

def train_final_model(best_params, dataset, device, output_dir):
    """Train the best model configuration for extended epochs to analyze convergence"""
    print("\nTraining final model with best parameters for 50 epochs...")
    
    # Convert parameters to appropriate types
    model_params = {
        'batch_size': int(best_params['batch_size']),
        'hidden_size': int(best_params['hidden_size']),
        'num_layers': int(best_params['num_layers']),
        'dropout': float(best_params['dropout']),
        'learning_rate': float(best_params['learning_rate'])
    }

    # Split data
    train_dataset, val_dataset = train_test_split(dataset, train_size=0.8, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'])
    
    # Initialize model with best parameters
    model = QuickDrawLSTM(
        input_size=2,
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        num_classes=10,
        dropout=model_params['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    
    # Train for 50 epochs
    _, history = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, device=device, early_stopping_patience=float('inf')
    )
    
    # Plot convergence
    history_df = pd.DataFrame(history)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    plt.savefig(os.path.join(output_dir, f'convergence_plot_{timestamp}.png'))
    plt.close()
    
    # Save training history
    history_df.to_csv(os.path.join(
        output_dir.replace('plots', 'stats'), 
        f'final_model_history_{timestamp}.csv'
    ), index=False)

def main():
    # Define hyperparameter grid
    param_grid = {
        'batch_size': [32, 64, 128],
        'learning_rate': [0.01, 0.001],
        'hidden_size': [64, 128, 256],
        'num_layers': [2],
        'dropout': [0.0]
    }
    
    # Create output directories
    plots_dir = 'tuning/output/plots'
    stats_dir = 'tuning/output/stats'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = QuickDrawDataset('processed_data/inliers', samples_per_class=1000)
    
    # Generate parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in itertools.product(*param_grid.values())]
    
    results = []
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

        # Ensure correct types for model parameters
        model_params = {
            'batch_size': int(params['batch_size']),
            'hidden_size': int(params['hidden_size']),
            'num_layers': int(params['num_layers']),
            'dropout': float(params['dropout']),
            'learning_rate': float(params['learning_rate'])
        }
        
        # Split data and create loaders
        train_dataset, val_dataset = train_test_split(dataset, train_size=0.8, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Initialize model
        model = QuickDrawLSTM(
            input_size=2,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=10,
            dropout=params['dropout']
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
        
        # Print summary after each combination
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
    train_final_model(best_config, dataset, device, plots_dir)

if __name__ == '__main__':
    main()
