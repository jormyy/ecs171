from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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

class QuickDrawCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(QuickDrawCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class QuickDrawMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(QuickDrawMLP, self).__init__()
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
        x = x.view(x.size(0), -1)
        return self.model(x)

def load_lstm_model():
    input_size = 2
    hidden_size = 128
    num_layers = 2
    num_classes = 10
    
    model = QuickDrawLSTM(input_size, hidden_size, num_layers, num_classes)
    checkpoint = torch.load('models/best_model_LSTM.pth', 
                          map_location=torch.device('cpu'),
                          weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_cnn_model():
    input_channels = 1
    num_classes = 10
    model = QuickDrawCNN(input_channels, num_classes)
    checkpoint = torch.load('models/best_model_CNN.pth', 
                          map_location=torch.device('cpu'),
                          weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_mlp_model():
    input_size = 28 * 28
    hidden_sizes = [512, 256, 128]
    num_classes = 10
    
    model = QuickDrawMLP(input_size, hidden_sizes, num_classes)
    checkpoint = torch.load('models/best_model_MLP.pth', 
                          map_location=torch.device('cpu'),
                          weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Initialize models
logger.info("Loading models...")
models = {}
try:
    models['lstm'] = load_lstm_model()
    logger.info("LSTM model loaded successfully")
except Exception as e:
    logger.error(f"Error loading LSTM model: {str(e)}")

try:
    models['cnn'] = load_cnn_model()
    logger.info("CNN model loaded successfully")
except Exception as e:
    logger.error(f"Error loading CNN model: {str(e)}")

try:
    models['mlp'] = load_mlp_model()
    logger.info("MLP model loaded successfully")
except Exception as e:
    logger.error(f"Error loading MLP model: {str(e)}")

classes = ['bird', 'car', 'cat', 'clock', 'dog', 'face', 'fish', 'house', 'sun', 'tree']

def preprocess_for_lstm(drawing_data):
    processed_strokes = []
    for stroke in drawing_data:
        points = np.array([[point['x'] for point in stroke],
                          [point['y'] for point in stroke]]).T
        points = points.astype(np.float32)
        points[:, 0] = points[:, 0] / 1000.0
        points[:, 1] = points[:, 1] / 1000.0
        processed_strokes.append(points)
    
    max_seq_length = 200
    padded_stroke = np.zeros((max_seq_length, 2), dtype=np.float32)
    total_points = 0
    
    for stroke in processed_strokes:
        points_to_copy = min(stroke.shape[0], max_seq_length - total_points)
        if points_to_copy <= 0:
            break
        padded_stroke[total_points:total_points + points_to_copy] = stroke[:points_to_copy]
        total_points += points_to_copy
    
    return torch.FloatTensor(padded_stroke).unsqueeze(0)

def preprocess_for_cnn_mlp(drawing_data):
    # Create a blank image using PIL
    image = Image.new('L', (280, 280), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw the strokes
    for stroke in drawing_data:
        points = [(point['x'], point['y']) for point in stroke]
        if len(points) > 1:
            draw.line(points, fill='black', width=3)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Create transform pipeline
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Convert numpy array back to PIL Image for transform pipeline
    image = Image.fromarray(image_array)
    
    # Apply transforms
    img_tensor = transform(image)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    logger.info(f"Preprocessed tensor shape: {img_tensor.shape}")
    logger.info(f"Tensor min/max: {img_tensor.min().item():.4f}/{img_tensor.max().item():.4f}")
    logger.info(f"Tensor mean/std: {img_tensor.mean().item():.4f}/{img_tensor.std().item():.4f}")
    
    return img_tensor

@app.route('/predict', methods=['POST'])
def predict():
    try:
        drawing_data = request.json['drawing']
        model_type = request.json.get('model_type', 'lstm').lower()
        
        logger.info(f"Received prediction request for model type: {model_type}")
        
        if not models:
            return jsonify({'error': 'No models are currently loaded'}), 500
        
        if model_type not in models:
            available_models = list(models.keys())
            return jsonify({
                'error': f'Invalid model type. Choose from: {", ".join(available_models)}'
            }), 400
        
        # Preprocess based on model type
        if model_type == 'lstm':
            processed_drawing = preprocess_for_lstm(drawing_data)
        else:  # CNN or MLP
            processed_drawing = preprocess_for_cnn_mlp(drawing_data)
        
        # Make prediction
        with torch.no_grad():
            outputs = models[model_type](processed_drawing)
            probabilities = torch.softmax(outputs, dim=1)
            
            logger.info("\nPrediction probabilities:")
            for i, prob in enumerate(probabilities[0]):
                logger.info(f"{classes[i]}: {prob.item()*100:.2f}%")
            
            predicted_class = classes[torch.argmax(probabilities).item()]
            confidence = probabilities.max().item()
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(classes, probabilities[0].tolist())
            }
        })
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
