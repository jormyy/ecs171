# ECS171 Class Project - Quick Draw Reimplementation Classification Backend

This project implements a machine learning backend for classifying hand-drawn sketches using three different models: CNN, LSTM, and MLP.

## Project Objects

The models are trained to recognize the following 10 objects:

- **Bird**: Any flying bird
- **Car**: Side view of a car
- **Cat**: A cat face or body
- **Clock**: An analog clock
- **Dog**: A dog face or body
- **Face**: A human face
- **Fish**: Any swimming fish
- **House**: Simple house with roof
- **Sun**: The sun, can include rays
- **Tree**: Any type of tree

## Project Structure
```
ecs171/
├── EDA/                    # Exploratory Data Analysis
│   ├── eda.py             # EDA script
│   └── output/            # EDA outputs and visualizations
├── processing_scripts/     # Data processing scripts
│   ├── convert_to_image.py # Convert drawings to images
│   └── outlier.py         # Outlier detection
├── training/              # Model training scripts
│   ├── train_CNN.py       # CNN model training
│   ├── train_LSTM.py      # LSTM model training
│   └── train_MLP.py       # MLP model training
├── models/                # Trained model weights
├── server.py              # Flask server for model inference
└── requirements.txt       # Python dependencies
```

## Prerequisites

- Python 3.9+
- Anaconda/Miniconda or Python venv
- CUDA-capable GPU (optional, for faster training)

## Setup Options

### Option 1: Using myenv (Python venv)

1. Clone the repository:
   ```bash
   git clone https://github.com/jormyy/ecs171.git
   cd ecs171
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv myenv
   
   # On Windows
   myenv\Scripts\activate
   
   # On Unix or MacOS
   source myenv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecs171.git
   cd ecs171
   ```

2. Create and activate a conda environment:
   ```bash
   conda create --name ecs171 python=3.9
   conda activate ecs171
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. Download the dataset from:
   https://drive.google.com/drive/folders/1RTILMf8xOB4rua_RgdXMv5xBvs40wfax?usp=drive_link

2. Create a "raw" folder in the home directory of the repo:
   ```bash
   mkdir raw
   ```

3. Add all the downloaded dataset files to the "raw" folder. Make sure to keep the name of the folder as "raw".

## Running the Server

1. Ensure all model weights are present in the `models` directory:
   - `best_model_CNN.pth`
   - `best_model_LSTM.pth`
   - `best_model_MLP.pth`

2. Start the Flask server:
   ```bash
   python server.py
   ```

The server will run on `http://localhost:5000` by default.

## API Endpoints

### POST /predict
Accepts drawing data and returns classification predictions.

Request body:
```json
{
  "drawing": [
    [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
    [{"x": 2, "y": 2}, {"x": 3, "y": 3}]
  ],
  "model_type": "lstm"  // Options: "lstm", "cnn", "mlp"
}
```

Response:
```json
{
  "prediction": "cat",
  "confidence": 0.95,
  "probabilities": {
    "bird": 0.01,
    "cat": 0.95,
    "dog": 0.02,
    ...
  }
}
```

## Data Processing

1. Run outlier detection:
   ```bash
   python processing_scripts/outlier.py
   ```

2. Convert drawings to images:
   ```bash
   python processing_scripts/convert_to_image.py
   ```

## Training Models

1. Train CNN:
   ```bash
   python training/train_CNN.py
   ```

2. Train LSTM:
   ```bash
   python training/train_LSTM.py
   ```

3. Train MLP:
   ```bash
   python training/train_MLP.py
   ```

Training logs and model weights will be saved in the `models` directory.

## Deactivating the Environment

When you're done working with the project:

For myenv:
```bash
deactivate
```

For Conda:
```bash
conda deactivate
```
