# Plant Disease Detection System

An AI-powered web application for detecting plant diseases using deep learning and computer vision.

## Overview

This project provides a complete solution for plant disease detection using Convolutional Neural Networks (CNN). Simply upload a photo of a plant leaf and get instant disease diagnosis with treatment recommendations.

## Features

- **Disease Detection**: Identifies 15+ different plant diseases
- **Multiple Plants**: Supports Tomato, Potato, and Bell Pepper plants
- **Treatment Advice**: Provides specific treatment recommendations
- **Web Interface**: Easy-to-use web application
- **Real-time Results**: Get predictions in seconds
- **High Accuracy**: 90%+ accuracy on validation data

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Flask 2.x
- PIL (Pillow)
- NumPy
- Matplotlib

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install tensorflow flask pillow numpy matplotlib
```

## Quick Start

### Step 1: Prepare Dataset
```bash
# Create dataset structure
dataset/
├── train/
│   ├── Tomato_healthy/
│   ├── Tomato_Early_blight/
│   └── ... (other classes)
└── val/
    ├── Tomato_healthy/
    ├── Tomato_Early_blight/
    └── ... (other classes)
```

### Step 2: Train Model
```bash
python train_model.py
```

### Step 3: Run Web App
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## Project Structure

```
plant-disease-detection/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── models/                # Saved models and metadata
│   ├── plant_disease_cnn.h5
│   ├── best_model.h5
│   ├── class_names.txt
│   └── training_info.json
├── static/
│   └── uploads/           # Uploaded images
├── templates/             # HTML templates
└── dataset/               # Training data
    ├── train/
    └── val/
```

## Supported Diseases

### Tomato Diseases
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Mosaic Virus
- Yellow Leaf Curl Virus
- Healthy

### Potato Diseases
- Early Blight
- Late Blight
- Healthy

### Bell Pepper Diseases
- Bacterial Spot
- Healthy

## API Endpoints

### POST /upload
Upload an image for disease detection.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
    "success": true,
    "plant": "Tomato",
    "disease": "Early Blight",
    "confidence": "85.67%",
    "treatment": "Use fungicide and improve ventilation"
}
```

### GET /health
Check application health status.

### GET /model-info
Get model information and statistics.

### GET /stats
Get usage statistics and analytics.

## Training Parameters

```python
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 16
EPOCHS = 10
```

## Model Architecture

- **Input Layer**: 224x224x3 images
- **Conv Block 1**: 32 filters, BatchNorm, MaxPool
- **Conv Block 2**: 64 filters, BatchNorm, MaxPool
- **Conv Block 3**: 128 filters, BatchNorm, MaxPool
- **Conv Block 4**: 256 filters, BatchNorm, MaxPool
- **Dense Layers**: 512 → 256 → num_classes
- **Activation**: ReLU (hidden), Softmax (output)
- **Optimizer**: Adam (lr=0.001)

## Data Augmentation

- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Horizontal flip: Yes
- Zoom range: ±20%
- Brightness: 0.8-1.2x

## Usage Example

1. **Start the application**
```bash
python app.py
```

2. **Open browser** and go to `http://localhost:5000`

3. **Upload image** by dragging and dropping or clicking browse

4. **View results** including:
   - Plant type
   - Disease name
   - Confidence level
   - Treatment recommendation

## Troubleshooting

### Common Issues

**Model not found**
```bash
# Train the model first
python train_model.py
```

**Memory errors during training**
```python
# Reduce batch size in train_model.py
BATCH_SIZE = 8  # Instead of 16
```

**Low accuracy**
- Increase training epochs
- Add more training data
- Use transfer learning with pre-trained models

**Upload errors**
- Check file format (PNG, JPG, JPEG, GIF only)
- Ensure file size < 16MB

## Performance Tips

1. **For better accuracy:**
   - Use more training data
   - Increase epochs (20-50)
   - Try transfer learning

2. **For faster training:**
   - Use GPU if available
   - Reduce image size
   - Use smaller batch size

3. **For production:**
   - Use model quantization
   - Implement caching
   - Add load balancing

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Future Improvements

- [ ] Add more plant types
- [ ] Implement transfer learning
- [ ] Mobile app version
- [ ] Batch processing
- [ ] Disease severity assessment
- [ ] Multi-language support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support:
- Email: your.email@example.com
- GitHub: @yourusername

---

**Note**: Make sure to have adequate training data for best results. The model performance depends on the quality and quantity of training images.
