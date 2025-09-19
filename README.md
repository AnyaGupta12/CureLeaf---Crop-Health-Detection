# CureLeaf - Crop Health Detection 

A comprehensive plant disease detection system that combines deep learning with AI-powered treatment recommendations to help farmers and gardeners identify and treat plant diseases effectively.

##  Features

- **Disease Detection**: Identifies 38 different plant diseases across 14 crop types
- **AI-Powered Recommendations**: Uses Google's Gemini AI to provide personalized treatment advice
- **User-Friendly Interface**: Interactive Streamlit web application
- **Real-time Predictions**: Upload images and get instant disease classification
- **Comprehensive Treatment Info**: Organic solutions, chemical treatments, and preventive measures

##  Supported Crops & Diseases

The model can detect diseases in the following crops:

### Apple
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

### Cherry
- Powdery Mildew
- Healthy

### Corn (Maize)
- Cercospora Leaf Spot/Gray Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy

### Grape
- Black Rot
- Esca (Black Measles)
- Leaf Blight (Isariopsis Leaf Spot)
- Healthy

### Peach
- Bacterial Spot
- Healthy

### Pepper (Bell)
- Bacterial Spot
- Healthy

### Potato
- Early Blight
- Late Blight
- Healthy

### Tomato
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted Spider Mite)
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy

### Other Crops
- Blueberry (Healthy)
- Orange (Citrus Greening)
- Raspberry (Healthy)
- Soybean (Healthy)
- Squash (Powdery Mildew)
- Strawberry (Leaf Scorch, Healthy)

##  Technology Stack

- **Deep Learning**: TensorFlow/Keras with EfficientNetB0
- **Frontend**: Streamlit
- **AI Integration**: Google Generative AI (Gemini 2.0 Flash)
- **Image Processing**: PIL, OpenCV-compatible preprocessing
- **Language**: Python 3.x

##  Project Structure

```
CureLeaf---Crop-Health-Detection/
├── app_g.py                        # Streamlit web application
├── plant-disease-detection.ipynb   # Model training notebook
├── best_model.keras                # Trained model file (generated)
└── README.md                       # Project documentation
```

##  Quick Start

### Prerequisites

```bash
pip install streamlit tensorflow pillow numpy google-generativeai
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AnyaGupta12/CureLeaf---Crop-Health-Detection.git
   cd CureLeaf---Crop-Health-Detection
   ```

2. **Configure Google API**
   - Get a Google Generative AI API key from [Google AI Studio](https://makersuite.google.com/)
   - Update the `GOOGLE_API_KEY` variable in `app_g.py`:
   ```python
   GOOGLE_API_KEY = "your_api_key_here"
   ```

3. **Ensure model file exists**
   - The app expects `best_model.keras` in the root directory
   - Train the model using the Jupyter notebook or download a pre-trained version

4. **Run the application**
   ```bash
   streamlit run app_g.py
   ```

5. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload a leaf image and get instant disease detection results

##  Model Training

The training process is documented in `plant-disease-detection.ipynb`:

### Dataset
- **Source**: New Plant Diseases Dataset (Augmented)
- **Size**: Large-scale dataset with augmented images
- **Classes**: 38 different disease/healthy combinations

### Architecture
- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned for plant disease classification
- **Input Size**: 224x224x3
- **Architecture**:
  - Data Augmentation Layer
  - EfficientNetB0 (unfrozen for fine-tuning)
  - Global Average Pooling
  - Dropout (0.2)
  - Dense Layer (38 classes, softmax)

### Training Configuration
- **Optimizer**: Adam (learning rate: 1e-5)
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: 
  - Early Stopping (patience=3)
  - Model Checkpoint (save best model)
- **Epochs**: 10 (with early stopping)

### Data Augmentation
- Random horizontal flip
- Random rotation (10%)
- Random zoom (10%)

##  How It Works

1. **Image Upload**: User uploads a leaf image through the Streamlit interface
2. **Preprocessing**: Image is resized to 224x224 and preprocessed using EfficientNet preprocessing
3. **Disease Detection**: The trained CNN model predicts the disease class and confidence
4. **AI Consultation**: Google Gemini AI generates detailed treatment recommendations
5. **Results Display**: 
   - Disease classification and confidence score
   - Health status (Healthy/Diseased)
   - Comprehensive treatment advice including:
     - Organic/natural treatments
     - Chemical solutions
     - Preventive measures
     - Recommended next steps

##  Model Performance

The model achieves high accuracy on the validation dataset. Key metrics:
- **Architecture**: EfficientNetB0-based transfer learning
- **Training Strategy**: Fine-tuning with data augmentation
- **Optimization**: Mixed precision training support

##  Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Google Generative AI API key for treatment recommendations

### Model Configuration
- **Input Shape**: (224, 224, 3)
- **Batch Size**: 32 (during training)
- **Preprocessing**: EfficientNet-specific preprocessing

##  Usage Examples

### Web Application
1. Start the Streamlit app: `streamlit run app_g.py`
2. Upload an image of a plant leaf
3. View the disease classification results
4. Read the AI-generated treatment recommendations

### Programmatic Usage
```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("best_model.keras")

# Preprocess image
img = Image.open("leaf_image.jpg").resize((224, 224))
img_array = np.expand_dims(np.array(img), axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


##  Acknowledgments

- **Dataset**: New Plant Diseases Dataset (Augmented)
- **Base Model**: EfficientNetB0 from TensorFlow/Keras
- **AI Integration**: Google Generative AI (Gemini)
- **Framework**: Streamlit for the web interface

##  Support

For support, please open an issue in the GitHub repository or contact the maintainers.

##  Future Enhancements

- [ ] Mobile app development
- [ ] Real-time camera integration
- [ ] Additional crop types and diseases
- [ ] Multi-language support
- [ ] Offline model deployment
- [ ] Integration with IoT sensors
- [ ] Historical disease tracking
- [ ] Weather-based recommendations

---