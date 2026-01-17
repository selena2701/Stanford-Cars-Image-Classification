# Stanford Cars Image Classification

A production-ready fine-grained image classification system that classifies car images into **196 fine-grained categories** (make, model, year) using deep learning techniques.

## Project Information

- **Course:** CO5177 — Nền tảng lập trình cho phân tích và trực quan dữ liệu
- **Lecturer:** Dr. Le Thanh Sach
- **Author:** Lê Thị Hồng Cúc (Student ID: 2470882)

## Goal

Build a production-ready classification system that:
- Classifies car images into **196 fine-grained categories** (make, model, year)
- Achieves high accuracy through modern deep learning techniques
- Provides model interpretability insights
- Deploys as an interactive web application

## Key Features

### ML Fundamentals
- Normalization techniques (BatchNorm, LayerNorm, InstanceNorm, GroupNorm)
- Detection geometry (IoU, anchors)
- Batch utilities and data processing

### Python & NumPy
- Decorators and OOP
- Advanced NumPy operations (broadcasting, masking, indexing)

### Image Processing
- Batch manipulation
- Layout conversion (NHWC↔NCHW)
- Visualization helpers

### Classification Pipeline
- Full 196-class support
- Data augmentation
- Class imbalance handling
- Model interpretability

### Production Ready
- End-to-end pipeline from EDA to deployment
- Interactive Streamlit web application

## Technology Stack

- **Framework:** PyTorch
- **Model Library:** timm (PyTorch Image Models)
- **Augmentation:** Albumentations
- **Interpretability:** Captum
- **Deployment:** Streamlit
- **Backbone:** ResNet50 (configurable)
- **Visualization:** Altair, Matplotlib

## Dataset

**Stanford Cars Dataset**
- **Training Set:** 8,144 images
- **Test Set:** 8,041 images
- **Classes:** 196 fine-grained car categories
- **Format:** JPEG images with MAT annotation files

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Git LFS (for large model files)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/selena2701/PF4DA-TeamTMC.git
   cd PF4DA-TeamTMC
   ```

2. **Navigate to project directory**
   ```bash
   cd "Stanford Cars — Image Classification"
   ```

3. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Project Structure

```
PF4DA-TeamTMC/
├── Stanford Cars — Image Classification/
│   ├── streamlit_app.py              # Main Streamlit application
│   ├── StanfordCarsImageClassification.ipynb  # Jupyter notebook for training
│   ├── class_mappings.json           # Class name mappings
│   ├── artifacts/                    # Model and visualization files
│   │   ├── best_model.pth           # Trained model (Git LFS)
│   │   ├── classification_report.json
│   │   ├── confusion_matrix.png
│   │   └── ...                      # Other artifacts
│   ├── dataset/                     # Dataset (not in repo, too large)
│   │   ├── cars_train/
│   │   ├── cars_test/
│   │   └── car_devkit/
│   └── .streamlit/
│       └── config.toml              # Streamlit configuration
├── requirements.txt                  # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## Model Architecture

**CarClassifier**
- **Backbone:** ResNet50 (ImageNet pretrained)
- **Feature Extraction:** Global Average Pooling → 2048-dim features
- **Classifier Head:** 
  - Dropout(0.3) → Linear(2048→512) → ReLU → Dropout(0.2) → Linear(512→196)
- **Total Parameters:** ~25.9M

## Performance

- **Top-1 Accuracy:** 86.25%
- **Top-5 Accuracy:** 97.71%
- **Macro F1 Score:** 0.8601
- **Classes:** 196/196 classes successfully learned

## Deployment

### Streamlit Cloud

The app is deployed on Streamlit Cloud:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path: `Stanford Cars — Image Classification/streamlit_app.py`
5. Deploy!

**Note:** Model files are stored via Git LFS for efficient deployment.

## Usage

### Interactive Web App

The Streamlit app provides:
- **Dataset Overview:** Statistics and class distribution
- **EDA:** Exploratory data analysis with visualizations
- **Model Architecture:** Detailed architecture information
- **Training Visualizations:** Loss, accuracy, and learning rate curves
- **Evaluation:** Comprehensive metrics and confusion matrix
- **Test Inference:** Predictions on test set
- **Live Prediction:** Upload and classify your own car images

### Training

To train the model, run the Jupyter notebook:
```bash
jupyter notebook StanfordCarsImageClassification.ipynb
```

## Configuration

### Model Configuration
- **Backbone:** ResNet50
- **Image Size:** 224×224
- **Batch Size:** 96
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** OneCycleLR
- **Loss:** CrossEntropyLoss with class weights

### Data Augmentation
- RandomResizedCrop
- HorizontalFlip
- ShiftScaleRotate
- RandomBrightnessContrast
- HueSaturationValue
- CLAHE
- Gaussian noise/blur
- CoarseDropout

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- streamlit
- torch
- timm
- albumentations
- pandas
- numpy
- altair
- scipy
- Pillow
- opencv-python-headless

## Contributing

This is a course project. For questions or suggestions, please open an issue.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset:** Stanford Cars Dataset (Krause et al., 2013)
- **Framework:** PyTorch, timm, Streamlit
- **Course:** CO5177 — Nền tảng lập trình cho phân tích và trực quan dữ liệu

## Contact

**Author:** Lê Thị Hồng Cúc  
**Student ID:** 2470882  
**Course:** CO5177

---

**Built for fine-grained car classification**
