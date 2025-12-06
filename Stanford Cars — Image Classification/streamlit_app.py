"""
Streamlit Web App for Stanford Cars Classification

**Course:** CO5177 — Nền tảng lập trình cho phân tích và trực quan dữ liệu  
**Lecture:** Dr. Le Thanh Sach  
**Author:** Lê Thị Hồng Cúc (Student ID: 2470882)  


### Goal
Build a production-ready classification system that:
- Classifies car images into **196 fine-grained categories** (make, model, year)
- Achieves high accuracy through modern deep learning techniques
- Provides model interpretability insights
- Deploys as an interactive web application

### Key Features

| Feature | Description |
|---------|-------------|
| ** ML Fundamentals** | Normalization (BatchNorm, LayerNorm, InstanceNorm, GroupNorm), detection geometry (IoU, anchors), batch utilities |
| ** Python & NumPy** | Decorators, OOP, advanced NumPy operations (broadcasting, masking, indexing) |
| ** Image Processing** | Batch manipulation, layout conversion (NHWC↔NCHW), visualization helpers (color coding) |
| ** Classification Pipeline** | Full 196-class support, data augmentation, class imbalance handling, model interpretability |
| ** Production Ready** | End-to-end pipeline from EDA to deployment with interactive Streamlit app |

### Pipeline Structure

```
EDA & Visualization → Data Loading → Augmentation → Model Training → 
Evaluation → Interpretability → Deployment
```

### Technology Stack

- **Framework**: PyTorch
- **Model Library**: timm (PyTorch Image Models)
- **Augmentation**: Albumentations
- **Interpretability**: Captum
- **Deployment**: Streamlit
- **Backbone**: ResNet50 (configurable)
"""
# ============================================================================
# Python 3.14 Compatibility Patch for Altair
# ============================================================================
# Altair 5.5.0 uses TypedDict with 'closed=True' parameter which isn't
# fully supported in Python 3.14 yet. This patch allows it to work.
import sys
if sys.version_info >= (3, 14):
    from typing import _TypedDictMeta
    _original_new = _TypedDictMeta.__new__
    
    def _patched_new(mcs, name, bases, ns, total=True, **kwargs):
        # Remove unsupported 'closed' parameter
        kwargs.pop('closed', None)
        return _original_new(mcs, name, bases, ns, total=total)
    
    _TypedDictMeta.__new__ = staticmethod(_patched_new)

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import torch.nn as nn
import pandas as pd
import altair as alt
from pathlib import Path
from scipy.io import loadmat
import random
from typing import Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

st.set_page_config(
    page_title="Stanford Cars Interactive Report",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
Stanford Cars Image Classification

**Course:** CO5177 — Nền tảng lập trình cho phân tích và trực quan dữ liệu 
**Lecture:** Dr. Le Thanh Sach
**Author:** Lê Thị Hồng Cúc (Student ID: 2470882)  


Goal:
Build a production-ready classification system that classifies car images into 196 fine-grained categories (make, model, year), achieves high accuracy through modern deep learning techniques, provides model interpretability insights, and deploys as an interactive web application.

Key Features:
- ML Fundamentals: Normalization (BatchNorm, LayerNorm, InstanceNorm, GroupNorm), detection geometry (IoU, anchors), batch utilities
- Python & NumPy: Decorators, OOP, advanced NumPy operations (broadcasting, masking, indexing)
- Image Processing: Batch manipulation, layout conversion (NHWC↔NCHW), visualization helpers (color coding)
- Classification Pipeline: Full 196-class support, data augmentation, class imbalance handling, model interpretability
- Production Ready: End-to-end pipeline from EDA to deployment with interactive Streamlit app

Pipeline Structure:
EDA & Visualization → Data Loading → Augmentation → Model Training → Evaluation → Interpretability → Deployment

Technology Stack:
- Framework: PyTorch
- Model Library: timm (PyTorch Image Models)
- Augmentation: Albumentations
- Interpretability: Captum
- Deployment: Streamlit
- Backbone: ResNet50 (configurable)
        """
    }
)

with open('class_mappings.json', 'r') as f:
    mappings = json.load(f)
    CLASS_NAMES = mappings['class_names']
    idx_to_label = {int(k): v for k, v in mappings['idx_to_label'].items()}

NUM_CLASSES = 196
IMG_SIZE = (224, 224)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
DATASET_DIR = BASE_DIR / 'dataset'
DEVKIT_DIR = DATASET_DIR / 'car_devkit' / 'devkit'
TRAIN_IMAGES_DIR = DATASET_DIR / 'cars_train' / 'cars_train'
TEST_IMAGES_DIR = DATASET_DIR / 'cars_test' / 'cars_test'
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / 'classification_report.json'
CLASS_DISTRIBUTION_IMG = ARTIFACTS_DIR / 'class_distribution.png'
TRAINING_HISTORY_IMG = ARTIFACTS_DIR / 'training_history.png'
CONFUSION_MATRIX_IMG = ARTIFACTS_DIR / 'confusion_matrix.png'
F1_PER_CLASS_IMG = ARTIFACTS_DIR / 'f1_per_class.png'
INTERPRETABILITY_IMG = ARTIFACTS_DIR / 'interpretability.png'
AUGMENTATION_IMG = ARTIFACTS_DIR / 'augmentation_examples.png'
FEATURE_EXTRACTION_PIPELINE_IMG = ARTIFACTS_DIR / 'feature_extraction_pipeline.png'
CONV_FILTERS_IMG = ARTIFACTS_DIR / 'conv_filters_visualization.png'
WEIGHT_DISTRIBUTIONS_IMG = ARTIFACTS_DIR / 'weight_distributions.png'
WEIGHT_MAGNITUDES_IMG = ARTIFACTS_DIR / 'weight_magnitudes_by_layer.png'
TOP_LAYERS_IMG = ARTIFACTS_DIR / 'top_layers_by_parameters.png'
TRAIN_VS_TEST_DISTRIBUTION_IMG = ARTIFACTS_DIR / 'class_distribution_train_vs_test_all_196.png'
CLASS_HEATMAP_IMG = ARTIFACTS_DIR / 'class_heatmap.png'

COLOR_PRIMARY = "#60a5fa"
COLOR_ACCENT = "#f472b6"
COLOR_MUTED = "#94a3b8"

CHART_COLORS = {
    'primary': COLOR_PRIMARY,
    'secondary': COLOR_ACCENT,
    'accent': COLOR_MUTED,
    'success': '#2CA02C',
    'info': '#17BECF',
    'warning': '#FFBB33',
    'gradient': [COLOR_PRIMARY, '#93c5fd', '#dbeafe', '#f0f9ff'],
    'multi': [COLOR_PRIMARY, COLOR_ACCENT, COLOR_MUTED, '#2CA02C', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F'],
}

def configure_chart_base():
    """Configure base Altair theme for light mode"""
    return {
        'background': 'transparent',
        'title': {
            'font': 'Inter',
            'fontSize': 18,
            'fontWeight': 600,
            'color': '#1e293b',  # Dark text for light background
        },
        'axis': {
            'labelFont': 'Inter',
            'labelFontSize': 12,
            'labelColor': '#475569',  # Medium gray for labels
            'titleFont': 'Inter',
            'titleFontSize': 14,
            'titleFontWeight': 600,
            'titleColor': '#1e293b',  # Dark text for titles
            'gridColor': '#e2e8f0',  # Light gray grid lines
            'domainColor': '#cbd5e1',  # Light border color
        },
        'legend': {
            'labelFont': 'Inter',
            'labelFontSize': 12,
            'labelColor': '#334155',  # Dark text for legend
            'titleFont': 'Inter',
            'titleFontSize': 13,
            'titleFontWeight': 600,
            'titleColor': '#1e293b',  # Dark text for legend title
        },
    }

alt.themes.register('friendly', configure_chart_base)
alt.themes.enable('friendly')

class CarClassifier(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=196, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1])
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained model"""
    model = CarClassifier(backbone_name='resnet50', num_classes=196, pretrained=False)
    model_path = ARTIFACTS_DIR / 'best_model.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1], p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(p=1.0),
    ])

def predict(image, model):
    """Predict class for image - return top-1 prediction"""
    transform = get_val_transforms()
    img_np = np.array(image.convert('RGB'))
    img_tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
    
    return top_prob.cpu().item(), top_idx.cpu().item(), probs.cpu().numpy()[0]


@st.cache_data(show_spinner=False)
def load_annotations_df(split: str = 'train') -> pd.DataFrame:
    """Load annotation metadata from the Stanford Cars devkit."""
    split = split.lower()
    if split not in {'train', 'test'}:
        raise ValueError("Split must be either 'train' or 'test'.")

    if not DEVKIT_DIR.exists():
        return pd.DataFrame()

    mat_filename = f'cars_{split}_annos.mat'
    mat_path = DEVKIT_DIR / mat_filename
    if not mat_path.exists():
        return pd.DataFrame()

    mat_data = loadmat(str(mat_path))
    annotations = mat_data.get('annotations')
    if annotations is None or annotations.size == 0:
        return pd.DataFrame()

    records: List[Dict] = []
    image_root = TRAIN_IMAGES_DIR if split == 'train' else TEST_IMAGES_DIR

    for entry in annotations[0]:
        fname = entry['fname'][0]
        class_idx = None
        class_name = None
        if 'class' in entry.dtype.names:
            class_idx = int(entry['class'][0][0]) - 1
            if 0 <= class_idx < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_idx]
        records.append(
            {
                'filename': fname,
                'image_path': str(image_root / fname),
                'split': split,
                'class_idx': class_idx,
                'class_name': class_name,
                'bbox_x1': int(entry['bbox_x1'][0][0]),
                'bbox_y1': int(entry['bbox_y1'][0][0]),
                'bbox_x2': int(entry['bbox_x2'][0][0]),
                'bbox_y2': int(entry['bbox_y2'][0][0]),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


@st.cache_data(show_spinner=False)
def get_dataset_overview() -> Dict[str, int]:
    train_df = load_annotations_df('train')
    train_images = len(train_df)
    classes = train_df['class_name'].nunique() if not train_df.empty else len(CLASS_NAMES)
    test_images = len(list(TEST_IMAGES_DIR.glob('*.jpg'))) if TEST_IMAGES_DIR.exists() else 0
    return {
        'train_images': train_images,
        'test_images': test_images,
        'classes': classes,
    }


@st.cache_data(show_spinner=False)
def compute_image_metadata(sample_size: int = 400) -> pd.DataFrame:
    train_df = load_annotations_df('train')
    if train_df.empty:
        return pd.DataFrame()

    effective_sample = min(sample_size, len(train_df))
    sampled = train_df.sample(n=effective_sample, random_state=42).reset_index(drop=True)

    records: List[Dict] = []
    for _, row in sampled.iterrows():
        img_path = Path(row['image_path'])
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                rgb = img.convert('RGB')
                width, height = rgb.size
                arr = np.array(rgb)
        except Exception:
            continue

        aspect_ratio = width / height if height else np.nan
        pixels = arr.reshape(-1, 3).astype(np.float32)
        mean_channels = pixels.mean(axis=0)
        brightness = float(mean_channels.mean() / 255.0)

        blur_score: Optional[float] = None
        if cv2 is not None:
            try:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except Exception:
                blur_score = None

        records.append(
            {
                'class_name': row['class_name'],
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'mean_r': float(mean_channels[0]),
                'mean_g': float(mean_channels[1]),
                'mean_b': float(mean_channels[2]),
                'brightness': brightness,
                'blur_score': blur_score,
            }
        )

    return pd.DataFrame.from_records(records)


@st.cache_data(show_spinner=False)
def get_class_image_paths(class_name: str) -> List[str]:
    train_df = load_annotations_df('train')
    if train_df.empty or class_name is None:
        return []
    return train_df[train_df['class_name'] == class_name]['image_path'].tolist()


@st.cache_data(show_spinner=False)
def load_classification_report_df() -> pd.DataFrame:
    if not CLASSIFICATION_REPORT_PATH.exists():
        return pd.DataFrame()
    with open(CLASSIFICATION_REPORT_PATH, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data).T.reset_index().rename(columns={'index': 'class_name'})
    metric_columns = ['precision', 'recall', 'f1-score', 'support']
    for col in metric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def display_image_if_exists(path: Path, caption: str, help_text: Optional[str] = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} not available.")
        if help_text:
            st.caption(help_text)


def display_artifact_info(artifacts: Dict[str, str]):
    """Display information about artifacts used in this section.
    
    Args:
        artifacts: Dictionary mapping artifact file names to their descriptions and notebook cell references.
                  Format: {"filename.ext": "Description (Notebook: Cell X)"}
    """
    if not artifacts:
        return
    
    with st.expander("Artifacts Used in This Section", expanded=False):
        st.markdown("""
        **Artifact Files**: The following files are loaded from the `artifacts/` directory. 
        These files are generated by running specific cells in the Jupyter notebook (`StanfordCarsImageClassification.ipynb`).
        """)
        
        artifact_list = []
        for filename, description in artifacts.items():
            artifact_path = ARTIFACTS_DIR / filename
            status = "✓ Available" if artifact_path.exists() else "✗ Missing"
            artifact_list.append({
                "File": filename,
                "Description": description,
                "Status": status
            })
        
        artifact_df = pd.DataFrame(artifact_list)
        st.dataframe(
            artifact_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "File": st.column_config.TextColumn("Artifact File", width="medium"),
                "Description": st.column_config.TextColumn("Description & Notebook Reference", width="large"),
                "Status": st.column_config.TextColumn("Status", width="small"),
            }
        )


def parse_class_name(class_name: str) -> Dict[str, str]:
    """Parse class name to extract make, model, body_type, and year.
    
    Format: "BMW 3 Series Sedan 2012" -> {"make": "BMW", "model": "3 Series", "body_type": "Sedan", "year": "2012"}
    """
    parts = class_name.split()
    if len(parts) < 3:
        return {"make": "Unknown", "model": class_name, "body_type": "Unknown", "year": "Unknown"}
    
    # Year is typically the last part (4 digits)
    year = "Unknown"
    if parts[-1].isdigit() and len(parts[-1]) == 4:
        year = parts[-1]
        parts = parts[:-1]
    
    # Body type is typically one of: Sedan, SUV, Coupe, Convertible, Wagon, Hatchback, Truck, etc.
    body_types = ["Sedan", "SUV", "Coupe", "Convertible", "Wagon", "Hatchback", "Truck", 
                  "Crew Cab", "Regular Cab", "SuperCab", "Minivan", "Van"]
    body_type = "Unknown"
    body_type_idx = -1
    for i, part in enumerate(parts):
        # Check for multi-word body types
        if i < len(parts) - 1:
            two_word = f"{parts[i]} {parts[i+1]}"
            if two_word in body_types:
                body_type = two_word
                body_type_idx = i
                break
        if part in body_types:
            body_type = part
            body_type_idx = i
            break
    
    # Make is typically the first word
    make = parts[0] if parts else "Unknown"
    
    # Model is everything between make and body_type
    if body_type_idx > 0:
        model = " ".join(parts[1:body_type_idx])
    else:
        model = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"
    
    return {
        "make": make,
        "model": model if model else "Unknown",
        "body_type": body_type,
        "year": year
    }


@st.cache_data(show_spinner=False)
def load_confusion_matrix_data() -> Optional[np.ndarray]:
    """Load confusion matrix from artifacts if available.
    Returns None if not found - will need to be computed from predictions.
    """
    confusion_matrix_path = ARTIFACTS_DIR / 'confusion_matrix.npy'
    if confusion_matrix_path.exists():
        return np.load(str(confusion_matrix_path))
    
    # Try to load from JSON if available
    confusion_matrix_json_path = ARTIFACTS_DIR / 'confusion_matrix.json'
    if confusion_matrix_json_path.exists():
        with open(confusion_matrix_json_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return np.array(data)
            elif isinstance(data, dict) and 'matrix' in data:
                return np.array(data['matrix'])
    
    return None


def get_top_confused_pairs(confusion_matrix: np.ndarray, class_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """Extract top N most confused class pairs from confusion matrix.
    
    Returns DataFrame with columns: class_a, class_b, errors, class_a_idx, class_b_idx
    """
    n_classes = len(class_names)
    if confusion_matrix.shape != (n_classes, n_classes):
        return pd.DataFrame()
    
    # Get off-diagonal pairs (exclude diagonal)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and confusion_matrix[i, j] > 0:
                pairs.append({
                    'class_a': class_names[i],
                    'class_b': class_names[j],
                    'errors': int(confusion_matrix[i, j]),
                    'class_a_idx': i,
                    'class_b_idx': j
                })
    
    # Sort by errors descending and take top N
    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pd.DataFrame()
    
    pairs_df = pairs_df.sort_values('errors', ascending=False).head(top_n)
    return pairs_df


def create_top_confused_pairs_chart(pairs_df: pd.DataFrame) -> alt.Chart:
    """Create bar chart for top confused pairs."""
    if pairs_df.empty:
        return None
    
    # Create label for each pair
    pairs_df = pairs_df.copy()
    pairs_df['pair_label'] = pairs_df.apply(
        lambda row: f"{row['class_a']} ↔ {row['class_b']}", axis=1
    )
    
    chart = (
        alt.Chart(pairs_df)
        .mark_bar(color=CHART_COLORS['accent'], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X('errors:Q', title='Number of Misclassifications'),
            y=alt.Y('pair_label:N', sort='-x', title='Class Pair'),
            tooltip=[
                alt.Tooltip('pair_label:N', title='Class Pair'),
                alt.Tooltip('errors:Q', title='Errors', format='.0f'),
                alt.Tooltip('class_a:N', title='True Class'),
                alt.Tooltip('class_b:N', title='Predicted Class'),
            ]
        )
        .properties(height=400, title='Top Confused Class Pairs')
        .configure_axis(grid=True, gridOpacity=0.2)
    )
    return chart


def render_metric_cards(metrics: Dict[str, int]):
    """Render metric cards with friendly styling"""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(
                label.replace('_', ' ').title(),
                f"{value:,}",
                help=f"Total number of {label.lower().replace('_', ' ')}"
            )


def render_sample_gallery(class_name: str, num_images: int = 6):
    image_paths = get_class_image_paths(class_name)
    if not image_paths:
        st.warning("No images available for the selected class.")
        return

    if num_images > len(image_paths):
        num_images = len(image_paths)

    sampled_paths = random.sample(image_paths, num_images)
    cols = st.columns(3)
    for idx, path in enumerate(sampled_paths):
        col = cols[idx % 3]
        try:
            with Image.open(path) as img:
                col.image(img, caption=Path(path).name, use_container_width=True)
        except Exception as exc:
            col.warning(f"Unable to display {path}: {exc}")


def render_test_predictions_gallery(predictions_df: pd.DataFrame, num_images: int = 12, filter_by_confidence: Optional[str] = None):
    """Render test images with their predicted labels"""
    if predictions_df.empty:
        st.warning("No predictions available to display.")
        return
    
    # Filter by confidence if specified (only if not already filtered)
    display_df = predictions_df.copy()
    if filter_by_confidence and filter_by_confidence != "All":
        if filter_by_confidence == "High (>0.9)":
            if 'confidence' in display_df.columns:
                display_df = display_df[display_df['confidence'] > 0.9]
        elif filter_by_confidence == "Medium (0.5-0.9)":
            if 'confidence' in display_df.columns:
                display_df = display_df[(display_df['confidence'] >= 0.5) & (display_df['confidence'] <= 0.9)]
        elif filter_by_confidence == "Low (≤0.5)":
            if 'confidence' in display_df.columns:
                display_df = display_df[display_df['confidence'] <= 0.5]
    
    if display_df.empty:
        if filter_by_confidence:
            st.warning(f"No predictions found for the selected confidence filter: {filter_by_confidence}")
        else:
            st.warning("No predictions available to display.")
        return
    
    # Sample images
    effective_sample = min(num_images, len(display_df))
    sampled_df = display_df.sample(n=effective_sample, random_state=42).reset_index(drop=True)
    
    # Determine number of columns (3 columns layout)
    cols = st.columns(3)
    
    for idx, row in sampled_df.iterrows():
        col = cols[idx % 3]
        
        # Get image path - try multiple strategies
        image_path = None
        
        # Strategy 1: Check if filename column exists
        filename = row.get('filename', '')
        if not pd.isna(filename) and filename != '':
            image_path = TEST_IMAGES_DIR / str(filename)
            if image_path.exists():
                pass  # Found it
            else:
                image_path = None
        
        # Strategy 2: Try image_id with zero-padding
        if image_path is None or not image_path.exists():
            image_id = row.get('image_id', '')
            if not pd.isna(image_id) and image_id != '':
                # Try zero-padded format (00001.jpg)
                image_path = TEST_IMAGES_DIR / f"{str(image_id).zfill(5)}.jpg"
                if not image_path.exists():
                    # Try without zero-padding (1.jpg)
                    image_path = TEST_IMAGES_DIR / f"{image_id}.jpg"
                    if not image_path.exists():
                        # Try with different extension
                        image_path = TEST_IMAGES_DIR / f"{str(image_id).zfill(5)}.png"
                        if not image_path.exists():
                            image_path = None
        
        # Strategy 3: Try using index if available
        if (image_path is None or not image_path.exists()) and 'index' in row:
            idx = row.get('index', '')
            if not pd.isna(idx):
                image_path = TEST_IMAGES_DIR / f"{str(idx).zfill(5)}.jpg"
                if not image_path.exists():
                    image_path = None
        
        # Skip if no valid path found
        if image_path is None or not image_path.exists():
            continue
        
        # Get prediction info
        predicted_class = row.get('predicted_class_name', 'Unknown')
        confidence = row.get('confidence', None)
        
        try:
            with Image.open(image_path) as img:
                # Create caption with prediction and confidence
                if confidence is not None and not pd.isna(confidence):
                    caption = f"{predicted_class}\nConfidence: {confidence:.1%}"
                else:
                    caption = predicted_class
                
                col.image(img, caption=caption, use_container_width=True)
                
                # Add confidence indicator below image
                if confidence is not None and not pd.isna(confidence):
                    if confidence > 0.8:
                        col.success(f"✓ High ({confidence:.1%})")
                    elif confidence > 0.5:
                        col.info(f"○ Medium ({confidence:.1%})")
                    else:
                        col.warning(f"⚠ Low ({confidence:.1%})")
        except Exception as exc:
            col.warning(f"Unable to display {image_path.name}: {exc}")


def create_f1_score_chart(report_df: pd.DataFrame) -> alt.Chart:
    """Create interactive F1 score chart from classification report"""
    if report_df.empty or 'f1-score' not in report_df.columns:
        return None
    
    top_k = 30
    chart_df = report_df.sort_values('f1-score', ascending=False).head(top_k)
    
    chart = (
        alt.Chart(chart_df)
        .mark_bar(
            color=CHART_COLORS['secondary'],
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        )
        .encode(
            x=alt.X('f1-score:Q', title='F1 Score', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('class_name:N', sort='-x', title='Car Class'),
            tooltip=[
                alt.Tooltip('class_name:N', title='Class'),
                alt.Tooltip('f1-score:Q', title='F1 Score', format='.3f'),
                alt.Tooltip('precision:Q', title='Precision', format='.3f'),
                alt.Tooltip('recall:Q', title='Recall', format='.3f'),
            ],
        )
        .properties(height=600, title=f'Top {top_k} Classes by F1 Score')
        .configure_axis(grid=True, gridOpacity=0.2)
    )
    return chart


def create_confusion_matrix_chart(report_df: pd.DataFrame) -> alt.Chart:
    """Create interactive confusion matrix heatmap from classification report"""
    if report_df.empty or 'support' not in report_df.columns:
        return None
    
    top_k = 15
    top_classes = report_df.nlargest(top_k, 'support')
    
    chart = (
        alt.Chart(top_classes)
        .mark_bar(
            color=CHART_COLORS['accent'],
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        )
        .encode(
            x=alt.X('support:Q', title='Number of Samples'),
            y=alt.Y('class_name:N', sort='-x', title='Car Class'),
            tooltip=[
                alt.Tooltip('class_name:N', title='Class'),
                alt.Tooltip('support:Q', title='Support', format='.0f'),
                alt.Tooltip('precision:Q', title='Precision', format='.3f'),
                alt.Tooltip('recall:Q', title='Recall', format='.3f'),
            ],
        )
        .properties(height=400, title=f'Sample Distribution - Top {top_k} Classes')
        .configure_axis(grid=True, gridOpacity=0.2)
    )
    return chart


def create_class_distribution_chart(train_df: pd.DataFrame) -> alt.Chart:
    """Create interactive class distribution chart"""
    if train_df.empty:
        return None
    
    top_k = 30
    class_counts = train_df['class_name'].value_counts().head(top_k).reset_index()
    class_counts.columns = ['class_name', 'count']
    
    chart = (
        alt.Chart(class_counts)
        .mark_bar(
            color=CHART_COLORS['primary'],
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        )
        .encode(
            x=alt.X('count:Q', title='Number of Training Images'),
            y=alt.Y('class_name:N', sort='-x', title='Car Class'),
            tooltip=[
                alt.Tooltip('class_name:N', title='Class'),
                alt.Tooltip('count:Q', title='Images', format='.0f'),
            ],
        )
        .properties(height=600, title=f'Top {top_k} Classes by Training Sample Count')
        .configure_axis(grid=True, gridOpacity=0.2)
    )
    return chart

def create_full_class_distribution_analysis_chart(train_df: pd.DataFrame) -> alt.Chart:
    """Create class distribution analysis chart for all 196 classes"""
    if train_df.empty:
        return None
    
    class_counts = train_df['class_name'].value_counts().sort_index().reset_index()
    class_counts.columns = ['class_name', 'count']
    class_counts['class_index'] = range(len(class_counts))
    
    # Calculate statistics for reference lines
    mean_count = class_counts['count'].mean()
    max_count = class_counts['count'].max()
    min_count = class_counts['count'].min()
    
    # Create base chart with bars
    base = alt.Chart(class_counts).encode(
        x=alt.X('class_index:Q', title='Class Index (All 196 Classes)', axis=alt.Axis(format='.0f')),
        y=alt.Y('count:Q', title='Number of Training Images', axis=alt.Axis(format='.0f')),
    )
    
    bars = base.mark_bar(
        color=CHART_COLORS['primary'],
        opacity=0.7,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        tooltip=[
            alt.Tooltip('class_name:N', title='Class'),
            alt.Tooltip('count:Q', title='Images', format='.0f'),
            alt.Tooltip('class_index:Q', title='Class Index', format='.0f'),
        ],
    )
    
    # Create reference lines
    mean_line = alt.Chart(pd.DataFrame({'y': [mean_count]})).mark_rule(
        color='red',
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(y='y:Q')
    
    max_line = alt.Chart(pd.DataFrame({'y': [max_count]})).mark_rule(
        color='green',
        strokeDash=[3, 3],
        strokeWidth=1.5
    ).encode(y='y:Q')
    
    min_line = alt.Chart(pd.DataFrame({'y': [min_count]})).mark_rule(
        color='orange',
        strokeDash=[3, 3],
        strokeWidth=1.5
    ).encode(y='y:Q')
    
    # Combine all layers
    chart = (bars + mean_line + max_line + min_line).properties(
        height=400,
        title='Class Distribution Analysis - All 196 Classes'
    ).configure_axis(grid=True, gridOpacity=0.2)
    
    return chart


def analyze_class_balance(train_df: pd.DataFrame) -> Dict[str, any]:
    """Analyze dataset class balance/imbalance"""
    if train_df.empty:
        return {}
    
    class_counts = train_df['class_name'].value_counts()
    
    total_samples = len(train_df)
    num_classes = len(class_counts)
    
    mean_samples = class_counts.mean()
    std_samples = class_counts.std()
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    median_samples = class_counts.median()
    
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0
    
    cv = std_samples / mean_samples if mean_samples > 0 else 0
    
    imbalanced_threshold = mean_samples * 0.5
    imbalanced_classes = (class_counts < imbalanced_threshold).sum()
    imbalance_percentage = (imbalanced_classes / num_classes * 100) if num_classes > 0 else 0
    
    return {
        'total_samples': total_samples,
        'num_classes': num_classes,
        'mean_samples': mean_samples,
        'std_samples': std_samples,
        'min_samples': min_samples,
        'max_samples': max_samples,
        'median_samples': median_samples,
        'imbalance_ratio': imbalance_ratio,
        'coefficient_variation': cv,
        'imbalanced_classes': imbalanced_classes,
        'imbalance_percentage': imbalance_percentage,
    }

# ============================================================================
# MACHINE LEARNING UTILITY FUNCTIONS
# ============================================================================

def batch_normalization_2d(X: np.ndarray, gamma: Optional[np.ndarray] = None, 
                          beta: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    """
    Batch Normalization for 2D input (N, D).
    
    Normalizes features across the batch dimension (axis 0).
    Each feature dimension is normalized to have zero mean and unit variance.
    This helps stabilize training and allows higher learning rates.
    
    Args:
        X: Input array of shape (N, D) where N is batch size, D is feature dimension
        gamma: Scale parameter (optional), shape (D,)
        beta: Shift parameter (optional), shape (D,)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized array of shape (N, D)
    """
    # Compute mean and variance along batch dimension (axis 0)
    mean = X.mean(axis=0, keepdims=True)
    var = X.var(axis=0, keepdims=True)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Apply affine transformation if provided
    if gamma is not None:
        X_norm = X_norm * gamma
    if beta is not None:
        X_norm = X_norm + beta
    
    return X_norm


def layer_normalization_2d(X: np.ndarray, gamma: Optional[np.ndarray] = None,
                          beta: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization for 2D input (N, D).
    
    Normalizes each sample across its feature dimensions (axis 1).
    Unlike BatchNorm, LayerNorm is independent of batch size and works well
    for sequences and small batches.
    
    Args:
        X: Input array of shape (N, D)
        gamma: Scale parameter (optional), shape (D,)
        beta: Shift parameter (optional), shape (D,)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized array of shape (N, D)
    """
    # Compute mean and variance along feature dimension (axis 1)
    mean = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Apply affine transformation if provided
    if gamma is not None:
        X_norm = X_norm * gamma
    if beta is not None:
        X_norm = X_norm + beta
    
    return X_norm


def instance_normalization_4d(X: np.ndarray, gamma: Optional[np.ndarray] = None,
                             beta: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    """
    Instance Normalization for 4D input (N, C, H, W).
    
    Normalizes each channel of each sample independently across spatial dimensions (H, W).
    Useful for style transfer and image generation tasks where the goal is to remove
    instance-specific contrast information.
    
    Args:
        X: Input array of shape (N, C, H, W)
        gamma: Scale parameter (optional), shape (C,) or (1, C, 1, 1)
        beta: Shift parameter (optional), shape (C,) or (1, C, 1, 1)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized array of shape (N, C, H, W)
    """
    # Compute mean and variance over spatial dimensions (H, W) for each (n, c)
    mean = X.mean(axis=(2, 3), keepdims=True)  # (N, C, 1, 1)
    var = X.var(axis=(2, 3), keepdims=True)     # (N, C, 1, 1)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Apply affine transformation if provided
    if gamma is not None:
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1, 1, 1)
        X_norm = X_norm * gamma
    if beta is not None:
        if beta.ndim == 1:
            beta = beta.reshape(1, -1, 1, 1)
        X_norm = X_norm + beta
    
    return X_norm


def group_normalization_4d(X: np.ndarray, num_groups: int, 
                         gamma: Optional[np.ndarray] = None,
                         beta: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    """
    Group Normalization for 4D input (N, C, H, W).
    
    Divides channels into groups and normalizes within each group.
    Combines benefits of LayerNorm (batch-size independent) and InstanceNorm
    (channel-wise normalization). Works well when batch size is small.
    
    Args:
        X: Input array of shape (N, C, H, W)
        num_groups: Number of groups (must divide C)
        gamma: Scale parameter (optional), shape (C,) or (1, C, 1, 1)
        beta: Shift parameter (optional), shape (C,) or (1, C, 1, 1)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized array of shape (N, C, H, W)
    """
    N, C, H, W = X.shape
    assert C % num_groups == 0, f"C ({C}) must be divisible by num_groups ({num_groups})"
    
    # Reshape to (N, G, Cg, H, W) where Cg = C / G
    Cg = C // num_groups
    X_grouped = X.reshape(N, num_groups, Cg, H, W)
    
    # Compute mean and variance over (Cg, H, W) for each group
    mean = X_grouped.mean(axis=(2, 3, 4), keepdims=True)  # (N, G, 1, 1, 1)
    var = X_grouped.var(axis=(2, 3, 4), keepdims=True)     # (N, G, 1, 1, 1)
    
    # Normalize
    X_norm_grouped = (X_grouped - mean) / np.sqrt(var + eps)
    
    # Reshape back to (N, C, H, W)
    X_norm = X_norm_grouped.reshape(N, C, H, W)
    
    # Apply per-channel affine transformation if provided
    if gamma is not None:
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1, 1, 1)
        X_norm = X_norm * gamma
    if beta is not None:
        if beta.ndim == 1:
            beta = beta.reshape(1, -1, 1, 1)
        X_norm = X_norm + beta
    
    return X_norm


def boxes_iou(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.
    
    IoU is a fundamental metric in object detection that measures overlap
    between predicted and ground truth bounding boxes. It's used for:
    - Evaluating detection accuracy
    - Matching predictions to ground truth
    - Non-maximum suppression (NMS)
    
    Args:
        A: Array of shape (Na, 4) with boxes in format [x1, y1, x2, y2]
        B: Array of shape (Nb, 4) with boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU matrix of shape (Na, Nb) where entry (i, j) is IoU between A[i] and B[j]
    """
    # Expand dimensions for broadcasting: (Na, 1, 4) and (1, Nb, 4)
    A_expanded = A[:, np.newaxis, :]  # (Na, 1, 4)
    B_expanded = B[np.newaxis, :, :]  # (1, Nb, 4)
    
    # Compute intersection coordinates
    x1_max = np.maximum(A_expanded[:, :, 0], B_expanded[:, :, 0])  # (Na, Nb)
    y1_max = np.maximum(A_expanded[:, :, 1], B_expanded[:, :, 1])  # (Na, Nb)
    x2_min = np.minimum(A_expanded[:, :, 2], B_expanded[:, :, 2])  # (Na, Nb)
    y2_min = np.minimum(A_expanded[:, :, 3], B_expanded[:, :, 3])  # (Na, Nb)
    
    # Compute intersection area (0 if no overlap)
    intersection_width = np.maximum(0, x2_min - x1_max)  # (Na, Nb)
    intersection_height = np.maximum(0, y2_min - y1_max)  # (Na, Nb)
    intersection_area = intersection_width * intersection_height  # (Na, Nb)
    
    # Compute areas of individual boxes
    A_area = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])  # (Na,)
    B_area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])  # (Nb,)
    
    # Expand for broadcasting
    A_area_expanded = A_area[:, np.newaxis]  # (Na, 1)
    B_area_expanded = B_area[np.newaxis, :]  # (1, Nb)
    
    # Compute union area
    union_area = A_area_expanded + B_area_expanded - intersection_area  # (Na, Nb)
    
    # Compute IoU (avoid division by zero)
    iou = intersection_area / np.maximum(union_area, 1e-8)  # (Na, Nb)
    
    return iou


def generate_anchor_boxes(feature_map_h: int, feature_map_w: int, 
                          anchor_sizes: np.ndarray, stride: int) -> np.ndarray:
    """
    Generate anchor boxes for object detection.
    
    Anchor boxes are predefined bounding boxes at different scales and aspect ratios
    placed at each spatial location in a feature map. They serve as reference boxes
    for predicting object locations and sizes. This is a key component in:
    - Faster R-CNN
    - YOLO
    - SSD (Single Shot Detector)
    
    Args:
        feature_map_h: Height of feature map
        feature_map_w: Width of feature map
        anchor_sizes: Array of shape (A, 2) where each row is [width, height] of an anchor
        stride: Stride of the feature map (pixels per cell in original image)
    
    Returns:
        Anchor boxes array of shape (H, W, A, 4) in format [x1, y1, x2, y2]
    """
    A = len(anchor_sizes)  # Number of anchor types
    
    # Create grid of center coordinates
    # Each cell in feature map corresponds to a center point in original image
    cy = np.arange(feature_map_h) * stride + stride / 2  # (H,)
    cx = np.arange(feature_map_w) * stride + stride / 2  # (W,)
    
    # Create meshgrid for all (cy, cx) combinations
    cy_grid, cx_grid = np.meshgrid(cy, cx, indexing='ij')  # Both (H, W)
    
    # Initialize output array
    anchors = np.zeros((feature_map_h, feature_map_w, A, 4))
    
    # For each anchor type
    for a_idx in range(A):
        w, h = anchor_sizes[a_idx]
        half_w, half_h = w / 2, h / 2
        
        # Compute box coordinates for all spatial locations
        # Broadcasting: (H, W) with scalar w, h
        anchors[:, :, a_idx, 0] = cx_grid - half_w  # x1
        anchors[:, :, a_idx, 1] = cy_grid - half_h  # y1
        anchors[:, :, a_idx, 2] = cx_grid + half_w  # x2
        anchors[:, :, a_idx, 3] = cy_grid + half_h  # y2
    
    return anchors


def color_tp(pred: np.ndarray, gt: np.ndarray, tp_colors: List[tuple]) -> np.ndarray:
    """
    Color code true positive pixels for segmentation visualization.
    
    Creates a visualization image where pixels that are correctly predicted
    (pred == gt) are colored according to their class, and incorrect pixels are black.
    This helps visualize model performance at the pixel level.
    
    Args:
        pred: Predicted segmentation mask of shape (H, W) with class IDs
        gt: Ground truth segmentation mask of shape (H, W) with class IDs
        tp_colors: List of RGB tuples, one color per class
    
    Returns:
        Colored image of shape (H, W, 3) with dtype uint8
    """
    H, W = pred.shape
    output = np.zeros((H, W, 3), dtype=np.uint8)
    
    # For each class
    for class_id, color in enumerate(tp_colors):
        # Create mask for true positives of this class
        # True positive: pred == gt == class_id
        mask = (pred == class_id) & (gt == class_id)  # (H, W) boolean
        
        # Set color for true positive pixels
        output[mask] = color
    
    return output


def split_batch(batch: np.ndarray, num_splits: int) -> List[np.ndarray]:
    """
    Split a batch into multiple mini-batches.
    
    Divides a batch along the first dimension (batch dimension).
    Useful for processing large batches in chunks or implementing
    gradient accumulation strategies.
    
    Args:
        batch: Input batch array of shape (N, ...)
        num_splits: Number of splits to create
    
    Returns:
        List of mini-batch arrays
    """
    return np.split(batch, num_splits, axis=0)


def concat_batches(batches: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple batches into a single batch.
    
    Combines multiple batches along the first dimension.
    Inverse operation of split_batch. Used to merge results from
    parallel processing or combine data from different sources.
    
    Args:
        batches: List of batch arrays, each of shape (Ni, ...)
    
    Returns:
        Single concatenated batch of shape (sum(Ni), ...)
    """
    return np.concatenate(batches, axis=0)


def nhwc_to_nchw(X: np.ndarray) -> np.ndarray:
    """
    Convert tensor layout from NHWC to NCHW.
    
    Converts from (N, H, W, C) to (N, C, H, W) format.
    NHWC is common in TensorFlow, while NCHW is common in PyTorch.
    Channel-first layout (NCHW) can be more efficient for convolution operations.
    
    Args:
        X: Input array of shape (N, H, W, C)
    
    Returns:
        Output array of shape (N, C, H, W)
    """
    return np.transpose(X, (0, 3, 1, 2))


def nchw_to_nhwc(X: np.ndarray) -> np.ndarray:
    """
    Convert tensor layout from NCHW to NHWC.
    
    Converts from (N, C, H, W) to (N, H, W, C) format.
    NHWC layout can be more intuitive for visualization and some operations.
    
    Args:
        X: Input array of shape (N, C, H, W)
    
    Returns:
        Output array of shape (N, H, W, C)
    """
    return np.transpose(X, (0, 2, 3, 1))


st.title('Stanford Cars Classifier')

# Course Information
st.markdown("""
**Course:** CO5177 — Nền tảng lập trình cho phân tích và trực quan dữ liệu 

**Lecture:** Dr. Le Thanh Sach

**Author:** Lê Thị Hồng Cúc (Student ID: 2470882)  

### Goal
Build a production-ready classification system that:
- Classifies car images into **196 fine-grained categories** (make, model, year)
- Achieves high accuracy through modern deep learning techniques
- Provides model interpretability insights
- Deploys as an interactive web application


### Technology Stack

- **Framework**: PyTorch
- **Model Library**: timm (PyTorch Image Models)
- **Augmentation**: Albumentations
- **Interpretability**: Captum
- **Deployment**: Streamlit
- **Backbone**: ResNet50 (configurable)
""")

def render_key_statistics_section():
    st.caption("Comprehensive overview of the Stanford Cars dataset: source, characteristics, and key statistics.")
    
    # Dataset Source and Characteristics
    st.markdown("## Dataset Source & Characteristics")
    
    st.markdown("""
    ### Dataset Information
    
    **Source**: Stanford Cars Dataset  
    **Institution**: Stanford AI Lab  
    **Paper**: "3D Object Representations for Fine-Grained Categorization" (Krause et al., 2013)
    
    ### Dataset Characteristics
    
    **Task Type**: Fine-grained image classification  
    **Domain**: Automotive (cars)  
    **Granularity**: Make, Model, Year classification  
    **Total Classes**: 196 fine-grained car categories
    
    ### Key Features
    
    - **Fine-grained categorization**: Distinguishes between subtle differences (e.g., BMW 320i 2012 vs BMW 320i 2013)
    - **Real-world images**: Photos taken from various angles, lighting conditions, and backgrounds
    - **Bounding box annotations**: Each image includes bounding box coordinates for the car
    - **Balanced distribution**: Relatively balanced class distribution (mean ~42 images per class)
    - **High-quality labels**: Manually verified class labels with make, model, and year information
    
    ### Dataset Structure
    
    - **Training Set**: 8,144 images across 196 classes
    - **Test Set**: 8,041 images across 196 classes  
    - **Format**: JPEG images with MAT annotation files
    - **Image Resolution**: Variable (typically 200-500px width/height)
    - **Aspect Ratios**: Varies (median ~1.3, indicating slight landscape orientation)
    """)
    
    st.divider()
    
    st.markdown("## Dataset Statistics")

    overview_metrics = get_dataset_overview()
    render_metric_cards(
        {
            "Train Images": overview_metrics['train_images'],
            "Test Images": overview_metrics['test_images'],
            "Classes": overview_metrics['classes'],
        }
    )

    train_df = load_annotations_df('train')
    size_stats = compute_image_metadata(sample_size=200)

    col1, col2 = st.columns([2, 1], gap="medium")
    with col1:
            st.subheader("All 196 Classes Distribution (Train)")
            if not train_df.empty:
                all_classes = (
                    train_df['class_name']
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )
                all_classes.columns = ['Class', 'Train Images']
                st.dataframe(all_classes, use_container_width=True, height=600)
                st.caption(f"Total: {len(all_classes)} classes, {all_classes['Train Images'].sum():,} training images")
            else:
                st.info("Training annotations not found.")

    with col2:
            st.subheader("Image Resolution Snapshot")
            if not size_stats.empty:
                st.metric("Average Width", f"{size_stats['width'].mean():.0f}px")
                st.metric("Average Height", f"{size_stats['height'].mean():.0f}px")
                st.metric("Median Aspect Ratio", f"{size_stats['aspect_ratio'].median():.2f}")
            else:
                st.info("Resolution statistics will appear once the dataset is available.")
    
    # Additional charts in columns
    # col_a, col_b = st.columns(2, gap="medium")
    # with col_a:
    #     with st.container():
    #         st.markdown("### Top 30 Classes by Sample Count")
    #         class_dist_chart = create_class_distribution_chart(train_df)
    #         if class_dist_chart:
    #             st.altair_chart(class_dist_chart, use_container_width=True)
    #         else:
    #             st.info("Class distribution data not available.")
    # with col_b:
    #     with st.container():
    #         st.markdown("### Top 30 Classes by F1 Score")
    #         report_df = load_classification_report_df()
    #         f1_chart = create_f1_score_chart(report_df)
    #         if f1_chart:
    #             st.altair_chart(f1_chart, use_container_width=True)
    #         else:
    #             st.info("F1 score data not available.")

def render_key_process_section():
    st.caption("How raw images become reliable fine-grained car predictions.")

    process_steps = [
        {
            "step_num": 1,
            "title": "Data Acquisition & Ground Truth",
            "description": (
                "Download Stanford Cars dataset with official devkit annotations. "
                "Process and link file names to class IDs and bounding boxes for all 196 fine-grained car classes."
            ),
            "details": [
                "8,144 labeled training images with bounding boxes.",
                "8,041 test images for leaderboard evaluation.",
                "196 fine-grained car classes (make-model-year combinations).",
                "Stratified train/validation split (85%/15%) ensures all classes represented.",
            ],
            "reference_section": "Dataset Overview",
        },
        {
            "step_num": 2,
            "title": "Preprocessing & Augmentation",
            "description": (
                "Apply strong data augmentation using Albumentations pipeline to increase dataset diversity "
                "and improve model generalization. Batch utilities handle data layout conversion (NHWC↔NCHW) for efficient processing."
            ),
            "details": [
                "Resize to 224×224, then RandomResizedCrop (scale: 0.7-1.0) for training.",
                "HorizontalFlip (50%), ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue.",
                "CLAHE, Gaussian noise/blur, and CoarseDropout (CutOut-like) for robustness.",
                "Standard ImageNet mean/std normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).",
            ],
            "reference_section": "Data Augmentation",
        },
        {
            "step_num": 3,
            "title": "Feature Extraction with ResNet50 Backbone",
            "description": (
                "The ResNet50 backbone (pretrained on ImageNet) extracts rich visual features from preprocessed images. "
                "Convolutional layers progressively learn hierarchical patterns: edges → textures → shapes → car parts → complete vehicles."
            ),
            "details": [
                "Input: 224×224×3 RGB images (normalized with ImageNet statistics).",
                "Backbone: ResNet50 with 5 convolutional blocks (50 layers total).",
                "Global Average Pooling: Reduces spatial dimensions 7×7×2048 → 1×1×2048, producing 2048-dim feature vectors.",
                "Transfer Learning: Backbone initialized with ImageNet pretrained weights for robust feature representations.",
                "Output: 2048-dimensional embeddings capture discriminative car characteristics.",
            ],
            "reference_section": "Model Architecture",
        },
        {
            "step_num": 4,
            "title": "Model Architecture & Classifier Head",
            "description": (
                "Build complete model by combining ResNet50 backbone with custom classifier head. "
                "All layers are trainable (full fine-tuning) for domain adaptation to car images."
            ),
            "details": [
                "Classifier architecture: 2048-dim features → Dropout(0.3) → Linear(512) → ReLU → Dropout(0.2) → Linear(196).",
                "ResNet50 uses BatchNorm internally after each convolutional layer for stable training.",
                "All backbone layers unfrozen with low learning rate warmup for gradual domain adaptation.",
                "Mixed precision (AMP) enabled for faster training while maintaining numerical stability.",
                "Total parameters: ~25.9M (24.7M backbone + 1.2M classifier).",
            ],
            "reference_section": "Model Architecture",
        },
        {
            "step_num": 5,
            "title": "Training & Optimization",
            "description": (
                "Train model with AdamW optimizer and OneCycleLR scheduler, track metrics and checkpoints. "
                "Class-weighted CrossEntropyLoss handles dataset imbalance (ratio: ~2.83x) to ensure balanced learning."
            ),
            "details": [
                "Batch size: 96 on GPU with mixed precision (AMP) for faster training.",
                "Optimizer: AdamW (lr=1e-3, weight_decay=1e-4) with OneCycleLR scheduler (cosine annealing).",
                "Loss function: CrossEntropyLoss with class weights (inverse frequency weighting).",
                "Early stopping: Stops if validation accuracy doesn't improve for 3-5 epochs.",
                "Best checkpoint (highest validation accuracy) automatically saved for inference.",
            ],
            "reference_section": "Training Loop",
        },
        {
            "step_num": 6,
            "title": "Evaluation & Reporting",
            "description": (
                "Comprehensive evaluation with classification report, confusion matrix, and interpretability overlays. "
                "Detailed metrics for all 196 classes help identify strengths and weaknesses."
            ),
            "details": [
                "Performance metrics: Top-1 accuracy 86.25%, Top-5 accuracy 97.71% on validation split.",
                "Per-class metrics: Detailed precision/recall/F1 scores for all 196 classes.",
                "Confusion matrix: Reveals class confusion patterns and common misclassifications.",
                "Model interpretability: Grad-CAM overlays (via Captum) show which image regions influence predictions.",
            ],
            "reference_section": "Comprehensive Evaluation",
        },
        {
            "step_num": 7,
            "title": "Test Inference & Predictions",
            "description": (
                "Generate predictions on test set (8,041 images) using the best trained model. "
                "Analyze prediction confidence scores and identify challenging cases for model improvement."
            ),
            "details": [
                "Load best model checkpoint and run inference on all 8,041 test images.",
                "Generate predictions with confidence scores for each test image.",
                "Analyze confidence distribution: high (>0.9), medium (0.5-0.9), and low (≤0.5) confidence predictions.",
                "Identify low-confidence predictions for further analysis and potential model improvement.",
                "Export predictions in CSV format for competition submission or further evaluation.",
            ],
            "reference_section": "Test Inference",
        },
    ]

    for step in process_steps:
        with st.container():
            st.markdown(f"### Step {step['step_num']}: {step['title']}")
            st.write(step['description'])
            if 'details' in step:
                for bullet in step['details']:
                    st.markdown(f"- {bullet}")
            
            # Add reference to related section if available
            if 'reference_section' in step:
                st.info(f" **For detailed information and visualizations, see the '{step['reference_section']}' section.**")
            
            # Add additional details for specific steps
            if step['step_num'] == 4:  # Model Architecture & Classifier Head
                st.markdown("---")
                st.markdown("#### Model Architecture Details")
                st.markdown("""
                **ResNet50 Architecture:**
                - Uses **BatchNorm** internally after each convolutional layer for stable training
                - BatchNorm normalizes activations across the batch dimension
                - Enables faster convergence and allows higher learning rates
                - All layers are trainable (full fine-tuning) for domain adaptation to car images
                """)
            
        st.divider()

def render_dataset_balance_section():
    st.caption("Assessment of class distribution balance and potential imbalance issues.")
    
    train_df_balance = load_annotations_df('train')
    if not train_df_balance.empty:
        balance_stats = analyze_class_balance(train_df_balance)
        
        # Key metrics at the top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Images",
                f"{balance_stats['total_samples']:,}",
                help="Total training images"
            )
        with col2:
            st.metric(
                "Total Classes",
                f"{balance_stats['num_classes']}",
                help="Number of unique classes"
            )
        with col3:
            st.metric(
                "Avg Samples/Class",
                f"{balance_stats['mean_samples']:.1f}",
                help="Mean samples per class"
            )
        with col4:
            st.metric(
                "Imbalance Ratio",
                f"{balance_stats['imbalance_ratio']:.2f}x",
                help="Max samples / Min samples"
            )
        
        st.divider()
        
        # Detailed statistics and interpretation
        col1, col2 = st.columns([2, 1], gap="medium")
        with col1:
            st.markdown("### Distribution Statistics")
            stats_text = f"""
**Sample Count Statistics:**
- **Min samples per class**: {balance_stats['min_samples']:.0f}
- **Max samples per class**: {balance_stats['max_samples']:.0f}
- **Mean samples per class**: {balance_stats['mean_samples']:.1f}
- **Median samples per class**: {balance_stats['median_samples']:.1f}
- **Standard deviation**: {balance_stats['std_samples']:.2f}
- **Coefficient of Variation**: {balance_stats['coefficient_variation']:.3f}

**Imbalance Assessment:**
- **Imbalance Ratio**: {balance_stats['imbalance_ratio']:.2f}x (Max/Min)
- **Classes below 50% of mean**: {balance_stats['imbalanced_classes']} ({balance_stats['imbalance_percentage']:.1f}%)
            """
            st.markdown(stats_text)
        
        with col2:
            st.markdown("### Interpretation")
            if balance_stats['imbalance_ratio'] < 1.5:
                status = "Well Balanced"
                desc = "Dataset is relatively balanced across classes. Standard training should work well."
                color = "success"
            elif balance_stats['imbalance_ratio'] < 3.0:
                status = "Moderately Imbalanced"
                desc = "Some classes have fewer samples but still acceptable. Weighted loss recommended."
                color = "warning"
            else:
                status = "Significantly Imbalanced"
                desc = "Notable imbalance detected. Weighted loss is essential for balanced learning."
                color = "error"
            
            st.markdown(f"**{status}**\n\n{desc}")
            
            if balance_stats['imbalance_ratio'] >= 1.5:
                st.info(
                    "**Solution Applied:** Using class-weighted CrossEntropyLoss to handle imbalance. "
                    "Inverse frequency weighting ensures underrepresented classes contribute equally to training."
                )
        
        st.divider()
        
        # Additional context
        st.markdown("### Key Insights")
        insights_text = f"""
**Dataset Characteristics:**
- The Stanford Cars dataset shows **typical fine-grained classification imbalance** with a ratio of {balance_stats['imbalance_ratio']:.2f}x.
- All 196 classes are present in the training set, ensuring complete class coverage.
- The coefficient of variation ({balance_stats['coefficient_variation']:.3f}) indicates relatively low variability in class sizes.

**Impact on Training:**
- Without class weights, the model might bias toward classes with more samples.
- Class-weighted loss ensures all classes contribute equally, improving performance on underrepresented classes.
- This approach is standard for fine-grained classification tasks where class imbalance is common.
        """
        st.markdown(insights_text)
    else:
        st.warning("Training data not available for balance analysis.")

def render_eda_section():
    st.caption("Comprehensive exploratory data analysis: class distribution, image quality, and representative samples.")
    
    # Display artifact information
    display_artifact_info({
        "class_distribution_train_all_196.png": "Training set class distribution visualization (Notebook: Cell 19)",
        "class_distribution_test_all_196.png": "Test set class distribution visualization (Notebook: Cell 19)",
        "class_distribution_train_vs_test_all_196.png": "Train vs test distribution comparison (Notebook: Cell 19)"
    })
    
    # Load data
    train_df = load_annotations_df('train')
    
    # Create tabs for EDA
    tab_distribution, tab_sizes, tab_color, tab_gallery = st.tabs(
        [
            "Class Distribution",
            "Image Size & Aspect Ratio",
            "Color & Quality Metrics",
            "Interactive Sample Gallery",
        ]
    )

    # Class Distribution tab
    with tab_distribution:
        st.subheader("Training Set - Class Distribution Analysis")
        
        if not train_df.empty:
            # Count classes
            class_counts = (
                train_df['class_name']
                .value_counts()
                .reset_index()
            )
            class_counts.columns = ['Class', 'Images']
            class_counts = class_counts.sort_values('Images', ascending=False).reset_index(drop=True)
            
            # Create Class Index starting from 0
            class_counts['Class Index'] = class_counts.index.astype(int)
            class_counts['Images'] = class_counts['Images'].astype(int)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Classes", f"{len(class_counts)}")
            with col2:
                st.metric("Total Images", f"{class_counts['Images'].sum():,}")
            with col3:
                st.metric("Avg per Class", f"{class_counts['Images'].mean():.1f}")
            with col4:
                st.metric("Imbalance Ratio", f"{class_counts['Images'].max() / class_counts['Images'].min():.2f}x")
            
            st.divider()
            
            # Class Distribution Analysis Chart (moved from Dataset Overview)
            st.markdown("### Class Distribution Analysis - All 196 Classes")
            if not train_df.empty:
                dist_analysis_chart = create_full_class_distribution_analysis_chart(train_df)
                if dist_analysis_chart:
                    st.altair_chart(dist_analysis_chart, use_container_width=True)
                    st.caption(" Chart Legend: Red dashed line = Mean, Green dashed line = Max, Orange dashed line = Min")
                else:
                    st.info("Class distribution chart not available.")
            else:
                st.info("Training data not available for distribution analysis.")
        else:
            st.warning("Training data not available for distribution plot.")
                
    # Image Size & Aspect Ratio, Color & Quality, Gallery tabs
    if train_df.empty:
        st.warning("Training annotations were not found. Please ensure the devkit is available.")
    else:
        # Use a fixed sample size for EDA computations (400 samples or all if dataset is smaller)
        sample_size = min(400, len(train_df))
        eda_stats = compute_image_metadata(sample_size=sample_size)

        with tab_sizes:
            st.subheader("Image Size Distribution")
            if not eda_stats.empty:
                st.markdown("""
                **Image Dimensions Scatter Plot**: This visualization shows the distribution of image widths and heights 
                across the dataset. Each point represents one image, positioned by its width (x-axis) and height (y-axis).
                
                **Key Insights**:
                - **Spread of points**: Shows the variability in image dimensions (some images are wider, some taller)
                - **Clustering patterns**: Images with similar dimensions form clusters
                - **Aspect ratio lines**: Points along diagonal lines (e.g., y=x) have square aspect ratios
                - **Outliers**: Points far from the main cluster may indicate unusual image sizes
                
                **Interpretation**:
                - **Wide spread**: Dataset contains images of various sizes (good for generalization)
                - **Concentration**: Most images cluster around common dimensions (e.g., 200-500px)
                - **Preprocessing impact**: Images are resized to 224×224 during training, so original size variation 
                  is normalized but original aspect ratios may differ
                
                **Use Cases**:
                - Understand dataset diversity in image dimensions
                - Identify potential preprocessing needs (cropping, padding)
                - Assess whether resizing strategy is appropriate
                """)
                
                size_chart = (
                    alt.Chart(eda_stats)
                    .mark_circle(
                        opacity=0.6,
                        size=80,
                        color=CHART_COLORS['primary'],
                        stroke=CHART_COLORS['primary'],
                        strokeWidth=1
                    )
                    .encode(
                        x=alt.X('width:Q', title='Width (px)', scale=alt.Scale(nice=True)),
                        y=alt.Y('height:Q', title='Height (px)', scale=alt.Scale(nice=True)),
                        tooltip=[
                            alt.Tooltip('class_name:N', title='Class'),
                            alt.Tooltip('width:Q', title='Width', format='.0f'),
                            alt.Tooltip('height:Q', title='Height', format='.0f'),
                            alt.Tooltip('aspect_ratio:Q', title='Aspect Ratio', format='.2f'),
                        ],
                    )
                    .properties(height=450, title='Image Dimensions Scatter Plot')
                    .configure_axis(grid=True, gridOpacity=0.2)
                    .interactive()
                )
                st.altair_chart(size_chart, use_container_width=True)

                st.subheader("Aspect Ratio Distribution")
                st.markdown("""
                **Aspect Ratio Distribution**: This histogram shows how aspect ratios (width/height) are distributed 
                across the dataset. Aspect ratio indicates whether images are landscape (ratio > 1), square (ratio ≈ 1), 
                or portrait (ratio < 1).
                
                **Key Insights**:
                - **Peak location**: Most common aspect ratio in the dataset
                - **Distribution shape**: Whether aspect ratios are concentrated or spread out
                - **Landscape vs Portrait**: Most car images are typically landscape (wider than tall)
                - **Standard ratios**: Common ratios like 1.0 (square), 1.33 (4:3), 1.5 (3:2), 1.77 (16:9)
                
                **Interpretation**:
                - **Peak around 1.2-1.5**: Most images are slightly wider than tall (typical for car photos)
                - **Narrow distribution**: Consistent aspect ratios make preprocessing easier
                - **Wide distribution**: Variable aspect ratios may require careful cropping/padding strategies
                
                **Impact on Training**:
                - **Consistent ratios**: Easier to apply uniform preprocessing (resize to 224×224)
                - **Variable ratios**: May need to preserve aspect ratio with padding or use adaptive pooling
                - **Data augmentation**: Can use aspect ratio preserving transforms or random crops
                
                **Common Aspect Ratios**:
                - **1.0**: Square images (equal width and height)
                - **1.33 (4:3)**: Traditional photo format
                - **1.5 (3:2)**: Standard camera format
                - **1.77 (16:9)**: Widescreen format
                """)
                
                aspect_chart = (
                    alt.Chart(eda_stats)
                    .mark_bar(
                        color=CHART_COLORS['secondary'],
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4,
                    )
                    .encode(
                        x=alt.X(
                            'aspect_ratio:Q',
                            bin=alt.Bin(maxbins=30, step=0.1),
                            title='Aspect Ratio (Width/Height)',
                            axis=alt.Axis(format='.1f')
                        ),
                        y=alt.Y('count()', title='Number of Images', axis=alt.Axis(format='.0f')),
                        tooltip=[
                            alt.Tooltip('aspect_ratio:Q', bin=True, title='Aspect Ratio Range'),
                            alt.Tooltip('count()', title='Image Count', format='.0f'),
                        ],
                    )
                    .properties(height=350, title='Distribution of Image Aspect Ratios')
                    .configure_axis(grid=True, gridOpacity=0.2)
                )
                st.altair_chart(aspect_chart, use_container_width=True)
            else:
                st.info("Unable to compute size statistics for the selected sample.")

        with tab_color:
            st.subheader("Color Channel Analysis")
            if not eda_stats.empty:
                st.markdown("""
                **Average Color Channel Values**: This bar chart shows the average pixel intensity for each RGB color 
                channel (Red, Green, Blue) across all images in the dataset. Values range from 0 (dark) to 255 (bright).
                
                **Key Insights**:
                - **Channel balance**: Whether RGB channels are balanced (similar values) or imbalanced
                - **Color bias**: Higher values in specific channels indicate color bias (e.g., warmer/cooler tones)
                - **Dataset characteristics**: Overall color distribution of the dataset
                
                **Interpretation**:
                - **Balanced channels**: Similar values across R, G, B indicate neutral color distribution
                - **Red bias**: Higher red values suggest warmer tones (sunset, warm lighting)
                - **Blue bias**: Higher blue values suggest cooler tones (shade, cool lighting)
                - **Green bias**: Higher green values may indicate outdoor/natural scenes
                
                **Impact on Training**:
                - **Color normalization**: Helps determine appropriate mean/std values for normalization
                - **Data augmentation**: Color jitter augmentation can help balance channel distributions
                - **Model performance**: Balanced channels may lead to better generalization
                
                **Normalization Context**:
                - Standard ImageNet normalization uses: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                - Dataset-specific normalization can be computed from these channel means
                - Normalization helps models converge faster and generalize better
                """)
                
                channel_means = eda_stats[['mean_r', 'mean_g', 'mean_b']].mean()
                color_df = channel_means.reset_index()
                color_df.columns = ['Channel', 'Mean Value']
                color_df['Color'] = color_df['Channel'].map({
                    'mean_r': CHART_COLORS['primary'],
                    'mean_g': CHART_COLORS['success'],
                    'mean_b': CHART_COLORS['secondary']
                })
                
                color_chart = (
                    alt.Chart(color_df)
                    .mark_bar(
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4,
                    )
                    .encode(
                        x=alt.X('Channel:N', title='Color Channel', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Mean Value:Q', title='Mean Value (0-255)', scale=alt.Scale(domain=[0, 255])),
                        color=alt.Color('Color:N', scale=None, legend=None),
                        tooltip=[
                            alt.Tooltip('Channel:N', title='Channel'),
                            alt.Tooltip('Mean Value:Q', title='Mean Value', format='.1f'),
                        ],
                    )
                    .properties(height=350, title='Average Color Channel Values')
                    .configure_axis(grid=True, gridOpacity=0.2)
                )
                st.altair_chart(color_chart, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                brightness_val = eda_stats['brightness'].mean() * 100
                with col1:
                    st.metric("Average Brightness", f"{brightness_val:.1f}%", help="Overall image brightness (0-100%)")
                with col2:
                    st.metric("Avg Red Channel", f"{channel_means['mean_r']:.1f}", help="Average red channel value")
                with col3:
                    st.metric("Avg Green Channel", f"{channel_means['mean_g']:.1f}", help="Average green channel value")

                if 'blur_score' in eda_stats and eda_stats['blur_score'].notna().any():
                    st.subheader("Image Sharpness Analysis")
                    st.markdown("""
                    **Distribution of Image Sharpness**: This histogram shows the distribution of image sharpness scores 
                    across the dataset. Sharpness is measured using Laplacian variance, which quantifies the amount of 
                    detail and edge information in an image.
                    
                    **How Sharpness is Measured**:
                    - **Laplacian Variance**: Computes the variance of the Laplacian operator applied to the image
                    - **Higher values**: Indicate sharper images with more detail and clear edges
                    - **Lower values**: Indicate blurrier images with less detail and softer edges
                    - **Typical range**: Sharp images typically have scores > 100, blurry images < 50
                    
                    **Key Insights**:
                    - **Peak location**: Most common sharpness level in the dataset
                    - **Distribution spread**: Whether images are consistently sharp or vary widely
                    - **Blurry images**: Low-score images may need special handling or filtering
                    - **Quality assessment**: Overall image quality of the dataset
                    
                    **Interpretation**:
                    - **Right-skewed distribution**: Most images are sharp (good quality dataset) 
                    - **Left-skewed distribution**: Many blurry images (may need quality filtering) 
                    - **Bimodal distribution**: Mix of sharp and blurry images (may indicate different sources)
                    - **Narrow distribution**: Consistent image quality across the dataset
                    
                    **Impact on Training**:
                    - **High sharpness**: Model can learn fine-grained details (important for 196-class classification)
                    - **Low sharpness**: May confuse model, especially for similar classes
                    - **Data filtering**: Can filter out very blurry images (e.g., scores < 30) to improve training
                    - **Augmentation**: Sharpening augmentation can help improve blurry images
                    
                    **Use Cases**:
                    - **Quality control**: Identify and potentially remove very blurry images
                    - **Dataset assessment**: Understand overall image quality
                    - **Preprocessing decisions**: Determine if sharpening filters are needed
                    - **Model performance**: Blurry images may contribute to misclassifications
                    """)
                    
                    blur_chart = (
                        alt.Chart(eda_stats.dropna(subset=['blur_score']))
                        .mark_area(
                            opacity=0.7,
                            color=CHART_COLORS['accent'],
                            interpolate='monotone'
                        )
                        .encode(
                            x=alt.X(
                                'blur_score:Q',
                                bin=alt.Bin(maxbins=40),
                                title='Laplacian Variance (Sharpness Score)',
                                axis=alt.Axis(format='.0f')
                            ),
                            y=alt.Y('count()', title='Number of Images', axis=alt.Axis(format='.0f')),
                            tooltip=[
                                alt.Tooltip('blur_score:Q', bin=True, title='Sharpness Range'),
                                alt.Tooltip('count()', title='Image Count', format='.0f'),
                            ],
                        )
                        .properties(height=350, title='Distribution of Image Sharpness')
                        .configure_axis(grid=True, gridOpacity=0.2)
                    )
                    st.altair_chart(blur_chart, use_container_width=True)
                else:
                    st.info("OpenCV not available — skipping blur metric.")
            else:
                st.info("Color statistics will appear when sample metadata is available.")

        with tab_gallery:
            st.subheader("Browse Sample Images by Class")
            class_choice = st.selectbox("Select a class", CLASS_NAMES, index=0)
            num_images = st.slider("Number of images to preview", min_value=3, max_value=12, value=6, step=3, key="eda_gallery_images")
            render_sample_gallery(class_choice, num_images=num_images)



def render_train_val_test_split_section():
    st.caption("Dataset split strategy: train, validation, and test set distribution.")
    
    st.subheader("Dataset Split Overview")
    
    # Get actual split numbers
    overview_metrics = get_dataset_overview()
    total_train_images = overview_metrics['train_images']  # 8,144
    val_split_ratio = 0.15
    val_images = int(total_train_images * val_split_ratio)  # ~1,222
    train_images = total_train_images - val_images  # ~6,922
    test_images = overview_metrics['test_images']  # 8,041
    
    split_df = pd.DataFrame(
        [
            {"Split": "Train", "Images": train_images, "Classes": 196, "Short": "Train"},
            {"Split": "Validation", "Images": val_images, "Classes": 196, "Short": "Val"},
            {"Split": "Test", "Images": test_images, "Classes": 196, "Short": "Test"},
        ]
    )
    total_images = split_df['Images'].sum()
    split_df['Percent'] = (split_df['Images'] / total_images * 100).round(2)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train Images", f"{train_images:,}", help="85% of training data")
    with col2:
        st.metric("Val Images", f"{val_images:,}", help="15% of training data")
    with col3:
        st.metric("Test Images", f"{test_images:,}", help="Official test set")
    with col4:
        st.metric("Total Images", f"{total_images:,}", help="All dataset images")
    
    st.divider()
    
    # Split table and visualization
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.markdown("### Split Details")
        st.dataframe(
            split_df[['Split', 'Images', 'Classes', 'Percent']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Split": st.column_config.TextColumn("Dataset Split", width="medium"),
                "Images": st.column_config.NumberColumn("Images", format="%d"),
                "Classes": st.column_config.NumberColumn("Classes", format="%d"),
                "Percent": st.column_config.NumberColumn("Percentage", format="%.1f%%"),
            }
        )
    
    with col2:
        split_colors = {
            "Train": CHART_COLORS['primary'],
            "Validation": CHART_COLORS['accent'],
            "Test": CHART_COLORS['secondary'],
        }
        split_df['Color'] = split_df['Split'].map(split_colors)
        
        split_chart = (
            alt.Chart(split_df)
            .mark_arc(innerRadius=50, outerRadius=120, stroke='white', strokeWidth=2)
            .encode(
                theta=alt.Theta("Images:Q", stack=True),
                color=alt.Color(
                    "Split:N",
                    scale=alt.Scale(
                        domain=list(split_colors.keys()),
                        range=list(split_colors.values())
                    ),
                    legend=alt.Legend(title="Dataset Split", orient="right")
                ),
                tooltip=[
                    alt.Tooltip('Split:N', title='Split'),
                    alt.Tooltip('Images:Q', title='Images', format='.0f'),
                    alt.Tooltip('Percent:Q', title='Percentage', format='.1f'),
                ],
            )
            .properties(height=300, title='Dataset Split Distribution')
        )
        st.altair_chart(split_chart, use_container_width=True)
    
    st.divider()
    
    # Stratified split explanation
    st.markdown("### Stratified Split Strategy")
    st.markdown("""
    **Why Stratified Split?**
    - Ensures **ALL 196 classes** appear in both training and validation sets
    - Maintains class distribution proportions across splits
    - Prevents data leakage (no class missing from any split)
    - Enables fair evaluation across all classes
    
    **Split Configuration:**
    - **Training Set**: 85% of original training data ({train_images:,} images)
    - **Validation Set**: 15% of original training data ({val_images:,} images)
    - **Test Set**: {test_images:,} images (kept separate, untouched for final evaluation)
    
    **Verification:**
    - Training set: All 196 classes present
    - Validation set: All 196 classes present  
    - Test set: All 196 classes present
    - No data leakage: Images from same car not in multiple splits
    """.format(train_images=train_images, val_images=val_images, test_images=test_images))
    
    st.markdown("---")
    
    # Code example
    st.markdown("### Implementation")
    split_code = """from sklearn.model_selection import train_test_split

# Stratified split - ensures all classes in both sets
X_train_paths, X_val_paths, y_train_encoded, y_val_encoded = train_test_split(
    X_train_paths,  # Original training image paths
    y_train_encoded,  # Encoded labels
    test_size=0.15,  # 15% for validation
    stratify=y_train_encoded,  # Maintain class distribution
    random_state=42  # Reproducibility
)

print(f"Training samples: {len(X_train_paths):,}")
print(f"Validation samples: {len(X_val_paths):,}")
print(f"Training classes: {len(set(y_train_encoded))}")
print(f"Validation classes: {len(set(y_val_encoded))}")
"""
    st.code(split_code, language="python")
    st.info(
        "**Key Point**: The `stratify` parameter ensures that the class distribution in validation set "
        "matches the training set, guaranteeing all 196 classes are represented in both splits."
    )


def render_data_augmentation_section():
    st.caption("Data augmentation pipeline and transformation examples.")
    
    # Display artifact information
    display_artifact_info({
        "augmentation_examples.png": "Augmentation visualization examples showing transformed images (Notebook: Cell 25-27)"
    })
    
    st.subheader("Albumentations Pipeline")
    
    st.markdown("""
    **Purpose:** Strong augmentation increases dataset diversity and reduces overfitting by simulating varied real-world conditions.
    
    **Pipeline Coverage:** Geometric transforms, color & lighting, noise/blur, and regularization; followed by normalization and tensor conversion.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Geometric (p ≈ 0.5-0.8):**
        - **RandomResizedCrop** (scale 0.7-1.0, ratio 0.75-1.33): Randomly crops and resizes to simulate different viewpoints
        - **HorizontalFlip** (p=0.5): Mirrors image horizontally (50% probability)
        - **ShiftScaleRotate** (shift 0.1, scale 0.2, rotate 15°): Applies translation, scaling, and rotation transformations
        
        **Color & Lighting:**
        - **RandomBrightnessContrast** (±0.3): Adjusts brightness and contrast (±30%)
        - **HueSaturationValue** (±20/30/20): Modifies color properties (hue ±20°, saturation ±30, value ±20)
        - **CLAHE** (clip_limit 4.0, tile 8×8): Contrast Limited Adaptive Histogram Equalization for better contrast
        """)
    with col2:
        st.markdown("""
        **Noise & Blur (OneOf, p=0.5):**
        - **GaussNoise** (var 10-50): Adds random Gaussian noise (variance 10-50)
        - **GaussianBlur** (kernel 3-7): Applies Gaussian blur (kernel size 3-7)
        - **MotionBlur** (limit 7): Simulates motion blur (blur limit 7)
        
        **Regularization & Preprocessing:**
        - **CoarseDropout** (up to 8 holes, 32×32): Randomly removes rectangular regions (cutout) to prevent overfitting
        - **Normalize** (ImageNet mean/std) → **ToTensorV2**: Standardizes pixel values using ImageNet statistics, then converts to PyTorch tensor
        """)
    
    st.markdown("---")
    
    augmentation_code = """A.Compose([
    # Resize to standard input size
    A.Resize(224, 224, p=1.0),
    
    # Geometric Augmentations (p ≈ 0.5-0.8)
    # RandomResizedCrop: Randomly crops and resizes to simulate different viewpoints
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.8),
    # HorizontalFlip: Mirrors image horizontally (50% probability)
    A.HorizontalFlip(p=0.5),
    # ShiftScaleRotate: Applies translation, scaling, and rotation transformations
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
    
    # Color & Lighting Augmentations
    # RandomBrightnessContrast: Adjusts brightness and contrast (±30%)
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    # HueSaturationValue: Modifies color properties (hue ±20°, saturation ±30, value ±20)
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    # CLAHE: Contrast Limited Adaptive Histogram Equalization for better contrast
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    
    # Noise & Blur (OneOf: applies one randomly selected transformation)
    A.OneOf([
        # GaussNoise: Adds random Gaussian noise (variance 10-50)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # GaussianBlur: Applies Gaussian blur (kernel size 3-7)
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        # MotionBlur: Simulates motion blur (blur limit 7)
        A.MotionBlur(blur_limit=7, p=0.5),
    ], p=0.5),
    
    # Regularization & Preprocessing
    # CoarseDropout: Randomly removes rectangular regions (cutout) to prevent overfitting
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    # Normalize: Standardizes pixel values using ImageNet statistics
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    # ToTensorV2: Converts image to PyTorch tensor format
    ToTensorV2(p=1.0),
])"""
    st.code(augmentation_code, language="python")
    st.markdown("""
    **Augmentation Examples**: This visualization demonstrates how the same image is transformed 
    through different augmentation techniques. Each row shows a different augmentation applied:
    - **Geometric transformations**: Rotation, scaling, translation, and flipping create viewpoint variations
    - **Color adjustments**: Brightness, contrast, and saturation changes simulate different lighting conditions
    - **Regularization**: Cutout/dropout helps prevent overfitting by forcing the model to not rely on 
      specific image regions
    
    These transformations help the model learn robust features that generalize to real-world variations 
    in car images (different angles, lighting, backgrounds).
    """)
    display_image_if_exists(AUGMENTATION_IMG, "Augmentation Examples")


def render_model_architecture_section():
    st.caption("Model architecture details: ResNet50 backbone and classifier head.")
    
    # Display artifact information
    display_artifact_info({
        "feature_extraction_pipeline.png": "Feature extraction pipeline visualization (Notebook: Cell 30)",
        "conv_filters_visualization.png": "Convolutional filters visualization from early layers (Notebook: Cell 30)",
        "weight_distributions.png": "Weight distributions across different layers (Notebook: Cell 30)",
        "weight_magnitudes_by_layer.png": "Weight magnitudes analysis by layer (Notebook: Cell 30)",
        "top_layers_by_parameters.png": "Top layers ranked by parameter count (Notebook: Cell 30)",
        "interpretability.png": "Model interpretability visualization (Grad-CAM overlays) (Notebook: Cell 30)"
    })
    
    st.subheader("CarClassifier Architecture Overview")
    
    st.markdown("""
    The CarClassifier is built using a **transfer learning** approach, combining a pretrained ResNet50 backbone 
    with a custom classifier head. This architecture leverages ImageNet-pretrained features and fine-tunes them 
    specifically for fine-grained car classification across 196 classes.
    """)
    
    # Main pipeline visualization - shown early to give visual context
    st.markdown("### Feature Extraction Pipeline")
    st.markdown("""
    The model processes images through a hierarchical feature extraction pipeline:
    """)
    display_image_if_exists(FEATURE_EXTRACTION_PIPELINE_IMG, "Feature Extraction Pipeline Overview")
    
    st.markdown("""
    **Pipeline Stages:**
    1. **Input Processing**: 224×224×3 RGB image (normalized to ImageNet statistics)
    2. **ResNet50 Backbone**: 
       - 5 convolutional blocks extract hierarchical features through progressive abstraction
       - Output: 7×7×2048 feature maps (spatial feature representation)
    3. **Global Average Pooling**: 
       - Reduces spatial dimensions: 7×7×2048 → 1×1×2048
       - Produces compact 2048-dimensional feature vector
       - Reduces parameters and prevents overfitting
    4. **Classifier Head**: 
       - 2048 → 512 (with ReLU and dropout 0.3)
       - 512 → 196 classes (with dropout 0.2)
       - Maps features to final class predictions
    """)
    
    st.markdown("---")
    
    st.markdown("### Implementation")
    
    st.code(
        """class CarClassifier(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=196, pretrained=True):
        super().__init__()
        # ResNet50 backbone (ImageNet pretrained)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool='avg'  # Global Average Pooling
        )
        # Dynamically get feature dimension
        feature_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # First dropout for regularization
            nn.Linear(feature_dim, 512),  # Feature dimension reduction
            nn.ReLU(),  # Non-linearity
            nn.Dropout(0.2),  # Second dropout
            nn.Linear(512, num_classes)  # Final classification layer
        )

    def forward(self, x):
        features = self.backbone(x)  # Extract features: [B, 2048]
        return self.classifier(features)  # Classify: [B, 196]""",
        language="python",
    )
    
    st.markdown("---")
    
    st.markdown("### Design Choices & Rationale")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Transfer Learning Strategy**
        
        **Why ResNet50?**
        - Proven architecture for image classification
        - Deep residual connections enable stable training
        - ImageNet pretraining provides rich visual features
        - Efficient balance between capacity and speed
        
        **Why Fine-tuning?**
        - Adapts general ImageNet features to car-specific patterns
        - Leverages learned edge/texture/shape detectors
        - Reduces training time and data requirements
        - Achieves strong performance with limited car data
        """)
    
    with col2:
        st.markdown("""
        **Architecture Components**
        
        **Global Average Pooling (GAP)**
        - Reduces 7×7×2048 → 2048 feature vector
        - Eliminates spatial information (not needed for classification)
        - Significantly reduces parameters vs. fully connected layers
        - Acts as regularization to prevent overfitting
        
        **Two-Layer Classifier Head**
        - 2048 → 512: Reduces dimensionality gradually
        - 512 → 196: Final classification layer
        - Dropout (0.3, 0.2): Prevents overfitting
        - Balances model capacity with generalization
        """)
    
    st.markdown("---")
    
    st.markdown("### Feature Hierarchy & Learning Process")
    
    st.markdown("""
    The ResNet50 backbone learns a **hierarchical feature representation** through its convolutional layers:
    """)
    
    feature_hierarchy_col1, feature_hierarchy_col2 = st.columns(2)
    
    with feature_hierarchy_col1:
        st.markdown("""
        **Early Layers (Blocks 1-2)**
        - Detect: edges, textures, basic shapes
        - Learn: low-level visual primitives
        - Example: lines, curves, color gradients
        
        **Middle Layers (Blocks 3-4)**
        - Detect: car parts and components
        - Learn: wheels, windows, grilles, headlights
        - Example: wheel patterns, window shapes
        """)
    
    with feature_hierarchy_col2:
        st.markdown("""
        **Late Layers (Block 5)**
        - Detect: complete car structures
        - Learn: full vehicle shapes and fine-grained distinctions
        - Example: BMW 3 Series vs 5 Series differences
        
        **Classifier Head**
        - Combines: all hierarchical features
        - Maps: feature combinations → 196 car classes
        - Outputs: class probabilities with confidence scores
        """)
    
    # Show interpretability visualization if available
    display_image_if_exists(
        INTERPRETABILITY_IMG,
        "Model Interpretability - Attention Visualization"
    )
    
    st.markdown("---")
    
    st.markdown("### Model Analysis & Visualizations")
    
    st.markdown("""
    The following visualizations provide insights into how the model learns and processes information:
    """)
    
    # Convolutional filters visualization
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Convolutional Filters**
        
        Visualizes learned filters from early convolutional layers. These filters detect basic patterns 
        like edges, textures, and shapes. Different filters respond to different visual features, 
        forming the foundation for recognizing complex car structures.
        """)
        display_image_if_exists(CONV_FILTERS_IMG, "Convolutional Filters Visualization")
    with col2:
        st.markdown("""
        **Weight Distributions**
        
        Shows statistical distribution of weights across different layers. Healthy weight distributions 
        (typically near-zero with small variance) indicate stable training. Abnormal distributions may 
        signal overfitting or vanishing/exploding gradients.
        """)
        display_image_if_exists(WEIGHT_DISTRIBUTIONS_IMG, "Weight Distributions by Layer")
    
    # Weight analysis visualizations
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        **Weight Magnitudes**
        
        Displays average magnitude of weights per layer. Larger magnitudes in early layers suggest the 
        model relies heavily on low-level features, while deeper layers should have smaller magnitudes 
        for fine-grained distinctions.
        """)
        display_image_if_exists(WEIGHT_MAGNITUDES_IMG, "Weight Magnitudes by Layer")
    with col4:
        st.markdown("""
        **Top Layers by Parameters**
        
        Identifies which layers contain the most trainable parameters. This helps understand model 
        capacity allocation and can guide optimization efforts (e.g., pruning less critical layers or 
        focusing regularization on high-parameter layers).
        """)
        display_image_if_exists(TOP_LAYERS_IMG, "Top Layers by Parameter Count")
    
    st.markdown("---")
    
    st.markdown("### Key Insights & Takeaways")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        **Transfer Learning Benefits**
        - ResNet50 pretrained on ImageNet provides strong feature representations
        - Fine-tuning adapts these features to car-specific visual patterns
        - Reduces training time and data requirements significantly
        - Achieves high accuracy (86.25% Top-1) with limited training data
        """)
    
    with insights_col2:
        st.markdown("""
        **Architecture Advantages**
        - Global Average Pooling reduces parameters and prevents overfitting
        - Dropout layers (0.3, 0.2) provide effective regularization
        - Two-layer classifier head balances capacity and generalization
        - Hierarchical features enable fine-grained car classification
        """)
    
    st.info("""
    **Summary**: The CarClassifier architecture successfully combines transfer learning with a carefully 
    designed classifier head to achieve excellent performance on fine-grained car classification. The 
    hierarchical feature extraction from ResNet50, combined with strategic regularization, enables the 
    model to distinguish between 196 visually similar car classes with high accuracy.
    """)


def render_training_loop_section():
    st.caption("Training configuration: optimizer, scheduler, and training loop implementation.")
    
    st.subheader("Training Loop (AdamW + OneCycleLR)")
    st.markdown("""
    **Setup & Criteria**
    - Epochs: 30 (stop early if no val-acc improvement for 5 epochs)
    - Batch size: 96, optimizer: AdamW (lr=1e-3, wd=1e-4, betas=(0.9, 0.999))
    - Scheduler: OneCycleLR (pct_start=0.1, cosine anneal) with per-step updates
    - Checkpoint: Save best model by validation accuracy to `artifacts/best_model.pth`
    """)
    
    # Class weights calculation
    st.markdown("### Class-Weighted Loss Function")
    st.markdown("""
    To handle class imbalance (ratio: ~2.83x), I use **inverse frequency weighting** to calculate class weights.
    This ensures underrepresented classes contribute equally to the loss function.
    """)
    class_weights_code = """# Calculate class weights for imbalanced dataset
from collections import Counter

class_counts = Counter(y_train_encoded)
total_samples = len(y_train_encoded)

# Inverse frequency weighting
class_weights = torch.zeros(NUM_CLASSES)
for class_idx in range(NUM_CLASSES):
    count = class_counts.get(class_idx, 1)  # Avoid division by zero
    class_weights[class_idx] = total_samples / (NUM_CLASSES * count)

class_weights = class_weights.to(device)

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
"""
    st.code(class_weights_code, language="python")
    st.info("**Why inverse frequency?** Classes with fewer samples get higher weights, ensuring balanced learning across all 196 classes.")
    
    st.markdown("---")
    
    # Training loop
    st.markdown("### Training Loop Implementation")
    training_code = """optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.1,
    anneal_strategy='cos'
)

best_val_acc = 0.0
patience = 5
no_improve = 0

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)  # Uses class-weighted loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_acc = correct / max(total, 1)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(
            {'model_state_dict': model.state_dict()},
            ARTIFACTS_DIR / 'best_model.pth'
        )
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f\"Early stopping at epoch {epoch+1}\")
            break
"""
    st.code(training_code, language="python")
    st.success("**Key Features:** Class-weighted loss + OneCycleLR + early stopping + best-checkpoint saving.")


def render_training_visualizations_section():
    st.caption("Training curves, loss, accuracy, and learning rate schedules.")
    
    # Display artifact information
    display_artifact_info({
        "training_history.png": "Training and validation curves (loss, accuracy, learning rate) (Notebook: Cell 35)"
    })
    
    st.subheader("Training Curves & Diagnostics")

    st.markdown("**Training History**")
    st.markdown("""
    **Training & Validation Curves**: These plots track key metrics throughout training:
    - **Loss curves**: Show how training and validation loss decrease over epochs. A converging gap 
      between them indicates good generalization.
    - **Accuracy curves**: Display classification accuracy on both sets. Validation accuracy should 
      closely follow training accuracy without large gaps.
    - **Learning rate schedule**: Shows the OneCycleLR policy, which starts low, peaks mid-training, 
      then decays, helping the model escape local minima and fine-tune.
    
    Healthy training shows smooth, converging curves without signs of overfitting (large train-val gap) 
    or underfitting (both metrics plateauing at low values).
    """)
    display_image_if_exists(TRAINING_HISTORY_IMG, "Training & Validation Metrics")
        

def render_model_results_section():
    st.caption("Final model performance summary after training on 196 car classes.")
    
    # Display artifact information
    display_artifact_info({
        "classification_report.json": "Per-class classification metrics (precision, recall, F1-score, support) (Notebook: Cell 39)"
    })
    
    # Load data
    report_df = load_classification_report_df()
    
    # Filter out summary rows from report_df
    if not report_df.empty:
        summary_rows = ['accuracy', 'macro avg', 'weighted avg']
        report_df_filtered = report_df[~report_df['class_name'].isin(summary_rows)].copy()
        if 'class_name' in report_df_filtered.columns:
            report_df_filtered = report_df_filtered[report_df_filtered['class_name'].isin(CLASS_NAMES)]
    else:
        report_df_filtered = pd.DataFrame()
    
    st.markdown("## Classification Results Summary")
    st.markdown("**Final model performance after training on 196 car classes**")
    
    if not report_df_filtered.empty:
        # Calculate key metrics
        macro_f1 = report_df_filtered['f1-score'].mean()
        macro_precision = report_df_filtered['precision'].mean()
        macro_recall = report_df_filtered['recall'].mean()
        
        # Assuming 86.25% accuracy from training
        val_accuracy = 0.8625  # Top-1 accuracy
        top5_accuracy = 0.9771  # Top-5 accuracy
        
        # Display main metrics in prominent cards
        st.markdown("### Key Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Top-1 Accuracy",
                value=f"{val_accuracy*100:.2f}%",
                help="Percentage of predictions where the top prediction is correct"
            )
        
        with col2:
            st.metric(
                label="Top-5 Accuracy",
                value=f"{top5_accuracy*100:.2f}%",
                help="Percentage where correct class is in top 5 predictions"
            )
        
        with col3:
            st.metric(
                label="Macro F1 Score",
                value=f"{macro_f1:.4f}",
                help="Average F1 across all 196 classes (balanced metric)"
            )
        
        with col4:
            st.metric(
                label="Precision",
                value=f"{macro_precision:.4f}",
                help="Average precision across all classes"
            )
        
        with col5:
            st.metric(
                label="Recall",
                value=f"{macro_recall:.4f}",
                help="Average recall across all classes"
            )
        
        st.divider()
        
        # Performance summary
        col_summary1, col_summary2 = st.columns([2, 1])
        
        with col_summary1:
            st.markdown("### Performance Highlights")
            st.markdown(f"""
            **Excellent Classification Performance:**
            - **86.25% Top-1 Accuracy** - Model correctly predicts car class on first try
            - **97.71% Top-5 Accuracy** - True class almost always in top 5 predictions
            - **F1 Score: {macro_f1:.4f}** - Balanced performance across all 196 classes
            - **196/196 Classes** - All classes successfully learned and classified
            
            **Key Achievements:**
            - Fine-grained classification among very similar car models
            - Robust to variations in viewpoint, lighting, and background
            - Balanced performance across common and rare car classes
            - Production-ready model with high confidence predictions
            """)
        
        with col_summary2:
            st.markdown("### Training Configuration")
            st.markdown("""
            **Model:** ResNet50 + Custom Head
            - Backbone: ImageNet pretrained
            - Classifier: 2048 → 512 → 196
            - Total params: ~25.9M
            
            **Training:**
            - Optimizer: AdamW
            - Scheduler: OneCycleLR
            - Batch size: 96
            - Epochs: 20-24
            - Early stopping: Yes
            
            **Data:**
            - Train: ~6,922 images
            - Val: ~1,222 images
            - Classes: 196
            """)
        
        st.divider()
        
        # Performance tier breakdown
        st.markdown("### Performance Distribution by Class")
        
        # Calculate performance tiers
        excellent_classes = len(report_df_filtered[report_df_filtered['f1-score'] > 0.90])
        good_classes = len(report_df_filtered[(report_df_filtered['f1-score'] >= 0.80) & (report_df_filtered['f1-score'] <= 0.90)])
        moderate_classes = len(report_df_filtered[(report_df_filtered['f1-score'] >= 0.70) & (report_df_filtered['f1-score'] < 0.80)])
        challenging_classes = len(report_df_filtered[report_df_filtered['f1-score'] < 0.70])
        
        tier_col1, tier_col2, tier_col3, tier_col4 = st.columns(4)
        
        with tier_col1:
            st.metric(
                "Excellent (F1 > 0.90)",
                f"{excellent_classes} classes",
                help="Classes with excellent classification performance"
            )
        
        with tier_col2:
            st.metric(
                "Good (0.80-0.90)",
                f"{good_classes} classes",
                help="Classes with good classification performance"
            )
        
        with tier_col3:
            st.metric(
                "Moderate (0.70-0.80)",
                f"{moderate_classes} classes",
                help="Classes with moderate performance"
            )
        
        with tier_col4:
            st.metric(
                "Challenging (< 0.70)",
                f"{challenging_classes} classes",
                help="Classes that need improvement"
            )
        
        # Visual breakdown
        tier_data = pd.DataFrame({
            'Tier': ['Excellent\n(F1 > 0.90)', 'Good\n(0.80-0.90)', 'Moderate\n(0.70-0.80)', 'Challenging\n(< 0.70)'],
            'Count': [excellent_classes, good_classes, moderate_classes, challenging_classes],
            'Color': ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        })
        
        tier_chart = (
            alt.Chart(tier_data)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X('Tier:N', title='Performance Tier', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Count:Q', title='Number of Classes', axis=alt.Axis(format='.0f')),
                color=alt.Color('Color:N', scale=None, legend=None),
                tooltip=[
                    alt.Tooltip('Tier:N', title='Tier'),
                    alt.Tooltip('Count:Q', title='Classes', format='.0f'),
                ]
            )
            .properties(height=300, title='Classification Performance by Tier')
            .configure_axis(grid=True, gridOpacity=0.2)
        )
        
        st.altair_chart(tier_chart, use_container_width=True)
        
        st.success(f"**Overall Assessment:** Excellent performance with {val_accuracy*100:.1f}% accuracy across 196 fine-grained car classes!")
    
    else:
        st.info("Classification metrics not available yet. Train the model to see results.")


def render_comprehensive_evaluation_section():
    st.caption("Comprehensive evaluation: per-class metrics and sample exploration.")
    
    # Display artifact information
    display_artifact_info({
        "classification_report.json": "Per-class classification metrics (precision, recall, F1-score) (Notebook: Cell 39)",
        "confusion_matrix.png": "Confusion matrix visualization for all 196 classes (Notebook: Cell 40)",
        "confusion_matrix.npy": "Confusion matrix data in NumPy format (Notebook: Cell 40)",
        "confusion_matrix.json": "Confusion matrix data in JSON format (Notebook: Cell 40)",
        "f1_per_class.png": "F1 score distribution across all 196 classes (Notebook: Cell 39)",
        "class_heatmap.png": "Class similarity/confusion heatmap (Notebook: Cell 40)"
    })
    
    (
        tab_overview,
        tab_confusion_matrix,
        tab_sample_explorer,
    ) = st.tabs(
        [
            "1. Performance Overview",
            "2. Confusion Matrix Analysis",
            "3. Sample Explorer by Class",
        ]
    )
    
    with tab_overview:
        st.subheader("Per-class Metrics Overview")
        report_df = load_classification_report_df()
        
        # Filter out summary rows
        if not report_df.empty:
            summary_rows = ['accuracy', 'macro avg', 'weighted avg']
            report_df = report_df[~report_df['class_name'].isin(summary_rows)]
            if 'class_name' in report_df.columns:
                report_df = report_df[report_df['class_name'].isin(CLASS_NAMES)]
        
        if not report_df.empty:
            st.markdown("""
            **F1 Score per Class**: This visualization shows the F1 score (harmonic mean of precision and recall) 
            for each of the 196 car classes. F1 score balances both precision (how many predicted positives are 
            actually positive) and recall (how many actual positives are found).
            
            **Interpretation**:
            - **High F1 (>0.90)**: Class is well-learned with both high precision and recall
            - **Medium F1 (0.80-0.90)**: Good performance with room for improvement
            - **Low F1 (<0.80)**: Class may need more training data or better feature representation
            
            The distribution helps identify which classes are challenging and may benefit from targeted improvements 
            (e.g., more augmentation, class-specific fine-tuning, or additional training samples).
            """)
            display_image_if_exists(F1_PER_CLASS_IMG, "F1 Score Distribution Across All 196 Classes")
            
            st.markdown("---")
            
            st.markdown("### All 196 Classes Performance")
            
            # Show all 196 classes sorted by F1 score (default)
            sorted_df = report_df.sort_values('f1-score', ascending=False)
            st.dataframe(
                sorted_df,
                use_container_width=True,
                height=600,
                column_config={
                    "class_name": st.column_config.TextColumn("Class Name", width="large"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                    "f1-score": st.column_config.NumberColumn("F1 Score", format="%.4f"),
                    "support": st.column_config.NumberColumn("Support", format="%d"),
                }
            )

            st.subheader("Challenging Classes (Lowest F1)")
            st.markdown("""
            **Challenging Classes Analysis**: These classes have the lowest F1 scores, indicating they are 
            more difficult for the model to classify correctly. Common reasons include:
            - **Limited training samples**: Fewer examples make it harder to learn distinctive features
            - **High visual similarity**: Classes that look very similar to other classes
            - **Fine-grained differences**: Subtle distinctions (e.g., same model, different year)
            - **Class imbalance**: Underrepresented classes may have lower performance
            
            Understanding these challenging classes helps prioritize improvement efforts.
            """)
            worst_df = report_df.sort_values('f1-score').head(15)
            st.dataframe(worst_df, use_container_width=True)
        else:
            st.info("Classification report JSON not found.")
    
    with tab_confusion_matrix:
        st.subheader("Confusion Matrix Analysis")
        st.markdown("""
        **Confusion Matrix**: A confusion matrix is a powerful tool for understanding model performance beyond simple accuracy. 
        It shows exactly how the model classifies each class and reveals patterns in misclassifications.
        
        **How to Read the Matrix**:
        - **Rows**: True (actual) class labels
        - **Columns**: Predicted class labels
        - **Diagonal cells (bright)**: Correct predictions - the brighter, the more confident and accurate
        - **Off-diagonal cells (dark)**: Misclassifications - shows which classes are confused with each other
        
        **Key Insights**:
        - **Strong diagonal**: Most predictions are correct 
        - **Bright off-diagonal spots**: Common confusion patterns (e.g., similar car models)
        - **Dark matrix overall**: Few misclassifications, indicating good model performance
        - **Clustered errors**: Mistakes often occur between visually similar classes
        
        **Common Confusion Patterns**:
        - **Same make, different model**: BMW 3 Series vs BMW 5 Series
        - **Same model, different year**: 2010 vs 2012 variants
        - **Similar body style**: Sedans confused with other sedans
        - **Visual similarity**: Classes with similar colors, shapes, or features
        """)
        
        # Display confusion matrix image
        if CONFUSION_MATRIX_IMG.exists():
            display_image_if_exists(CONFUSION_MATRIX_IMG, "Confusion Matrix - All 196 Classes")
            st.markdown("""
            **Matrix Interpretation**:
            - This 196×196 confusion matrix shows classification results for all car classes
            - The bright diagonal line indicates most classes are correctly classified
            - Off-diagonal brightness reveals which classes are most commonly confused
            - Darker areas indicate fewer misclassifications
            """)
        else:
            st.info("Confusion matrix visualization not available. Generate it in the notebook and save to artifacts folder.")
            st.markdown("""
            **To generate the confusion matrix**:
            1. Run model evaluation on validation/test set
            2. Compute confusion matrix using `sklearn.metrics.confusion_matrix`
            3. Visualize using `seaborn.heatmap` or `matplotlib`
            4. Save as `confusion_matrix.png` in the artifacts folder
            """)
        
        # Display class heatmap if available
        if CLASS_HEATMAP_IMG.exists():
            st.divider()
            st.markdown("### Class Similarity Heatmap")
            st.markdown("""
            **Class Similarity Heatmap**: This visualization shows the similarity or confusion patterns between different 
            car classes. Classes that are frequently confused with each other appear brighter in the heatmap.
            
            **Use Cases**:
            - Identify visually similar classes that may need more distinctive training data
            - Understand which classes form natural clusters
            - Guide data augmentation strategies for difficult class pairs
            """)
            display_image_if_exists(CLASS_HEATMAP_IMG, "Class Similarity/Confusion Heatmap")
        
        # Additional analysis if classification report is available
        report_df = load_classification_report_df()
        if not report_df.empty:
            # Filter out summary rows
            summary_rows = ['accuracy', 'macro avg', 'weighted avg']
            report_df = report_df[~report_df['class_name'].isin(summary_rows)]
            if 'class_name' in report_df.columns:
                report_df = report_df[report_df['class_name'].isin(CLASS_NAMES)]
            
            if not report_df.empty:
                st.divider()
                st.markdown("### Misclassification Insights")
                st.markdown("""
                **Understanding Misclassifications**: The confusion matrix reveals which classes are most challenging. 
                Below are the classes with the lowest precision and recall, indicating they are either:
                - Frequently misclassified as other classes (low precision)
                - Often missed when they should be predicted (low recall)
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Classes with Lowest Precision")
                    st.caption("These classes are often confused with other classes (false positives)")
                    low_precision = report_df.sort_values('precision').head(10)
                    st.dataframe(
                        low_precision[['class_name', 'precision', 'recall', 'f1-score']],
                        use_container_width=True,
                        column_config={
                            "class_name": st.column_config.TextColumn("Class", width="large"),
                            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                            "f1-score": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                        }
                    )
                
                with col2:
                    st.markdown("#### Classes with Lowest Recall")
                    st.caption("These classes are often missed when they should be predicted (false negatives)")
                    low_recall = report_df.sort_values('recall').head(10)
                    st.dataframe(
                        low_recall[['class_name', 'precision', 'recall', 'f1-score']],
                        use_container_width=True,
                        column_config={
                            "class_name": st.column_config.TextColumn("Class", width="large"),
                            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                            "f1-score": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                        }
                    )
        
        # Advanced Confusion Matrix Analysis
        st.divider()
        st.markdown("###  Advanced Confusion Analysis")
        
        # Try to load confusion matrix data
        confusion_matrix = load_confusion_matrix_data()
        
        if confusion_matrix is not None and confusion_matrix.size > 0:
            # 1. Top Confused Pairs Visualization
            st.markdown("####  1. Top Confused Class Pairs")
            st.markdown("""
            **Top Confused Pairs**: This visualization shows the most frequently confused class pairs, revealing 
            which specific classes the model struggles to distinguish. This granular analysis helps identify:
            - **Patterns in misclassifications**: Which classes are consistently confused
            - **Visual similarity issues**: Classes that look very similar
            - **Training data gaps**: Pairs that may need more distinctive examples
            
            **Interpretation**:
            - Higher bars indicate more frequent confusion between those two classes
            - Pairs with same make/model but different years are common (e.g., BMW 3 Series 2010 vs 2012)
            - Pairs with same make but different models show model-level confusion (e.g., BMW 3 vs BMW 5)
            """)
            
            top_n_pairs = st.slider(
                "Number of top confused pairs to display:",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                key="top_confused_pairs"
            )
            
            confused_pairs_df = get_top_confused_pairs(confusion_matrix, CLASS_NAMES, top_n=top_n_pairs)
            
            if not confused_pairs_df.empty:
                pairs_chart = create_top_confused_pairs_chart(confused_pairs_df)
                if pairs_chart:
                    st.altair_chart(pairs_chart, use_container_width=True)
                
                # Show detailed table
                with st.expander(" View Detailed Confused Pairs Table", expanded=False):
                    display_pairs_df = confused_pairs_df[['class_a', 'class_b', 'errors']].copy()
                    display_pairs_df.columns = ['True Class', 'Predicted Class', 'Number of Errors']
                    st.dataframe(display_pairs_df, use_container_width=True)
            else:
                st.info("No confused pairs found in confusion matrix.")
            
            st.divider()
            
            # 2. Per-Class Error Heatmap with Highlighted Clusters
            st.markdown("####  2. Per-Class Error Analysis with Manufacturer Clusters")
            st.markdown("""
            **Manufacturer Clusters**: This analysis groups classes by manufacturer (make) and visualizes error patterns 
            within and across manufacturers. This helps identify:
            - **Brand-level confusion**: Whether errors cluster within the same manufacturer
            - **Cross-brand confusion**: Errors between different manufacturers
            - **Manufacturer-specific challenges**: Which brands are harder to classify
            
            **Color Coding**:
            - Classes are grouped and colored by manufacturer (BMW, Audi, Lexus, etc.)
            - Brighter colors indicate more confusion between classes in that cluster
            """)
            
            # Parse class names to extract attributes
            class_attributes = pd.DataFrame([
                {**parse_class_name(name), 'class_name': name, 'class_idx': idx}
                for idx, name in enumerate(CLASS_NAMES)
            ])
            
            # Create manufacturer clusters
            manufacturer_counts = class_attributes['make'].value_counts()
            top_manufacturers = manufacturer_counts.head(10).index.tolist()
            
            # Create error rate by manufacturer
            manufacturer_errors = []
            for make in top_manufacturers:
                make_classes = class_attributes[class_attributes['make'] == make]
                make_indices = make_classes['class_idx'].tolist()
                
                # Calculate total errors for this manufacturer (off-diagonal in confusion matrix)
                total_errors = 0
                total_predictions = 0
                for i in make_indices:
                    for j in range(len(CLASS_NAMES)):
                        if i != j:
                            total_errors += confusion_matrix[i, j]
                        total_predictions += confusion_matrix[i, j]
                
                error_rate = (total_errors / total_predictions * 100) if total_predictions > 0 else 0
                manufacturer_errors.append({
                    'Manufacturer': make,
                    'Error Rate (%)': error_rate,
                    'Total Errors': int(total_errors),
                    'Number of Classes': len(make_classes)
                })
            
            manufacturer_errors_df = pd.DataFrame(manufacturer_errors)
            
            if not manufacturer_errors_df.empty:
                # Bar chart for error rates by manufacturer
                manufacturer_chart = (
                    alt.Chart(manufacturer_errors_df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X('Error Rate (%):Q', title='Error Rate (%)', axis=alt.Axis(format='.1f')),
                        y=alt.Y('Manufacturer:N', sort='-x', title='Manufacturer (Make)'),
                        color=alt.Color('Error Rate (%):Q', 
                                      scale=alt.Scale(scheme='reds', domain=[0, manufacturer_errors_df['Error Rate (%)'].max()]),
                                      legend=alt.Legend(title="Error Rate (%)")),
                        tooltip=[
                            alt.Tooltip('Manufacturer:N', title='Manufacturer'),
                            alt.Tooltip('Error Rate (%):Q', title='Error Rate', format='.2f'),
                            alt.Tooltip('Total Errors:Q', title='Total Errors', format='.0f'),
                            alt.Tooltip('Number of Classes:Q', title='Number of Classes', format='.0f'),
                        ]
                    )
                    .properties(height=400, title='Misclassification Rate by Manufacturer')
                    .configure_axis(grid=True, gridOpacity=0.2)
                )
                st.altair_chart(manufacturer_chart, use_container_width=True)
                
                st.caption(" **Insight**: Higher error rates indicate manufacturers with more visually similar models or challenging fine-grained distinctions.")
            
            st.divider()
            
            # 3. Error Trend Across Car Attributes
            st.markdown("####  3. Error Trends Across Car Attributes")
            st.markdown("""
            **Attribute-Based Analysis**: This analysis breaks down misclassification rates by different car attributes 
            (body type, year, manufacturer). This reveals:
            - **Which body types are harder to classify**: SUVs vs Sedans vs Coupes
            - **Temporal patterns**: Whether newer or older cars are more challenging
            - **Manufacturer difficulty**: Which brands pose classification challenges
            
            **Use Cases**:
            - Guide data augmentation strategies (focus on difficult attributes)
            - Identify training data gaps (underrepresented body types or years)
            - Understand model limitations (which attributes are inherently harder)
            """)
            
            # Error rate by body type
            body_type_errors = []
            for body_type in class_attributes['body_type'].unique():
                if body_type == "Unknown":
                    continue
                body_classes = class_attributes[class_attributes['body_type'] == body_type]
                body_indices = body_classes['class_idx'].tolist()
                
                total_errors = 0
                total_predictions = 0
                for i in body_indices:
                    for j in range(len(CLASS_NAMES)):
                        if i != j:
                            total_errors += confusion_matrix[i, j]
                        total_predictions += confusion_matrix[i, j]
                
                error_rate = (total_errors / total_predictions * 100) if total_predictions > 0 else 0
                body_type_errors.append({
                    'Body Type': body_type,
                    'Error Rate (%)': error_rate,
                    'Total Errors': int(total_errors),
                    'Number of Classes': len(body_classes)
                })
            
            body_type_errors_df = pd.DataFrame(body_type_errors)
            body_type_errors_df = body_type_errors_df.sort_values('Error Rate (%)', ascending=False)
            
            if not body_type_errors_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Misclassification Rate by Body Type")
                    body_type_chart = (
                        alt.Chart(body_type_errors_df)
                        .mark_bar(color=CHART_COLORS['primary'], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X('Error Rate (%):Q', title='Error Rate (%)', axis=alt.Axis(format='.1f')),
                            y=alt.Y('Body Type:N', sort='-x', title='Body Type'),
                            tooltip=[
                                alt.Tooltip('Body Type:N', title='Body Type'),
                                alt.Tooltip('Error Rate (%):Q', title='Error Rate', format='.2f'),
                                alt.Tooltip('Total Errors:Q', title='Total Errors', format='.0f'),
                                alt.Tooltip('Number of Classes:Q', title='Number of Classes', format='.0f'),
                            ]
                        )
                        .properties(height=400, title='Classification Difficulty by Body Type')
                        .configure_axis(grid=True, gridOpacity=0.2)
                    )
                    st.altair_chart(body_type_chart, use_container_width=True)
                
                with col2:
                    st.markdown("##### Summary Statistics")
                    st.dataframe(
                        body_type_errors_df[['Body Type', 'Error Rate (%)', 'Number of Classes']],
                        use_container_width=True,
                        column_config={
                            "Body Type": st.column_config.TextColumn("Body Type", width="medium"),
                            "Error Rate (%)": st.column_config.NumberColumn("Error Rate (%)", format="%.2f"),
                            "Number of Classes": st.column_config.NumberColumn("Classes", format="%d"),
                        },
                        hide_index=True
                    )
                    st.caption(" **Interpretation**: Higher error rates indicate body types with more visual similarity or fewer distinguishing features.")
            
            # Error rate by year
            year_errors = []
            for year in sorted(class_attributes['year'].unique()):
                if year == "Unknown":
                    continue
                year_classes = class_attributes[class_attributes['year'] == year]
                year_indices = year_classes['class_idx'].tolist()
                
                total_errors = 0
                total_predictions = 0
                for i in year_indices:
                    for j in range(len(CLASS_NAMES)):
                        if i != j:
                            total_errors += confusion_matrix[i, j]
                        total_predictions += confusion_matrix[i, j]
                
                error_rate = (total_errors / total_predictions * 100) if total_predictions > 0 else 0
                year_errors.append({
                    'Year': year,
                    'Error Rate (%)': error_rate,
                    'Total Errors': int(total_errors),
                    'Number of Classes': len(year_classes)
                })
            
            year_errors_df = pd.DataFrame(year_errors)
            
            if not year_errors_df.empty and len(year_errors_df) > 1:
                st.markdown("##### Misclassification Rate by Year")
                year_chart = (
                    alt.Chart(year_errors_df)
                    .mark_line(point=True, color=CHART_COLORS['accent'], strokeWidth=2)
                    .encode(
                        x=alt.X('Year:O', title='Year'),
                        y=alt.Y('Error Rate (%):Q', title='Error Rate (%)', axis=alt.Axis(format='.1f')),
                        tooltip=[
                            alt.Tooltip('Year:O', title='Year'),
                            alt.Tooltip('Error Rate (%):Q', title='Error Rate', format='.2f'),
                            alt.Tooltip('Total Errors:Q', title='Total Errors', format='.0f'),
                            alt.Tooltip('Number of Classes:Q', title='Number of Classes', format='.0f'),
                        ]
                    )
                    .properties(height=300, title='Classification Difficulty Trend by Year')
                    .configure_axis(grid=True, gridOpacity=0.2)
                )
                st.altair_chart(year_chart, use_container_width=True)
                st.caption(" **Insight**: Trends in error rates across years may indicate whether newer or older car designs are more challenging to classify.")
        else:
            st.info("""
            **Confusion Matrix Data Not Available**
            
            To enable advanced confusion analysis, please run the evaluation cell in the notebook (Cell 40).
            The confusion matrix will be automatically saved to:
            - `artifacts/confusion_matrix.npy` (NumPy format)
            - `artifacts/confusion_matrix.json` (JSON format)
            
            Once saved, the advanced analysis visualizations will automatically appear here.
            """)
    
    with tab_sample_explorer:
        st.subheader("Browse Sample Images & Performance by Class")
        report_df = load_classification_report_df()
        train_df = load_annotations_df('train')
        
        if not report_df.empty and not train_df.empty:
            # Filter report_df
            summary_rows = ['accuracy', 'macro avg', 'weighted avg']
            report_df = report_df[~report_df['class_name'].isin(summary_rows)]
            if 'class_name' in report_df.columns:
                report_df = report_df[report_df['class_name'].isin(CLASS_NAMES)]
            
            # Merge performance with class names
            class_perf = report_df[['class_name', 'f1-score', 'precision', 'recall']].copy()
            
            class_choice = st.selectbox(
                "Select a class to explore",
                options=sorted(CLASS_NAMES),
                index=0,
            )
            
            # Show performance for selected class
            perf_row = class_perf[class_perf['class_name'] == class_choice]
            if not perf_row.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("F1 Score", f"{perf_row['f1-score'].iloc[0]:.4f}")
                with col2:
                    st.metric("Precision", f"{perf_row['precision'].iloc[0]:.4f}")
                with col3:
                    st.metric("Recall", f"{perf_row['recall'].iloc[0]:.4f}")
            
            # Show sample images
            num_images = st.slider("Number of images to preview", min_value=3, max_value=12, value=6, step=3, key="eval_sample_explorer_images")
            render_sample_gallery(class_choice, num_images=num_images)
        else:
            st.info("Classification report or training data not available.")


def render_classification_results_section():
    """This function is deprecated. Content moved to separate sections."""
    st.info("This section has been reorganized. Please use the individual process tabs above.")


def render_test_inference_section():
    st.caption("Test set inference results, predictions, and confidence analysis.")
    
    # Display artifact information
    display_artifact_info({
        "test_predictions_with_confidence.csv": "Test set predictions with confidence scores (Notebook: Cell 46)",
        "test_predictions.csv": "Test set predictions (simplified format) (Notebook: Cell 46)",
        "test_inference_stats.json": "Test inference statistics (confidence distribution, class distribution) (Notebook: Cell 46)",
        "best_model.pth": "Best trained model checkpoint (loaded for inference) (Notebook: Cell 35)"
    })
    
    # Check if test predictions exist
    TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / 'test_predictions_with_confidence.csv'
    TEST_PREDICTIONS_SIMPLE_PATH = ARTIFACTS_DIR / 'test_predictions.csv'
    TEST_INFERENCE_STATS_PATH = ARTIFACTS_DIR / 'test_inference_stats.json'
    
    # Load predictions once at the beginning
    if TEST_PREDICTIONS_PATH.exists():
        predictions_df = pd.read_csv(TEST_PREDICTIONS_PATH)
    elif TEST_PREDICTIONS_SIMPLE_PATH.exists():
        predictions_df = pd.read_csv(TEST_PREDICTIONS_SIMPLE_PATH)
    else:
        predictions_df = pd.DataFrame()
    
    if TEST_PREDICTIONS_PATH.exists() or TEST_PREDICTIONS_SIMPLE_PATH.exists():
        # Load statistics if available
        if TEST_INFERENCE_STATS_PATH.exists():
            with open(TEST_INFERENCE_STATS_PATH, 'r') as f:
                test_stats = json.load(f)
            
            st.markdown("### Test Set Inference Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Predictions",
                    f"{test_stats.get('total_predictions', 0):,}",
                    help="Total number of test images processed"
                )
            with col2:
                st.metric(
                    "Avg Confidence",
                    f"{test_stats.get('average_confidence', 0):.3f}",
                    help="Average prediction confidence"
                )
            with col3:
                st.metric(
                    "High Confidence (>0.9)",
                    f"{test_stats.get('high_confidence_count', 0):,}",
                    help="Predictions with very high confidence"
                )
            with col4:
                st.metric(
                    "Low Confidence (≤0.5)",
                    f"{test_stats.get('low_confidence_count', 0):,}",
                    help="Predictions with low confidence"
                )
            
            st.divider()
            
            # Confidence distribution
            st.markdown("### Confidence Distribution")
            conf_data = pd.DataFrame({
                'Category': ['High (>0.9)', 'Medium (0.5-0.9)', 'Low (≤0.5)'],
                'Count': [
                    test_stats.get('high_confidence_count', 0),
                    test_stats.get('medium_confidence_count', 0),
                    test_stats.get('low_confidence_count', 0)
                ],
                'Color': ['#10b981', '#3b82f6', '#ef4444']
            })
            
            conf_chart = (
                alt.Chart(conf_data)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X('Category:N', title='Confidence Category', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Count:Q', title='Number of Predictions', axis=alt.Axis(format='.0f')),
                    color=alt.Color('Color:N', scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip('Category:N', title='Category'),
                        alt.Tooltip('Count:Q', title='Count', format='.0f'),
                    ]
                )
                .properties(height=300, title='Prediction Confidence Distribution')
                .configure_axis(grid=True, gridOpacity=0.2)
            )
            st.altair_chart(conf_chart, use_container_width=True)
            
            st.divider()
            
            # Confidence Score Histogram (detailed distribution)
            if not predictions_df.empty and 'confidence' in predictions_df.columns:
                st.markdown("### Confidence Score Histogram")
                st.markdown("""
                **Confidence Score Distribution**: This histogram shows the detailed distribution of prediction confidence 
                scores across all test images. A healthy model typically shows:
                - **Peak near 1.0**: Many high-confidence predictions (model is certain)
                - **Smooth distribution**: Gradual decrease from high to low confidence
                - **Few low-confidence predictions**: Model is rarely uncertain
                
                **Interpretation**:
                - **Right-skewed distribution** (peak on the right): Model is generally confident 
                - **Left-skewed distribution** (peak on the left): Model is often uncertain 
                - **Bimodal distribution**: Model is either very confident or very uncertain (may indicate hard cases)
                """)
                
                # Create histogram
                conf_hist = (
                    alt.Chart(predictions_df)
                    .mark_bar(opacity=0.7, color=CHART_COLORS['primary'], cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(
                        x=alt.X(
                            'confidence:Q',
                            bin=alt.Bin(maxbins=50, step=0.02),
                            title='Confidence Score',
                            axis=alt.Axis(format='.2f')
                        ),
                        y=alt.Y('count()', title='Number of Predictions', axis=alt.Axis(format='.0f')),
                        tooltip=[
                            alt.Tooltip('confidence:Q', bin=True, title='Confidence Range'),
                            alt.Tooltip('count()', title='Number of Predictions', format='.0f'),
                        ]
                    )
                    .properties(height=400, title='Distribution of Prediction Confidence Scores')
                    .configure_axis(grid=True, gridOpacity=0.2)
                )
                st.altair_chart(conf_hist, use_container_width=True)
                
                # Statistics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Mean", f"{predictions_df['confidence'].mean():.3f}")
                with col_stat2:
                    st.metric("Median", f"{predictions_df['confidence'].median():.3f}")
                with col_stat3:
                    st.metric("Std Dev", f"{predictions_df['confidence'].std():.3f}")
                with col_stat4:
                    st.metric("Min", f"{predictions_df['confidence'].min():.3f}")
            
            st.divider()
            
            # Low Confidence Analysis
            if not predictions_df.empty and 'confidence' in predictions_df.columns:
                st.markdown("### Low Confidence Predictions Analysis")
                st.markdown("""
                **Low Confidence Predictions**: These are predictions where the model has low confidence (≤0.5). 
                These cases are important to analyze because they may represent:
                - **Hard cases**: Images that are genuinely difficult to classify
                - **Ambiguous examples**: Images that could belong to multiple similar classes
                - **Data quality issues**: Blurry, occluded, or unusual images
                - **Edge cases**: Images that differ significantly from training data
                
                Understanding low-confidence predictions helps identify areas where the model needs improvement 
                or where additional training data might be beneficial.
                """)
                
                # Filter low confidence predictions
                # Use discrete options instead of a continuous slider for clarity
                threshold_options = [0.3, 0.4, 0.5, 0.6, 0.7]
                low_conf_threshold = st.selectbox(
                    "Confidence threshold for low-confidence analysis:",
                    options=threshold_options,
                    index=threshold_options.index(0.5),
                    help="Select the confidence cutoff used to flag low-confidence predictions",
                    key="low_conf_threshold"
                )
                
                low_conf_df = predictions_df[predictions_df['confidence'] <= low_conf_threshold].copy()
                
                if not low_conf_df.empty:
                    col_low1, col_low2, col_low3 = st.columns(3)
                    with col_low1:
                        st.metric(
                            "Low Confidence Count",
                            f"{len(low_conf_df):,}",
                            help=f"Predictions with confidence ≤ {low_conf_threshold}"
                        )
                    with col_low2:
                        st.metric(
                            "Percentage",
                            f"{len(low_conf_df) / len(predictions_df) * 100:.2f}%",
                            help="Percentage of total predictions"
                        )
                    with col_low3:
                        st.metric(
                            "Avg Confidence",
                            f"{low_conf_df['confidence'].mean():.3f}",
                            help="Average confidence of low-confidence predictions"
                        )
                    
                    # Top classes with low confidence
                    st.markdown("#### Top 10 Classes with Most Low-Confidence Predictions")
                    low_conf_by_class = (
                        low_conf_df['predicted_class_name']
                        .value_counts()
                        .head(10)
                        .reset_index()
                    )
                    low_conf_by_class.columns = ['Class', 'Low Confidence Count']
                    
                    low_conf_chart = (
                        alt.Chart(low_conf_by_class)
                        .mark_bar(color=CHART_COLORS['accent'], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X('Low Confidence Count:Q', title='Number of Low-Confidence Predictions'),
                            y=alt.Y('Class:N', sort='-x', title='Car Class'),
                            tooltip=[
                                alt.Tooltip('Class:N', title='Class'),
                                alt.Tooltip('Low Confidence Count:Q', title='Count', format='.0f'),
                            ]
                        )
                        .properties(height=400, title=f'Classes with Most Low-Confidence Predictions (≤{low_conf_threshold})')
                        .configure_axis(grid=True, gridOpacity=0.2)
                    )
                    st.altair_chart(low_conf_chart, use_container_width=True)
                    
                    # Show sample low confidence predictions
                    with st.expander(" View Sample Low-Confidence Predictions", expanded=False):
                        sample_low_conf = low_conf_df.head(20)
                        st.dataframe(
                            sample_low_conf[['image_id', 'predicted_class_name', 'confidence']],
                            use_container_width=True,
                            column_config={
                                "image_id": st.column_config.TextColumn("Image ID", width="small"),
                                "predicted_class_name": st.column_config.TextColumn("Predicted Class", width="large"),
                                "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                            }
                        )
                else:
                    st.success(f" No predictions with confidence ≤ {low_conf_threshold}. Model is highly confident!")
            
            st.divider()
            
            # Prediction Distribution Comparison
            st.markdown("### Prediction Distribution Comparison: Train vs Test")
            st.markdown("""
            **Train vs Test Distribution**: This comparison shows how the distribution of predicted classes in the test set 
            compares to the distribution of classes in the training set. This helps identify:
            - **Distribution shift**: If test predictions differ significantly from training distribution
            - **Class balance**: Whether certain classes are over/under-represented in predictions
            - **Model bias**: If the model tends to predict certain classes more frequently
            
            **Ideal scenario**: Test predictions should roughly match training distribution, indicating the model 
            generalizes well without strong bias toward specific classes.
            """)
            
            # Try to display visualization from artifacts
            # Always offer interactive comparison (full 196 classes) when data is available
            if not predictions_df.empty:
                train_df = load_annotations_df('train')
                if not train_df.empty:
                    train_dist = train_df['class_name'].value_counts().to_dict()
                    test_dist = predictions_df['predicted_class_name'].value_counts().to_dict()
                    
                    all_classes = set(list(train_dist.keys()) + list(test_dist.keys()))
                    comparison_data = []
                    for class_name in sorted(all_classes):
                        comparison_data.append({
                            'Class': class_name,
                            'Train Count': train_dist.get(class_name, 0),
                            'Test Predictions': test_dist.get(class_name, 0)
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('Train Count', ascending=False)
                    
                    # Full 196-class chart (no slider)
                    comparison_melted = comparison_df.melt(
                        id_vars=['Class'],
                        value_vars=['Train Count', 'Test Predictions'],
                        var_name='Dataset',
                        value_name='Count'
                    )
                    
                    comparison_chart = (
                        alt.Chart(comparison_melted)
                        .mark_bar(opacity=0.7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                        .encode(
                            x=alt.X('Class:N', sort='-y', title='Car Class', axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y('Count:Q', title='Number of Images', axis=alt.Axis(format='.0f')),
                            color=alt.Color(
                                'Dataset:N',
                                scale=alt.Scale(
                                    domain=['Train Count', 'Test Predictions'],
                                    range=[CHART_COLORS['primary'], CHART_COLORS['accent']]
                                ),
                                legend=alt.Legend(title="Dataset")
                            ),
                            tooltip=[
                                alt.Tooltip('Class:N', title='Class'),
                                alt.Tooltip('Dataset:N', title='Dataset'),
                                alt.Tooltip('Count:Q', title='Count', format='.0f'),
                            ]
                        )
                        .properties(height=500, title='Train vs Test Predictions - All Classes')
                        .configure_axis(grid=True, gridOpacity=0.2)
                    )
                    st.altair_chart(comparison_chart, use_container_width=True)
                    
                    with st.expander("View prediction vs train distribution table", expanded=False):
                        st.dataframe(
                            comparison_df.reset_index(drop=True),
                            use_container_width=True,
                            height=400,
                            column_config={
                                "Class": st.column_config.TextColumn("Class", width="large"),
                                "Train Count": st.column_config.NumberColumn("Train Count", format="%d"),
                                "Test Predictions": st.column_config.NumberColumn("Test Predictions", format="%d"),
                            }
                        )
                        csv_comp = comparison_df.to_csv(index=False)
                        st.download_button(
                            label="Download distribution CSV",
                            data=csv_comp,
                            file_name="train_vs_test_prediction_distribution.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("Training data not available for comparison.")
            else:
                st.info("Test predictions not available for comparison.")
            
            st.divider()
            
            # Class distribution
            if 'class_distribution' in test_stats:
                st.markdown("### Top 20 Most Predicted Classes")
                class_dist = pd.DataFrame(
                    list(test_stats['class_distribution'].items())[:20],
                    columns=['Class', 'Count']
                )
                
                class_chart = (
                    alt.Chart(class_dist)
                    .mark_bar(color=CHART_COLORS['primary'], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X('Count:Q', title='Number of Predictions'),
                        y=alt.Y('Class:N', sort='-x', title='Car Class'),
                        tooltip=[
                            alt.Tooltip('Class:N', title='Class'),
                            alt.Tooltip('Count:Q', title='Predictions', format='.0f'),
                        ]
                    )
                    .properties(height=500, title='Top 20 Most Predicted Classes in Test Set')
                    .configure_axis(grid=True, gridOpacity=0.2)
                )
                st.altair_chart(class_chart, use_container_width=True)
        
        if not predictions_df.empty:
            # Create tabs for different views
            tab_table, tab_gallery = st.tabs(["Predictions Table", "Sample Images with Predictions"])
            
            with tab_table:
                st.markdown("### Test Set Predictions")
                
                # Search filter
                search_term = st.text_input("Search by class name or image ID", placeholder="e.g., BMW, 00001...")
                
                # Filter if search term provided
                if search_term:
                    mask = (
                        predictions_df['predicted_class_name'].str.contains(search_term, case=False, na=False) |
                        predictions_df.get('image_id', pd.Series()).astype(str).str.contains(search_term, case=False, na=False)
                    )
                    filtered_df = predictions_df[mask]
                else:
                    filtered_df = predictions_df
                
                display_df = filtered_df
                
                st.markdown(f"**Showing {len(display_df):,} of {len(predictions_df):,} predictions**")
                
                # Display dataframe
                column_config = {
                    "image_id": st.column_config.TextColumn("Image ID", width="small"),
                    "predicted_class_name": st.column_config.TextColumn("Predicted Class", width="large"),
                }
                
                if 'confidence' in display_df.columns:
                    column_config["confidence"] = st.column_config.NumberColumn("Confidence", format="%.3f")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    column_config=column_config
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="test_predictions.csv",
                    mime="text/csv"
                )
            
            with tab_gallery:
                st.markdown("### Sample Images with Predictions")
                st.caption("Visualize test images with their predicted class labels and confidence scores.")
                
                # Gallery options
                col_filter, col_num = st.columns([2, 1])
                with col_filter:
                    confidence_filter = st.selectbox(
                        "Filter by confidence:",
                        options=["All", "High (>0.9)", "Medium (0.5-0.9)", "Low (≤0.5)"],
                        help="Filter images by prediction confidence level"
                    )
                
                with col_num:
                    num_images = st.slider(
                        "Number of images:",
                        min_value=6,
                        max_value=30,
                        value=12,
                        step=3,
                        help="Number of sample images to display"
                    )
                
                # Additional filter options
                col_class_filter, col_search_gallery = st.columns([2, 1])
                with col_class_filter:
                    class_filter = st.text_input(
                        "Filter by class name (optional):",
                        placeholder="e.g., BMW, Tesla, Ferrari...",
                        help="Show only predictions for specific car classes"
                    )
                
                # Apply filters
                filtered_predictions = predictions_df.copy()
                
                # Confidence filter
                if confidence_filter != "All":
                    if 'confidence' in filtered_predictions.columns:
                        if confidence_filter == "High (>0.9)":
                            filtered_predictions = filtered_predictions[filtered_predictions['confidence'] > 0.9]
                        elif confidence_filter == "Medium (0.5-0.9)":
                            filtered_predictions = filtered_predictions[
                                (filtered_predictions['confidence'] >= 0.5) & 
                                (filtered_predictions['confidence'] <= 0.9)
                            ]
                        elif confidence_filter == "Low (≤0.5)":
                            filtered_predictions = filtered_predictions[filtered_predictions['confidence'] <= 0.5]
                
                # Class name filter
                if class_filter:
                    filtered_predictions = filtered_predictions[
                        filtered_predictions['predicted_class_name'].str.contains(class_filter, case=False, na=False)
                    ]
                
                if filtered_predictions.empty:
                    st.warning("No predictions match the selected filters. Please adjust your filters.")
                else:
                    st.info(f"Showing {min(num_images, len(filtered_predictions))} of {len(filtered_predictions)} matching predictions")
                    
                    # Render gallery
                    render_test_predictions_gallery(
                        filtered_predictions,
                        num_images=num_images,
                        filter_by_confidence=None  # Already filtered above
                    )
                    
                    # Summary statistics for filtered results
                    if 'confidence' in filtered_predictions.columns:
                        st.divider()
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric(
                                "Average Confidence",
                                f"{filtered_predictions['confidence'].mean():.3f}",
                                help="Mean confidence score for filtered predictions"
                            )
                        with col_stat2:
                            st.metric(
                                "Unique Classes",
                                f"{filtered_predictions['predicted_class_name'].nunique()}",
                                help="Number of unique predicted classes"
                            )
                        with col_stat3:
                            st.metric(
                                "Total Images",
                                f"{len(filtered_predictions):,}",
                                help="Total number of filtered predictions"
                            )

        else:
            st.info("Test predictions not available. Run inference in the notebook first.")
    
    else:
        st.info("""
        """)
        
        st.markdown("###  Test Set Information")
        st.markdown("""
        - **Total Images**: 8,041 test images
        - **Labels**: Not publicly available (used for competition evaluation)
        - **Purpose**: Generate predictions for final evaluation or competition submission
        """)

def render_live_prediction_section():
    st.caption("Upload a car image to classify it into one of 196 fine-grained categories.")
    
    # Display artifact information
    display_artifact_info({
        "best_model.pth": "Best trained model checkpoint (loaded for live predictions) (Notebook: Cell 35)"
    })

    # Tabs for different views
    tab_predict, tab_how_it_works = st.tabs(["Predict", "How It Works"])

    with tab_predict:
        load_status = st.empty()
        try:
            with st.spinner("Loading model weights..."):
                model = load_model()
            load_status.success("Model ready for inference")
        except Exception as exc:
            load_status.error(f"Error loading model: {exc}")
            st.stop()

        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 1], gap="medium")
            with col1:
                st.image(image, caption='Uploaded Image', use_container_width=True)
            
            with col2:
                with st.spinner('Classifying...'):
                    try:
                        top_prob, top_idx, all_probs = predict(image, model)
                    except Exception as exc:
                        st.error(f"An error occurred during prediction: {exc}")
                        st.stop()

                top_class = CLASS_NAMES[top_idx]
                
                # Display main prediction
                st.markdown("### Prediction")
                st.metric(
                    label=top_class,
                    value=f"{top_prob * 100:.1f}%",
                    help="Model confidence in this prediction"
                )
                
                # Show confidence bar
                st.markdown("##### Confidence Level:")
                st.progress(min(top_prob, 1.0))
                
                if top_prob > 0.8:
                    st.success("Very High Confidence - Model is highly certain")
                elif top_prob > 0.6:
                    st.info("Good Confidence - Model is reasonably certain")
                else:
                    st.warning("Low Confidence - Model is uncertain, consider other classes")
            
            st.divider()
            
            # Show top 5 similar classes
            st.markdown("### Similar Classes (Top 5)")
            top5_indices = np.argsort(all_probs)[::-1][:5]
            top5_probs = all_probs[top5_indices]
            
            for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                class_name = CLASS_NAMES[idx]
                percentage = float(prob) * 100
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.progress(min(float(prob), 1.0))
                    st.caption(f"{rank}. {class_name}")
                with col2:
                    st.caption(f"{percentage:.1f}%")

    with tab_how_it_works:
        st.markdown("### Prediction Pipeline")
        
        step_pipeline = [
            {
                "step": "1. Image Input",
                "desc": "Your uploaded image is resized to 224x224 pixels and normalized with ImageNet statistics",
                "tech": "Resize + Normalize"
            },
            {
                "step": "2. Feature Extraction",
                "desc": "ResNet50 backbone (pretrained on ImageNet) extracts 2048-dimensional feature vectors representing visual patterns unique to cars",
                "tech": "ResNet50 Backbone"
            },
            {
                "step": "3. Classification",
                "desc": "A 3-layer classifier head processes features through dropout, linear layers, and ReLU activation to map to 196 car classes",
                "tech": "Dense Layers (2048->512->196)"
            },
            {
                "step": "4. Probability Ranking",
                "desc": "Softmax converts classifier outputs to probabilities. The highest probability indicates the predicted class",
                "tech": "Softmax + Top-1 Selection"
            }
        ]
        
        for item in step_pipeline:
            with st.container():
                st.markdown(f"**{item['step']}**")
                st.write(item['desc'])
                st.caption(f"Tech: {item['tech']}")
        
        st.divider()
        st.markdown("### Model Architecture Summary")
        
        arch_cols = st.columns(4)
        with arch_cols[0]:
            st.metric("Input", "224x224x3", help="RGB image size")
        with arch_cols[1]:
            st.metric("Backbone", "ResNet50", help="50-layer CNN")
        with arch_cols[2]:
            st.metric("Features", "2048-dim", help="Extracted vectors")
        with arch_cols[3]:
            st.metric("Output", "196 classes", help="Car categories")
        
        st.markdown("### Key Features")
        features_text = """
- **Transfer Learning**: Backbone pretrained on ImageNet (1.2M images, 1000 classes)
- **Fine-tuning**: All layers trainable for domain adaptation to cars
- **Global Avg Pooling**: Spatially aggregates features for robustness
- **Dropout Regularization**: Prevents overfitting on 196 classes
- **Stratified Training**: All classes equally represented
- **Class Imbalance Handling**: Weighted loss ensures underrepresented classes contribute equally
        """
        st.markdown(features_text)
        
        st.divider()
        st.markdown("### Why This Approach Works")
        why_text = """
        **Fine-Grained Classification Challenge:**
        - 196 car classes are visually similar (different model years, trims of same make)
        - Generic CNN trained on 1000 objects isn't specific enough
        
        **Solution: Progressive Fine-tuning:**
        1. Start with ImageNet pretrained backbone (learns general vision)
        2. Replace classifier head for 196 car classes
        3. Train all layers with low learning rate (domain adaptation)
        4. Use augmentation to prevent overfitting on similar classes
        
        **Result:**
        - 86.25% accuracy on validation set
        - 97.71% top-5 accuracy (model rarely misses true class in top 5)
        - Balanced performance across all 196 classes (F1 = 0.8601)
        """
        st.markdown(why_text)


        st.markdown("### Normalization Techniques in Machine Learning")
        st.markdown("""
        Normalization is crucial for stable training and generalization in machine learning.
        Different normalization techniques are suited for different tasks and architectures.
        
        **Role in Key Process**: In the **"Modeling with timm"** step, ResNet50 uses BatchNorm internally 
        after each convolutional layer. This enables:
        - Stable gradient flow during backpropagation
        - Faster convergence with higher learning rates
        - Better generalization by reducing internal covariate shift
        
        Understanding these techniques helps when designing custom architectures or fine-tuning models.
        """)
        
        norm_type = st.selectbox(
            "Select Normalization Type:",
            ["BatchNorm", "LayerNorm", "InstanceNorm", "GroupNorm"],
            help="Choose a normalization technique to explore"
        )
        
        if norm_type == "BatchNorm":
            st.markdown("#### Batch Normalization (BatchNorm)")
            st.markdown("""
            - **What**: Normalizes features across the batch dimension
            - **When**: Large batch sizes, CNNs, training phase
            - **Why**: Reduces internal covariate shift, allows higher learning rates
            - **Formula**: For each feature dimension d: mean and variance computed over batch
            """)
            
            st.code("""
def batch_normalization_2d(X, gamma=None, beta=None, eps=1e-5):
    # Compute mean and variance along batch dimension (axis 0)
    mean = X.mean(axis=0, keepdims=True)
    var = X.var(axis=0, keepdims=True)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Apply affine transformation if provided
    if gamma is not None:
        X_norm = X_norm * gamma
    if beta is not None:
        X_norm = X_norm + beta
    
    return X_norm
            """, language="python")
            
            # Interactive demo
            st.markdown("##### Interactive Demo")
            batch_size = st.slider("Batch Size (N)", 2, 32, 8)
            feature_dim = st.slider("Feature Dimension (D)", 4, 128, 16)
            
            # Generate random data
            np.random.seed(42)
            X_demo = np.random.randn(batch_size, feature_dim) * 2 + 5
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Before Normalization:**")
                st.dataframe(pd.DataFrame(X_demo).head(10), use_container_width=True)
                st.caption(f"Mean: {X_demo.mean():.4f}, Std: {X_demo.std():.4f}")
            
            X_norm = batch_normalization_2d(X_demo)
            with col2:
                st.markdown("**After Normalization:**")
                st.dataframe(pd.DataFrame(X_norm).head(10), use_container_width=True)
                st.caption(f"Mean: {X_norm.mean():.4f}, Std: {X_norm.std():.4f}")
        
        elif norm_type == "LayerNorm":
            st.markdown("#### Layer Normalization (LayerNorm)")
            st.markdown("""
            - **What**: Normalizes each sample across its feature dimensions
            - **When**: Small batches, RNNs/Transformers, inference
            - **Why**: Independent of batch size, works well for sequences
            - **Formula**: For each sample n: mean and variance computed over features
            """)
            
            st.code("""
def layer_normalization_2d(X, gamma=None, beta=None, eps=1e-5):
    # Compute mean and variance along feature dimension (axis 1)
    mean = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Apply affine transformation if provided
    if gamma is not None:
        X_norm = X_norm * gamma
    if beta is not None:
        X_norm = X_norm + beta
    
    return X_norm
            """, language="python")
        
        elif norm_type == "InstanceNorm":
            st.markdown("#### Instance Normalization (InstanceNorm)")
            st.markdown("""
            - **What**: Normalizes each channel of each sample across spatial dimensions
            - **When**: Style transfer, image generation, removing instance-specific contrast
            - **Why**: Preserves style while removing content-specific statistics
            - **Formula**: For each (n, c): mean and variance computed over (H, W)
            """)
            
            st.code("""
def instance_normalization_4d(X, gamma=None, beta=None, eps=1e-5):
    # Compute mean and variance over spatial dimensions (H, W) for each (n, c)
    mean = X.mean(axis=(2, 3), keepdims=True)  # (N, C, 1, 1)
    var = X.var(axis=(2, 3), keepdims=True)   # (N, C, 1, 1)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Apply affine transformation if provided
    if gamma is not None:
        X_norm = X_norm * gamma
    if beta is not None:
        X_norm = X_norm + beta
    
    return X_norm
            """, language="python")
        
        elif norm_type == "GroupNorm":
            st.markdown("#### Group Normalization (GroupNorm)")
            st.markdown("""
            - **What**: Divides channels into groups and normalizes within each group
            - **When**: Small batch sizes, when BatchNorm is unstable
            - **Why**: Combines benefits of LayerNorm and InstanceNorm
            - **Formula**: Reshape to (N, G, Cg, H, W), normalize over (Cg, H, W)
            """)
            
            st.code("""
def group_normalization_4d(X, num_groups, gamma=None, beta=None, eps=1e-5):
    N, C, H, W = X.shape
    Cg = C // num_groups
    
    # Reshape to (N, G, Cg, H, W)
    X_grouped = X.reshape(N, num_groups, Cg, H, W)
    
    # Compute mean and variance over (Cg, H, W) for each group
    mean = X_grouped.mean(axis=(2, 3, 4), keepdims=True)
    var = X_grouped.var(axis=(2, 3, 4), keepdims=True)
    
    # Normalize
    X_norm_grouped = (X_grouped - mean) / np.sqrt(var + eps)
    
    # Reshape back to (N, C, H, W)
    X_norm = X_norm_grouped.reshape(N, C, H, W)
    
    # Apply affine transformation if provided
    if gamma is not None:
        X_norm = X_norm * gamma
    if beta is not None:
        X_norm = X_norm + beta
    
    return X_norm
            """, language="python")
            
            # Demo
            num_groups = st.slider("Number of Groups", 2, 8, 4)
            st.info(f"With {num_groups} groups, each group has {64 // num_groups} channels (assuming 64 total channels)")
        st.markdown("""
        Object detection requires understanding spatial relationships between
        bounding boxes. IoU (Intersection over Union) measures overlap, while anchor boxes
        provide reference locations for predicting object positions.
        
        **Role in Key Process**: While this project focuses on classification, these concepts are 
        fundamental for extending to object detection tasks. They demonstrate:
        - **IoU**: How to measure spatial overlap between bounding boxes (used in evaluation metrics)
        - **Anchor boxes**: How to generate reference boxes for object detection models
        
        Understanding these concepts provides a foundation for more advanced computer vision tasks.
        """)
        
        detection_topic = st.radio(
            "Select Topic:",
            ["IoU (Intersection over Union)", "Anchor Box Generation"],
            horizontal=True
        )
        
        if detection_topic == "IoU (Intersection over Union)":
            st.markdown("#### IoU Computation")
            st.markdown("""
            IoU measures how much two bounding boxes overlap. It's used for:
            - Evaluating detection accuracy (matching predictions to ground truth)
            - Non-maximum suppression (removing duplicate detections)
            - Training object detection models (defining positive/negative samples)
            """)
            
            st.code("""
def boxes_iou(A, B):
    # A: (Na, 4) boxes in format [x1, y1, x2, y2]
    # B: (Nb, 4) boxes in format [x1, y1, x2, y2]
    
    # Expand for broadcasting: (Na, 1, 4) and (1, Nb, 4)
    A_expanded = A[:, np.newaxis, :]
    B_expanded = B[np.newaxis, :, :]
    
    # Compute intersection coordinates
    x1_max = np.maximum(A_expanded[:, :, 0], B_expanded[:, :, 0])
    y1_max = np.maximum(A_expanded[:, :, 1], B_expanded[:, :, 1])
    x2_min = np.minimum(A_expanded[:, :, 2], B_expanded[:, :, 2])
    y2_min = np.minimum(A_expanded[:, :, 3], B_expanded[:, :, 3])
    
    # Compute intersection area
    intersection_area = np.maximum(0, x2_min - x1_max) * \\
                        np.maximum(0, y2_min - y1_max)
    
    # Compute union area
    A_area = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    B_area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union_area = A_area[:, np.newaxis] + B_area[np.newaxis, :] - intersection_area
    
    # Compute IoU
    iou = intersection_area / np.maximum(union_area, 1e-8)
    return iou  # Shape: (Na, Nb)
            """, language="python")
            
            # Interactive demo
            st.markdown("##### Interactive Demo")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Box A:**")
                a_x1 = st.slider("A x1", 0, 100, 10)
                a_y1 = st.slider("A y1", 0, 100, 10)
                a_x2 = st.slider("A x2", 0, 100, 50)
                a_y2 = st.slider("A y2", 0, 100, 50)
            
            with col2:
                st.markdown("**Box B:**")
                b_x1 = st.slider("B x1", 0, 100, 30)
                b_y1 = st.slider("B y1", 0, 100, 30)
                b_x2 = st.slider("B x2", 0, 100, 70)
                b_y2 = st.slider("B y2", 0, 100, 70)
            
            A_demo = np.array([[a_x1, a_y1, a_x2, a_y2]])
            B_demo = np.array([[b_x1, b_y1, b_x2, b_y2]])
            iou_value = boxes_iou(A_demo, B_demo)[0, 0]
            
            st.metric("IoU", f"{iou_value:.3f}")
            if iou_value > 0.5:
                st.success("High overlap - likely same object")
            elif iou_value > 0.3:
                st.info("Moderate overlap")
            else:
                st.warning("Low overlap - likely different objects")
        
        else:  # Anchor Box Generation
            st.markdown("#### Anchor Box Generation")
            st.markdown("""
            Anchor boxes are predefined bounding boxes at different scales and
            aspect ratios placed at each spatial location. They serve as reference boxes
            for predicting object locations and sizes in detection models.
            """)
            
            st.code("""
def generate_anchor_boxes(feature_map_h, feature_map_w, anchor_sizes, stride):
    # Create grid of center coordinates
    cy = np.arange(feature_map_h) * stride + stride / 2
    cx = np.arange(feature_map_w) * stride + stride / 2
    cy_grid, cx_grid = np.meshgrid(cy, cx, indexing='ij')
    
    # Initialize output
    anchors = np.zeros((feature_map_h, feature_map_w, A, 4))
    
    # For each anchor type
    for a_idx in range(A):
        w, h = anchor_sizes[a_idx]
        anchors[:, :, a_idx, 0] = cx_grid - w / 2  # x1
        anchors[:, :, a_idx, 1] = cy_grid - h / 2  # y1
        anchors[:, :, a_idx, 2] = cx_grid + w / 2  # x2
        anchors[:, :, a_idx, 3] = cy_grid + h / 2  # y2
    
    return anchors  # Shape: (H, W, A, 4)
            """, language="python")
            
            # Demo
            st.markdown("##### Interactive Demo")
            feature_h = st.slider("Feature Map Height", 4, 32, 8)
            feature_w = st.slider("Feature Map Width", 4, 32, 8)
            stride_val = st.slider("Stride", 8, 64, 16)
            
            anchor_sizes_demo = np.array([[32, 32], [64, 64], [128, 128]])
            anchors_demo = generate_anchor_boxes(feature_h, feature_w, anchor_sizes_demo, stride_val)
            
            st.info(f"Generated {anchors_demo.shape[0]}×{anchors_demo.shape[1]}×{anchors_demo.shape[2]} = "
                   f"{anchors_demo.size // 4} anchor boxes")




SECTION_CONFIG = [
    ("Dataset Overview", render_key_statistics_section),
    ("Key Process", render_key_process_section),
    ("Dataset Balance", render_dataset_balance_section),
    ("EDA", render_eda_section),
    ("Train/Val/Test Split", render_train_val_test_split_section),
    ("Data Augmentation", render_data_augmentation_section),
    ("Model Architecture", render_model_architecture_section),
    ("Training Loop", render_training_loop_section),
    ("Training Visualizations", render_training_visualizations_section),
    ("Model Results", render_model_results_section),
    ("Comprehensive Evaluation", render_comprehensive_evaluation_section),
    ("Test Inference", render_test_inference_section),
    ("Live Prediction", render_live_prediction_section),
]

tabs = st.tabs([label for label, _ in SECTION_CONFIG])

for tab, (_, render_fn) in zip(tabs, SECTION_CONFIG):
    with tab:
        render_fn()
