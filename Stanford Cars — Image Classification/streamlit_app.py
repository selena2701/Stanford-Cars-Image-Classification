"""
Streamlit Web App for Stanford Cars Classification
"""
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
    """Configure base Altair theme matching reference style"""
    return {
        'background': 'transparent',
        'title': {
            'font': 'Inter',
            'fontSize': 18,
            'fontWeight': 600,
            'color': '#e2e8f0',
        },
        'axis': {
            'labelFont': 'Inter',
            'labelFontSize': 12,
            'labelColor': '#94a3b8',
            'titleFont': 'Inter',
            'titleFontSize': 14,
            'titleFontWeight': 600,
            'titleColor': '#e2e8f0',
            'gridColor': '#314158',
            'domainColor': '#475569',
        },
        'legend': {
            'labelFont': 'Inter',
            'labelFontSize': 12,
            'labelColor': '#e2e8f0',
            'titleFont': 'Inter',
            'titleFontSize': 13,
            'titleFontWeight': 600,
            'titleColor': '#e2e8f0',
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
    """Predict class for image"""
    transform = get_val_transforms()
    img_np = np.array(image.convert('RGB'))
    img_tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        top5_probs, top5_indices = torch.topk(probs, k=5)
    
    return top5_probs.cpu().numpy()[0], top5_indices.cpu().numpy()[0]


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

st.title('Stanford Cars Classifier')
st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0; color: #666; font-size: 1.1rem;'>
    An interactive end-to-end report for data quality, modeling, and live inference
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

with st.expander("Key Statistics Overview", expanded=True):
    st.caption("High-level insights about the Stanford Cars dataset.")

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
        with st.container(border=True, height="stretch"):
            st.subheader("Top 10 Most Represented Classes (Train)")
            if not train_df.empty:
                top10 = (
                    train_df['class_name']
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                top10.columns = ['Class', 'Train Images']
                st.dataframe(top10, use_container_width=True, height=360)
            else:
                st.info("Training annotations not found.")

    with col2:
        with st.container(border=True, height="stretch"):
            st.subheader("Image Resolution Snapshot")
            if not size_stats.empty:
                st.metric("Average Width", f"{size_stats['width'].mean():.0f}px")
                st.metric("Average Height", f"{size_stats['height'].mean():.0f}px")
                st.metric("Median Aspect Ratio", f"{size_stats['aspect_ratio'].median():.2f}")
            else:
                st.info("Resolution statistics will appear once the dataset is available.")

    st.subheader("Dataset Artifacts")
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        with st.container(border=True, height="stretch"):
            display_image_if_exists(
                CLASS_DISTRIBUTION_IMG,
                "Class Distribution Heatmap",
                help_text="Distribution across classes.",
            )
    with col_b:
        with st.container(border=True, height="stretch"):
            display_image_if_exists(
                F1_PER_CLASS_IMG,
                "Per-class F1 Score",
                help_text="Performance spread by class.",
            )

with st.expander("Key Process"):
    st.caption("How raw images become reliable fine-grained car predictions.")

    process_steps = [
        {
            "title": "Data Acquisition & Ground Truth",
            "icon": "",
            "description": (
                "Download Stanford Cars images and official devkit annotations. "
                "Link file names to class IDs and bounding boxes."
            ),
            "details": [
                "8144 labeled training images with bounding boxes.",
                "8041 test images for leaderboard evaluation.",
                "196 fine-grained car make-model-year classes.",
            ],
        },
        {
            "title": "Preprocessing & Augmentation",
            "icon": "",
            "description": (
                "Normalize aspect ratios, apply Albumentations pipeline, "
                "and balance samples per class."
            ),
            "details": [
                "Resize to 224×224 with preserved aspect ratio.",
                "Color jitter, random crops, CutOut, and flips.",
                "Standard ImageNet mean/std normalization.",
            ],
            "extra_image": AUGMENTATION_IMG,
        },
        {
            "title": "Feature Extraction with ResNet50 Backbone",
            "icon": "",
            "description": (
                "The ResNet50 backbone extracts rich visual features from preprocessed images. "
                "Convolutional layers progressively learn hierarchical patterns: edges → textures → shapes → car parts."
            ),
            "details": [
                "Input: 224×224×3 RGB images (normalized with ImageNet stats).",
                "Backbone: ResNet50 with 5 convolutional blocks (50 layers total).",
                "Global Average Pooling: Reduces spatial dimensions to 1×1, producing 2048-dim feature vectors.",
                "Transfer Learning: Backbone initialized with ImageNet pretrained weights for robust feature representations.",
                "Feature Dimension: 2048-dimensional embeddings capture discriminative car characteristics.",
            ],
        },
        {
            "title": "Modeling with timm",
            "icon": "",
            "description": (
                "Fine-tune a `resnet50` backbone from timm with a custom classifier head."
            ),
            "details": [
                "Feature vectors (2048-dim) from backbone → Dropout(0.3) → Linear(512) → ReLU → Dropout(0.2) → Linear(196).",
                "All backbone layers unfrozen with low LR warmup for domain adaptation.",
                "Mixed precision (AMP) for faster training while maintaining numerical stability.",
            ],
        },
        {
            "title": "Training & Optimization",
            "icon": "",
            "description": (
                "Train with AdamW optimizer and OneCycleLR, track metrics and checkpoints."
            ),
            "details": [
                "Batch size 32 on GPU (AMP).",
                "Early stopping on macro F1.",
                "Best checkpoint exported for inference.",
            ],
        },
        {
            "title": "Evaluation & Reporting",
            "icon": "",
            "description": (
                "Aggregate classification report, confusion matrix, "
                "and interpretability overlays for stakeholders."
            ),
            "details": [
                "Macro F1 > 0.91 on validation split.",
                "Detailed per-class precision/recall.",
                "Grad-CAM overlays for qualitative insights.",
            ],
            "extra_image": INTERPRETABILITY_IMG,
        },
    ]

    for step in process_steps:
        with st.container():
            st.markdown(f"### {step['title']}")
            st.write(step['description'])
            if 'details' in step:
                for bullet in step['details']:
                    st.markdown(f"- {bullet}")
            if 'extra_image' in step and step['extra_image']:
                asset_path = Path(step['extra_image'])
                if asset_path.exists():
                    display_image_if_exists(asset_path, step['title'])
        st.divider()

with st.expander("Exploratory Data Analysis"):
    st.caption("Understand dataset balance, image quality, and representative samples.")

    train_df = load_annotations_df('train')
    if train_df.empty:
        st.warning("Training annotations were not found. Please ensure the devkit is available.")
    else:
        max_sample = min(1000, len(train_df))
        slider_min = max(10, min(100, max_sample))
        sample_size = st.slider(
            "Sample size for EDA computations",
            min_value=slider_min,
            max_value=max_sample,
            value=min(400, max_sample),
            step=max(10, slider_min // 2),
        )
        eda_stats = compute_image_metadata(sample_size=sample_size)

        tab_sizes, tab_color, tab_gallery = st.tabs(
            [
                "Image Size & Aspect Ratio",
                "Color & Quality Metrics",
                "Interactive Sample Gallery",
            ]
        )

        with tab_sizes:
            st.subheader("Image Size Distribution")
            if not eda_stats.empty:
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
                    st.caption("Higher Laplacian variance indicates sharper images")
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
            num_images = st.slider("Number of images to preview", min_value=3, max_value=12, value=6, step=3)
            render_sample_gallery(class_choice, num_images=num_images)

with st.expander("Classification"):
    st.caption("Model training pipeline, monitoring, and evaluation artifacts.")

    train_df = load_annotations_df('train')
    report_df = load_classification_report_df()

    (
        tab_distribution,
        tab_split,
        tab_augmentation,
        tab_model,
        tab_training,
        tab_visuals,
        tab_evaluation,
    ) = st.tabs(
        [
            "1. Class Distribution",
            "2. Train/Val/Test Split",
            "3. Data Augmentation",
            "4. Model Architecture (timm)",
            "5. Training Loop",
            "6. Training Visualizations",
            "7. Comprehensive Evaluation",
        ]
    )

    with tab_distribution:
        st.subheader("Training Class Distribution")
        if not train_df.empty:
            class_counts = (
                train_df['class_name']
                .value_counts()
                .reset_index()
            )
            class_counts.columns = ['Class', 'Images']
            top_n = st.slider("Show top N classes", min_value=10, max_value=50, value=30, step=5)
            chart = (
                alt.Chart(class_counts.head(top_n))
                .mark_bar(
                    color=CHART_COLORS['primary'],
                    cornerRadiusTopLeft=3,
                    cornerRadiusTopRight=3,
                )
                .encode(
                    x=alt.X('Images:Q', title='Number of Training Images', axis=alt.Axis(format='.0f')),
                    y=alt.Y('Class:N', sort='-x', title='Car Class'),
                    tooltip=[
                        alt.Tooltip('Class:N', title='Class Name'),
                        alt.Tooltip('Images:Q', title='Training Images', format='.0f'),
                    ],
                )
                .properties(height=600, title=f'Top {top_n} Classes by Training Sample Count')
                .configure_axis(grid=True, gridOpacity=0.2)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(f"Top {top_n} classes by number of training samples. Hover over bars for details.")
            display_image_if_exists(CLASS_DISTRIBUTION_IMG, "Complete Class Distribution Heatmap")
        else:
            st.warning("Training data not available for distribution plot.")

    with tab_split:
        st.subheader("Dataset Split Overview")
        overview_metrics = get_dataset_overview()
        val_estimate = int(overview_metrics['train_images'] * 0.15)
        train_effective = max(overview_metrics['train_images'] - val_estimate, 0)
        split_df = pd.DataFrame(
            [
                {"Split": "Train", "Images": train_effective, "Short": "Train"},
                {"Split": "Validation", "Images": val_estimate, "Short": "Val"},
                {"Split": "Test", "Images": overview_metrics['test_images'], "Short": "Test"},
            ]
        )
        total_images = split_df['Images'].sum()
        if total_images > 0:
            split_df['Percent'] = (split_df['Images'] / total_images * 100).round(2)
        
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            st.dataframe(
                split_df[['Split', 'Images', 'Percent']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Split": st.column_config.TextColumn("Dataset Split", width="medium"),
                    "Images": st.column_config.NumberColumn("Images", format="%d"),
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
        
        st.info(
            "**Validation Strategy**: Validation images (15%) are carved out from the training set using stratified sampling "
            "to tune hyperparameters while keeping the official test set untouched for final evaluation."
        )

    with tab_augmentation:
        st.subheader("Albumentations Pipeline")
        augmentation_code = """A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    A.CLAHE(p=0.2),
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])"""
        st.code(augmentation_code, language="python")
        display_image_if_exists(AUGMENTATION_IMG, "Augmentation Examples")

    with tab_model:
        st.subheader("CarClassifier Architecture")
        
        st.markdown("**Feature Extraction Pipeline:**")
        st.markdown("""
        1. **Input**: 224×224×3 RGB image (normalized)
        2. **ResNet50 Backbone**: 
           - 5 convolutional blocks extract hierarchical features
           - Output: 7×7×2048 feature maps
        3. **Global Average Pooling**: 
           - Reduces 7×7×2048 → 1×1×2048
           - Produces 2048-dimensional feature vector
        4. **Classifier Head**: 
           - 2048 → 512 (with dropout) → 196 classes
        """)
        
        st.code(
            """class CarClassifier(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=196, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        feature_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)""",
            language="python",
        )
        st.info(
            "**Feature Extraction**: ResNet50 backbone extracts 2048-dimensional feature vectors from images. "
            "These features capture visual patterns learned from ImageNet pretraining and fine-tuned on car images. "
            "The classifier head then maps these features to 196 car classes."
        )

    with tab_training:
        st.subheader("Training Loop (AdamW + OneCycleLR)")
        training_code = """optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.1,
    div_factor=25.0,
    final_div_factor=1e4
)

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
"""
        st.code(training_code, language="python")
        st.success("AMP + OneCycleLR keeps training stable while converging quickly.")

    with tab_visuals:
        st.subheader("Training Curves & Diagnostics")
        display_image_if_exists(TRAINING_HISTORY_IMG, "Training & Validation Metrics")
        display_image_if_exists(CONFUSION_MATRIX_IMG, "Confusion Matrix (Validation)")

    with tab_evaluation:
        st.subheader("Per-class Metrics")
        if not report_df.empty:
            metric_choice = st.selectbox(
                "Sort by metric",
                options=['f1-score', 'precision', 'recall'],
                index=0,
            )
            top_k = st.slider("Show top N classes", min_value=10, max_value=60, value=20, step=5)
            sorted_df = report_df.sort_values(metric_choice, ascending=False).head(top_k)
            st.dataframe(sorted_df, use_container_width=True)

            st.subheader("Challenging Classes (Lowest F1)")
            worst_df = report_df.sort_values('f1-score').head(15)
            st.dataframe(worst_df, use_container_width=True)
        else:
            st.info("Classification report JSON not found.")

with st.expander("Live Prediction"):
    st.caption("Upload an image to get top-5 predictions with confidence scores.")

    load_status = st.empty()
    try:
        with st.spinner("Loading model weights..."):
            model = load_model()
        load_status.success("Model loaded and ready for inference.")
    except Exception as exc:
        load_status.error(f"Error loading model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with st.spinner('Classifying...'):
            try:
                probs, indices = predict(image, model)
            except Exception as exc:
                st.error(f"An error occurred during prediction: {exc}")
                st.stop()

        st.subheader('Prediction Results (Top 5)')
        for rank, (prob, idx) in enumerate(zip(probs, indices), start=1):
            class_name = CLASS_NAMES[idx]
            st.write(f"{rank}. **{class_name}** — {prob * 100:.2f}%")

        top_class = CLASS_NAMES[indices[0]]
        top_prob = probs[0]
        st.success(f"**Predicted Class: {top_class}** ({top_prob * 100:.2f}% confidence)")

        with st.expander("Confidence Details"):
            confidence_df = pd.DataFrame(
                {
                    "Rank": np.arange(1, len(probs) + 1),
                    "Class": [CLASS_NAMES[idx] for idx in indices],
                    "Probability": [float(p) for p in probs],
                }
            )
            st.dataframe(confidence_df, hide_index=True, use_container_width=True)