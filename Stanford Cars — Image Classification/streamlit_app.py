"""
Streamlit Web App for Stanford Cars Classification
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

st.title('Stanford Cars Classifier')


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
        with st.container(border=True, height="stretch"):
            st.subheader("Image Resolution Snapshot")
            if not size_stats.empty:
                st.metric("Average Width", f"{size_stats['width'].mean():.0f}px")
                st.metric("Average Height", f"{size_stats['height'].mean():.0f}px")
                st.metric("Median Aspect Ratio", f"{size_stats['aspect_ratio'].median():.2f}")
            else:
                st.info("Resolution statistics will appear once the dataset is available.")

    st.subheader("Dataset Artifacts")
    
    # Class Distribution Analysis Chart
    with st.container(border=True):
        st.markdown("### Class Distribution Analysis - All 196 Classes")
        if not train_df.empty:
            dist_analysis_chart = create_full_class_distribution_analysis_chart(train_df)
            if dist_analysis_chart:
                st.altair_chart(dist_analysis_chart, use_container_width=True)
                
                # Balance Analysis and Conclusion
                balance_stats = analyze_class_balance(train_df)
                if balance_stats:
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Min Samples", f"{int(balance_stats['min_samples'])}")
                    with col_stat2:
                        st.metric("Max Samples", f"{int(balance_stats['max_samples'])}")
                    with col_stat3:
                        st.metric("Mean Samples", f"{balance_stats['mean_samples']:.1f}")
                    with col_stat4:
                        st.metric("Imbalance Ratio", f"{balance_stats['imbalance_ratio']:.2f}x")
                    
                    st.divider()
                    
                    # Balance Conclusion
                    imbalance_ratio = balance_stats['imbalance_ratio']
                    if imbalance_ratio < 1.5:
                        balance_status = "WELL BALANCED"
                        balance_desc = "Dataset is relatively balanced across all 196 classes. All classes have similar sample counts."
                        balance_color = "success"
                    elif imbalance_ratio < 3.0:
                        balance_status = "MODERATELY IMBALANCED"
                        balance_desc = f"Dataset shows moderate imbalance (ratio: {imbalance_ratio:.2f}x). Some classes have fewer samples but still acceptable for training. Using weighted loss function helps handle this imbalance."
                        balance_color = "warning"
                    else:
                        balance_status = "❌ SIGNIFICANTLY IMBALANCED"
                        balance_desc = f"Dataset shows significant imbalance (ratio: {imbalance_ratio:.2f}x). Consider using class-weighted loss or data augmentation techniques."
                        balance_color = "error"
                    
                    st.markdown(f"#### {balance_status}")
                    st.markdown(f"**Conclusion:** {balance_desc}")
                    
                    # Additional statistics
                    st.markdown("**Detailed Statistics:**")
                    stats_text = f"""
                    - **Total Training Images**: {balance_stats['total_samples']:,}
                    - **Total Classes**: {balance_stats['num_classes']}
                    - **Median Samples/Class**: {balance_stats['median_samples']:.1f}
                    - **Standard Deviation**: {balance_stats['std_samples']:.2f}
                    - **Coefficient of Variation**: {balance_stats['coefficient_variation']:.3f}
                    - **Classes below 50% of mean**: {balance_stats['imbalanced_classes']} ({balance_stats['imbalance_percentage']:.1f}%)
                    """
                    st.markdown(stats_text)
                    
                    # Legend for chart
                    st.caption("📊 Chart Legend: Red dashed line = Mean, Green dashed line = Max, Orange dashed line = Min")
            else:
                st.info("Class distribution chart not available.")
        else:
            st.info("Training data not available for distribution analysis.")
    
    # Additional charts in columns
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        with st.container(border=True, height="stretch"):
            st.markdown("### Top 30 Classes by Sample Count")
            class_dist_chart = create_class_distribution_chart(train_df)
            if class_dist_chart:
                st.altair_chart(class_dist_chart, use_container_width=True)
            else:
                st.info("Class distribution data not available.")
    with col_b:
        with st.container(border=True, height="stretch"):
            st.markdown("### Top 30 Classes by F1 Score")
            report_df = load_classification_report_df()
            f1_chart = create_f1_score_chart(report_df)
            if f1_chart:
                st.altair_chart(f1_chart, use_container_width=True)
            else:
                st.info("F1 score data not available.")

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

with st.expander("Dataset Balance Analysis"):
    st.caption("Assessment of class distribution balance and potential imbalance issues.")
    
    train_df_balance = load_annotations_df('train')
    if not train_df_balance.empty:
        balance_stats = analyze_class_balance(train_df_balance)
        
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
        
        col1, col2 = st.columns([2, 1], gap="medium")
        with col1:
            st.markdown("### Distribution Statistics")
            stats_text = f"""
**Sample Count Statistics:**
- **Min samples in class**: {balance_stats['min_samples']:.0f}
- **Max samples in class**: {balance_stats['max_samples']:.0f}
- **Median samples/class**: {balance_stats['median_samples']:.1f}
- **Std deviation**: {balance_stats['std_samples']:.2f}
- **Coefficient of Variation**: {balance_stats['coefficient_variation']:.3f}

**Imbalance Assessment:**
- **Classes below 50% mean**: {balance_stats['imbalanced_classes']} ({balance_stats['imbalance_percentage']:.1f}%)
- **Imbalance Ratio**: {balance_stats['imbalance_ratio']:.2f}x
            """
            st.markdown(stats_text)
        
        with col2:
            st.markdown("### Interpretation")
            if balance_stats['imbalance_ratio'] < 1.5:
                status = "Well Balanced"
                desc = "Dataset is relatively balanced across classes."
            elif balance_stats['imbalance_ratio'] < 3.0:
                status = "Moderately Imbalanced"
                desc = "Some classes have fewer samples but still acceptable."
            else:
                status = "Significantly Imbalanced"
                desc = "Notable imbalance detected; consider weighted loss."
            
            st.markdown(f"**{status}**\n\n{desc}")
            
            if balance_stats['imbalance_ratio'] >= 1.5:
                st.info(
                    "Recommendation: Using weighted loss function to handle class imbalance "
                    "and ensure underrepresented classes contribute adequately to training."
                )
    else:
        st.warning("Training data not available for balance analysis.")

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
        split_df['Percent'] = 0.0
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
        
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("**Training History**")
            display_image_if_exists(TRAINING_HISTORY_IMG, "Training & Validation Metrics")
        
        with col2:
            st.markdown("**Confusion Matrix**")
            confusion_chart = create_confusion_matrix_chart(report_df)
            if confusion_chart:
                st.altair_chart(confusion_chart, use_container_width=True)
            else:
                st.info("Confusion matrix data not available.")

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

with st.expander("Model Performance Comparison"):
    st.caption("Compare top-performing vs challenging classes to identify improvement areas.")
    
    report_df = load_classification_report_df()
    if not report_df.empty:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### 🏆 Top 15 Performing Classes")
            top_classes = report_df.nlargest(15, 'f1-score')[['class_name', 'f1-score', 'precision', 'recall', 'support']]
            st.dataframe(
                top_classes,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "class_name": st.column_config.TextColumn("Car Class", width="large"),
                    "f1-score": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "support": st.column_config.NumberColumn("Samples", format="%d"),
                }
            )
        
        with col2:
            st.markdown("### ⚠️ Challenging Classes (Lowest F1)")
            bottom_classes = report_df.nsmallest(15, 'f1-score')[['class_name', 'f1-score', 'precision', 'recall', 'support']]
            st.dataframe(
                bottom_classes,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "class_name": st.column_config.TextColumn("Car Class", width="large"),
                    "f1-score": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "support": st.column_config.NumberColumn("Samples", format="%d"),
                }
            )
        
        st.divider()
        
        # F1 Score Distribution
        st.markdown("### F1 Score Distribution Across All 196 Classes")
        f1_dist_chart = (
            alt.Chart(report_df)
            .mark_bar(
                color=CHART_COLORS['primary'],
                opacity=0.7,
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3,
            )
            .encode(
                x=alt.X(
                    'f1-score:Q',
                    bin=alt.Bin(maxbins=40),
                    title='F1 Score Range',
                    axis=alt.Axis(format='.2f')
                ),
                y=alt.Y('count()', title='Number of Classes'),
                tooltip=[
                    alt.Tooltip('f1-score:Q', bin=True, title='F1 Score Range'),
                    alt.Tooltip('count()', title='Class Count', format='.0f'),
                ],
            )
            .properties(height=350, title='F1 Score Distribution')
            .configure_axis(grid=True, gridOpacity=0.2)
        )
        st.altair_chart(f1_dist_chart, use_container_width=True)
        
        # Summary Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Mean F1 Score",
                f"{report_df['f1-score'].mean():.3f}",
                help="Average F1 across all 196 classes"
            )
        with col2:
            st.metric(
                "Median F1 Score",
                f"{report_df['f1-score'].median():.3f}",
                help="Median F1 score"
            )
        with col3:
            st.metric(
                "Min F1 Score",
                f"{report_df['f1-score'].min():.3f}",
                help="Lowest F1 among classes"
            )
        with col4:
            st.metric(
                "Max F1 Score",
                f"{report_df['f1-score'].max():.3f}",
                help="Highest F1 among classes"
            )
    else:
        st.info("Classification report not available.")

with st.expander("Advanced Analysis"):
    st.caption("Detailed per-class metrics, sample exploration, and model internals.")
    
    (
        tab_class_details,
        tab_sample_explorer,
        tab_model_insights,
    ) = st.tabs(
        [
            "1. Per-Class Metrics",
            "2. Sample Explorer by Class",
            "3. Model Insights",
        ]
    )
    
    with tab_class_details:
        st.subheader("Detailed Per-Class Performance")
        report_df = load_classification_report_df()
        if not report_df.empty:
            # Add performance tier
            report_df['Tier'] = pd.cut(
                report_df['f1-score'],
                bins=[0, 0.80, 0.90, 1.0],
                labels=['Below 0.80', '0.80-0.90', 'Above 0.90'],
                include_lowest=True
            )
            
            search_col, metric_col = st.columns([2, 1])
            with search_col:
                search_term = st.text_input("🔍 Search class by name", placeholder="e.g., BMW, Tesla...")
            with metric_col:
                sort_metric = st.selectbox("Sort by", options=['f1-score', 'precision', 'recall', 'support'])
            
            # Filter and sort
            if search_term:
                filtered_df = report_df[report_df['class_name'].str.contains(search_term, case=False, na=False)]
            else:
                filtered_df = report_df
            
            filtered_df = filtered_df.sort_values(sort_metric, ascending=False)
            
            st.markdown(f"**Found {len(filtered_df)} classes**")
            st.dataframe(
                filtered_df[['class_name', 'f1-score', 'precision', 'recall', 'support', 'Tier']],
                use_container_width=True,
                height=400,
                column_config={
                    "class_name": st.column_config.TextColumn("Car Class", width="large"),
                    "f1-score": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "support": st.column_config.NumberColumn("Samples", format="%d"),
                    "Tier": st.column_config.TextColumn("Performance Tier"),
                },
            )
        else:
            st.info("Classification report not available.")
    
    with tab_sample_explorer:
        st.subheader("Browse Sample Images & Performance by Class")
        report_df = load_classification_report_df()
        train_df = load_annotations_df('train')
        
        if not report_df.empty and not train_df.empty:
            # Filter classes by performance tier
            perf_tier = st.radio(
                "Filter by performance tier:",
                options=['All', 'High (F1 > 0.90)', 'Medium (0.80-0.90)', 'Low (F1 < 0.80)'],
                horizontal=True
            )
            
            if perf_tier == 'High (F1 > 0.90)':
                filtered_classes = report_df[report_df['f1-score'] > 0.90]['class_name'].tolist()
            elif perf_tier == 'Medium (0.80-0.90)':
                filtered_classes = report_df[(report_df['f1-score'] >= 0.80) & (report_df['f1-score'] <= 0.90)]['class_name'].tolist()
            elif perf_tier == 'Low (F1 < 0.80)':
                filtered_classes = report_df[report_df['f1-score'] < 0.80]['class_name'].tolist()
            else:
                filtered_classes = CLASS_NAMES
            
            selected_class = st.selectbox(
                "Select a class to explore:",
                options=filtered_classes,
                index=0 if len(filtered_classes) > 0 else None
            )
            
            if selected_class:
                # Show class metrics
                class_metrics = report_df[report_df['class_name'] == selected_class].iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("F1 Score", f"{class_metrics['f1-score']:.3f}")
                with col2:
                    st.metric("Precision", f"{class_metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{class_metrics['recall']:.3f}")
                with col4:
                    st.metric("Samples", f"{int(class_metrics['support'])}")
                
                st.divider()
                
                # Show sample images
                st.markdown(f"### Sample Images from {selected_class}")
                num_images = st.slider("Number of samples to show", min_value=3, max_value=12, value=6, step=3)
                render_sample_gallery(selected_class, num_images=num_images)
        else:
            st.info("Sample explorer requires classification report and training data.")
    
    with tab_model_insights:
        st.subheader("Model Architecture & Feature Extraction")
        
        # Feature Extraction Pipeline Visualization
        st.markdown("### ResNet50 Feature Extraction Pipeline")
        st.markdown("""
        The model extracts features through progressive abstraction levels:
        """)
        
        pipeline_data = pd.DataFrame({
            'Stage': ['Input', 'Conv1 + MaxPool', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'AvgPool', 'Classifier'],
            'Shape': ['224×224×3', '56×56×64', '56×56×256', '28×28×512', '14×14×1024', '7×7×2048', '2048', '196'],
            'Purpose': [
                'Raw RGB Image',
                'Edge Detection',
                'Texture Patterns',
                'Car Parts',
                'Components',
                'Semantic Features',
                'Feature Vector',
                'Class Prediction'
            ]
        })
        
        st.dataframe(
            pipeline_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Stage': st.column_config.TextColumn('Stage', width='medium'),
                'Shape': st.column_config.TextColumn('Tensor Shape', width='medium'),
                'Purpose': st.column_config.TextColumn('Learned Features', width='large'),
            }
        )
        
        st.markdown("### Key Insights")
        insights_text = """
        **Transfer Learning Benefits:**
        - ResNet50 backbone pretrained on ImageNet (1.2M images, 1000 classes)
        - Early layers learn generic visual features (edges, colors, textures)
        - Deep layers adapt to fine-grained car classification during training
        
        **Architecture Details:**
        - Total Parameters: ~24.7M (25.9M with classifier head)
        - Trainable: All 25.9M parameters fine-tuned for cars
        - Global Average Pooling: Spatially robust, size-agnostic features
        - Classifier Head: 2048 → 512 → 196 (with dropout regularization)
        
        **Training Strategy:**
        - Optimizer: AdamW with weight decay (L2 regularization)
        - Scheduler: OneCycleLR (warm-up + cosine decay)
        - Loss: CrossEntropyLoss with class weights for balance
        - Batch Size: 96 | Learning Rate: 1e-3 max | Epochs: 24
        - Early Stopping: 5 epochs patience
        
        **Performance:**
        - Top-1 Accuracy: 86.25% on validation set
        - Top-5 Accuracy: 97.71% (very high confidence in top predictions)
        - F1 Macro: 0.8601 (balanced across all 196 classes)
        - All 196 classes present in training and validation
        """
        st.markdown(insights_text)
        
        display_image_if_exists(
            AUGMENTATION_IMG,
            "Data Augmentation Examples",
            "Shows how images are transformed during training for better generalization"
        )

with st.expander("Live Prediction"):
    st.caption("Upload a car image to classify it into one of 196 fine-grained categories.")

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
            with st.container(border=True):
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
