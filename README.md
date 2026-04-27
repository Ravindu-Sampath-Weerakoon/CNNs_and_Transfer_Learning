# Tea Leaf Disease Classification using CNNs and Ensemble Methods

This project focuses on identifying and classifying common tea leaf diseases using deep learning techniques. We explore various Convolutional Neural Network (CNN) architectures, including VGG16 and ResNet50, and employ advanced ensemble methods (Hybrid and Stacking) to achieve high accuracy in disease detection.

## 🌿 Project Overview
Timely and accurate detection of tea leaf diseases is critical for effective crop management and reducing yield losses. This project implements a robust pipeline for:
- **Preprocessing**: Custom white-padded centered cropping to focus on leaf regions.
- **Transfer Learning**: Fine-tuning pre-trained VGG16 and ResNet50 models.
- **Ensemble Learning**: Combining models through intelligent weighting (Hybrid) and meta-learning (Stacking).

---

## 📁 Project Structure
```text
├── main.ipynb                  # Core Jupyter notebook for training and evaluation
├── configs/                    # Configuration files for ensembles and weights
│   ├── hybrid_ensemble_config.json
│   └── hybrid_weights.json
├── data/                       # Dataset organized by class labels
│   ├── algal leaf/             # Algal Leaf spot images
│   ├── brown blight/           # Brown Blight images
│   └── white spot/             # White Spot images
├── logs/                       # Training history and performance logs (JSON)
├── models/                     # Saved model checkpoints (.pth)
└── outputs/                    # Evaluation reports, confusion matrices, and plots
```

---

## 🛠️ Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- `torch` (PyTorch)
- `torchvision`
- `Pillow` (PIL)
- `numpy`
- `opencv-python` (cv2)
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Usage
1.  **Clone the repository** and ensure the `data/` directory contains the leaf images.
2.  **Open `main.ipynb`** in Google Colab or a local Jupyter environment.
3.  **Adjust Paths**: If running locally, update the `PROJECT_BASE_PATH` in the notebook from `/content/drive/...` to your local path.
4.  **Run the notebook** to reproduce the training and evaluation results.

---

## 📊 Dataset & Preprocessing
The dataset consists of 368 images across three classes: `algal leaf`, `brown blight`, and `white spot`.

### Data Pipeline
- **Splitting**: Stratified Shuffle Split (60% Train, 20% Val, 20% Test).
- **Custom Preprocessing**:
  - `get_white_padded_centered_crop`: Uses OpenCV to isolate the leaf from the background, centers it, and adds white padding to maintain aspect ratio during resizing.
  - **Augmentation**: Horizontal flips, rotations, and color jittering.
  - **Normalization**: ImageNet standard (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

---

## 🚀 Model Performance

### Individual Models
| Model | Test Accuracy | Test Loss |
| :--- | :---: | :---: |
| **VGG16** | 78.38% | 0.4818 |
| **ResNet50** | 82.43% | 0.6202 |

### Ensemble Methods
| Method | Test Accuracy | Test Loss |
| :--- | :---: | :---: |
| **Hybrid Ensemble** | **86.49%** | 0.7794 |
| **Stacking (NN Meta-Learner)** | 78.38% | 0.7899 |
| **Logistic Stacking** | 75.68% | 0.6092 |

> **Note**: The **Hybrid Ensemble** performed best by combining VGG16 and ResNet50 using class-wise intelligent weights derived from their respective F1-scores.

---

## 📈 Evaluation Results
You can find detailed classification reports and visualizations in the `outputs/` folder:
- **Confusion Matrices**: Visual representation of model performance per class.
- **Training Curves**: Loss and Accuracy plots for VGG16 and ResNet50.
- **Comparison Summary**: `final_model_comparison.json`.

---

## 🔮 Inference
To use the trained ResNet50 model for new predictions, refer to the documentation in `models/README.md`. It provides a complete code snippet for loading `resnet50_best.pth` and performing inference on a single image.

---

## 📝 Conclusion
The Hybrid Ensemble approach demonstrates that combining complementary CNN architectures significantly improves disease classification accuracy. While ResNet50 provides a strong baseline (82.4%), the Hybrid Ensemble achieves a peak performance of **86.5%**.
