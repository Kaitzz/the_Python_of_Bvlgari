# House of Py: Bvlgari Divas' Dream Pendant Authentication System

[![Python](https://img.shields.io/badge/Python-3.8%2B-lightblue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-pink)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/)

## 💎 Project Overview

A deep learning-based luxury goods authentication system specifically designed for Bulgari Divas' Dream mother-of-pearl pendants. The system generates training data from 3D models and uses deep learning techniques to achieve high-precision authenticity detection.

### ✨ Key Features

- 🎨 **3D Model Rendering**: Generate high-quality training data from official GLB models
- 🤖 **Deep Learning**: Dual-branch network architecture combining visual and geometric features
- 📸 **Angle Robustness**: Accurately identify pendants from various shooting angles
- 🎯 **High Accuracy**: Expected accuracy of 85-95%

### 🏆 Project Highlights

1. **Innovative Data Generation**: No need for large collections of authentic photos - automatically generated through 3D rendering
2. **Perfect Camera Calibration**: Optimal rendering parameters already determined
3. **Practical Value**: Can replace or assist online luxury goods authenticators

## 🚀 Quick Start

### System Requirements

- Python 3.8+
- CUDA 11.0+ (recommended, CPU also works)
- 8GB+ RAM
- 10GB+ disk space

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd bulgari_project
```

2. **Create virtual environment**
```bash
conda create -n bulgari python=3.8
conda activate bulgari
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

Create `requirements.txt`:
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
trimesh>=3.20.0
pyrender>=0.1.45
opencv-python>=4.7.0
Pillow>=9.4.0
scikit-learn>=1.2.0
tqdm>=4.65.0
matplotlib>=3.6.0
pyglet>=2.0.0
pyopengl>=3.1.0
```

### System Dependencies (Linux)

```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglu1-mesa libosmesa6-dev
```

### Windows Environment Setup

```powershell
# If encountering OpenGL issues
$env:PYOPENGL_PLATFORM = "windows"
```

## 📐 Camera Calibration Parameters (Completed)

After precise calibration, we've found the perfect camera parameters:

```json
{
  "distance": 3.0,
  "azimuth": 0,
  "elevation": 5,
  "center_offset": [0, -1.0, 0],
  "fov": 60
}
```

These parameters ensure:
-  Perfect frontal view
-  Slight overhead angle (5°)
-  Centered pendant display

## ✅ Usage Guide

### 1️⃣ Quick Test

Verify environment and parameters:

```bash
python quick_start.py bulgari_pendant.glb test
```

Generate small test dataset:

```bash
python quick_start.py bulgari_pendant.glb dataset
```

### 2️⃣ Generate Authentic Training Data

```bash
# Generate 2000 authentic samples (recommended)
python production_config.py bulgari_pendant.glb --num-views 2000

# Or generate in batches (if memory limited)
python production_config.py bulgari_pendant.glb --num-views 500
```

### 3️⃣ Collect Counterfeit Data

Manual collection of counterfeit images required (200-500 recommended):

**Suggested Sources:**
- E-commerce platforms (search "divas dream pendant", sort by price)
- Wholesale websites
- Second-hand marketplaces

Save images to `./fake_images/` directory

### 4️⃣ Run Complete Training Pipeline

```bash
python workflow.py \
    --glb-path bulgari_pendant.glb \
    --fake-dir ./fake_images \
    --num-samples 2000
```

This will automatically:
1. Generate authentic data
2. Organize counterfeit data
3. Create balanced dataset
4. Train deep learning model
5. Generate evaluation report

### 5️⃣ Test Authentication Performance

Single image test:
```bash
python predict.py --image test_pendant.jpg
```

Batch testing:
```bash
python predict.py --batch --dir ./test_images
```

## 📁 Project Structure

```
bulgari_project/
├── 📄 Core Scripts
│   ├── pendant_renderer.py        # 3D rendering engine (independent camera control)
│   ├── pendant_classifier.py      # Deep learning model
│   ├── production_config.py       # Production environment configuration
│   ├── calibration_tool.py        # Camera calibration tool
│   ├── workflow.py                # Complete pipeline management
│   └── quick_start.py            # Quick test script
│
├── 📊 Data Files
│   ├── bulgari_pendant.glb       # 3D model file
│   └── camera_params.json        # Calibration parameters
│
├── 📁 Generated Directories
│   ├── dataset/                  # Dataset directory
│   │   ├── authentic/           # Authentic renders
│   │   ├── fake/               # Counterfeit images
│   │   ├── train/              # Training set
│   │   ├── val/                # Validation set
│   │   └── test/               # Test set
│   ├── models/                  # Trained models
│   ├── results/                 # Prediction results
│   └── calibration/            # Calibration outputs
│
└── 📝 Documentation
    ├── README.md               # This document
    └── requirements.txt        # Dependencies list
```

## 🔧 Technical Architecture

### 3D Rendering System
- **Framework**: Trimesh + PyRender
- **Features**: Independent camera control, parametric view generation
- **Augmentation**: Lighting variations, noise, blur, and other realistic effects

### Deep Learning Model
- **Architecture**: Dual-branch network (visual features + geometric features)
- **Backbone**: EfficientNet-B0
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: AdamW

### Data Processing
- **Balancing Strategy**: Automatic balancing of authentic/counterfeit samples
- **Data Augmentation**: Rotation, scaling, lighting, perspective transforms
- **Split Ratio**: 80% training, 10% validation, 10% testing

## 📊 Performance Metrics

### Expected Performance
- **Accuracy**: 85-95%
- **Recall**: >90% (minimize false negatives for authentic items)
- **Precision**: >85% (minimize false positives for counterfeits)
- **Inference Speed**: <100ms/image (GPU)

### Improvement Methods
1. Increase data volume (1000+ samples per class)
2. Fine-tune hyperparameters
3. Ensemble multiple models
4. Add more features (material, luster, etc.)

### 🐛 Troubleshooting

#### Common Issues

**Q: OpenGL-related errors**
```bash
# Windows
set PYOPENGL_PLATFORM=windows

# Linux (headless server)
export PYOPENGL_PLATFORM=osmesa
```

**Q: Out of memory**
```python
# Reduce batch size or generation count
python production_config.py bulgari_pendant.glb --num-views 100
```

**Q: Rendered images are black/abnormal**
- Check if GLB file is correct
- Verify camera parameters
- Run `python quick_start.py bulgari_pendant.glb test`

## 📈 Project Progress

### ✅ Completed
- [x] 3D model loading and rendering
- [x] Camera parameter calibration
- [x] Data generation system
- [x] Data augmentation algorithms
- [x] Project architecture design
- [x] Workflow automation

### 🚧 In Progress
- [ ] Collecting counterfeit data
- [ ] Model training optimization
- [ ] Web interface development

### 📝 To-Do
- [ ] Mobile deployment
- [ ] API service
- [ ] Batch processing optimization
- [ ] More pricy jewrly to learn!

### 🤝 Contributing as a Community

#### Tech Contribute
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

#### Data Point Contribute
Contact me with the email below if you have more images for training!

#### 📜 Disclaimer

-  License
    MIT License - For educational and research purposes only
- This system is for technical research and educational purposes only
- Authentication results are for reference only and should not be used as legal evidence
- For important transactions, please consult professional authentication institutions.

#### 🩷 Acknowledgments

- PyTorch, Trimesh and PyRender developers
- Open source community

#### 📧 Contact
For questions or suggestions: me@paradiselab.io

---

**Last Updated**: May 2025 
**Version**: 1.0.0  
**Status**: 🟢 Active Development