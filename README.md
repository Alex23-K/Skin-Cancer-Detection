---
title: Skin Cancer Detection App
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
---

Link to the demo: https://huggingface.co/spaces/Alex5262/Skin-Cancer-Detection-App


# 🔬 Skin Cancer Detection App

An AI-powered skin lesion analysis tool using advanced deep learning techniques for educational and research purposes.

## 🎯 Features

- **Advanced AI Model**: MobileNetV4-based architecture trained on 148,000+ skin lesion images
- **Multi-class Detection**: Identifies 4 types of skin conditions
- **Confidence Scoring**: Provides detailed probability analysis
- **Metadata Integration**: Optional age and gender inputs for improved accuracy
- **Mobile-Friendly**: Responsive design for all devices
- **Real-time Analysis**: Instant results upon image upload

## 🏥 Classifications

- 🔴 **Melanoma Detected**: Most serious form of skin cancer
- 🟠 **Other Skin Cancer Detected**: Other malignant skin conditions  
- 🟡 **Benign Lesion Detected**: Non-cancerous skin growth
- 🟢 **No Lesion Detected**: Normal skin or no significant findings

## 🚀 How to Use

1. **Upload Image**: Take or upload a clear photo of the skin area
2. **Add Details** (Optional): Provide patient age and gender for enhanced accuracy
3. **Get Results**: Receive instant AI analysis with confidence scores
4. **Interpret**: Review detailed probability breakdown for all classes

## 📊 Model Performance

- **Architecture**: MobileNetV4ConvMedium with fusion layers
- **Training Dataset**: 148,000+ professionally annotated skin images
- **Validation**: Rigorous testing on diverse skin types and conditions
- **Optimization**: Balanced for sensitivity and specificity across all classes

## 💡 Best Practices for Accurate Results

- Use well-lit, clear photographs
- Ensure the lesion fills most of the image frame
- Avoid blurry or low-resolution images
- Consider multiple angles for comprehensive assessment
- Provide accurate metadata when possible

## ⚠️ Important Medical Disclaimer

**This tool is for educational and research purposes only.**

- **NOT a medical device**: This AI system is not approved for clinical diagnosis
- **NOT a replacement**: Always consult qualified dermatologists and healthcare professionals
- **Educational use**: Designed for learning about AI in medical imaging
- **Research tool**: Useful for understanding machine learning applications in healthcare
- **No guarantees**: Results should never be used as the sole basis for medical decisions

**If you have concerns about skin lesions, please consult a healthcare professional immediately.**

## 🔬 Technical Details

### Model Architecture
- **Base Model**: MobileNetV4ConvMedium
- **Custom Layers**: Fusion architecture combining image and metadata features
- **Input Resolution**: 224x224 pixels
- **Color Channels**: RGB (3 channels)
- **Metadata Features**: Age (normalized) and gender (encoded)

### Training Process
- **Data Augmentation**: Random crops, flips, rotations, color jittering
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Cross-entropy loss
- **Validation**: 70/15/15 train/validation/test split
- **Hardware**: GPU-accelerated training

### Performance Metrics
- **Accuracy**: Measured across all four classes
- **Sensitivity**: True positive rate for each condition type
- **Specificity**: True negative rate for each condition type
- **AUC-ROC**: Area under the receiver operating characteristic curve

## 📱 Usage Examples

### Basic Usage
1. Upload any skin image
2. Click "Analyze Image"
3. Review results and confidence scores

### Enhanced Usage
1. Upload skin image
2. Enable "Include patient information"
3. Set age and gender
4. Get more personalized analysis

## 🛠️ Development

This space was created as part of a Data Science final project, demonstrating the application of deep learning to medical imaging challenges.

### Technologies Used
- **Deep Learning**: PyTorch framework
- **Computer Vision**: Torchvision transformations
- **Web Interface**: Gradio for interactive demos
- **Model Hub**: Hugging Face for deployment
- **Data Processing**: NumPy, Pandas, Scikit-learn

## 📄 License

This project is licensed under the MIT License - see the space configuration for details.

## 🙏 Acknowledgments

- MobileNetV4 architecture from research community
- Open-source skin lesion datasets
- Hugging Face platform for hosting
- Medical professionals who provide ground truth annotations



---

**This is an educational demonstration. For real-life concerns, please consult healthcare professionals for medical advice.**