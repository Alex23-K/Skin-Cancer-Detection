# app.py
import gradio as gr
import torch
import os
from model import load_model, predict_image, LABELS

# ───────────────────────────────────────────────────────────────────────────────
# 1) Initialize model and device
# ───────────────────────────────────────────────────────────────────────────────
model = None
device = None


def initialize_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "fusion_scratch_best.pth"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    print(f"Loading model from {model_path} onto {device}...")
    model = load_model(model_path, device).to(device)
    model.eval()
    print("Model loaded and ready.")

    # Debug model behavior with test images
    print("\n🔍 DEBUGGING MODEL ON DIFFERENT IMAGE TYPES:")
    try:
        from model import debug_model_predictions
        debug_model_predictions(model, device)
    except:
        print("Debug function not available")

    # Test with simple nature image
    print("\n🌿 Testing with green nature-like image:")
    try:
        from PIL import Image
        green_img = Image.new('RGB', (224, 224), color=(34, 139, 34))  # Forest green
        result = predict_image(model, green_img, metadata=None, device=device)
        print(f"Green image prediction: {LABELS[result['predicted_class']]} ({result['confidence']:.3f})")

        # Fix the f-string syntax error
        prob_details = []
        for i in range(4):
            label = LABELS[i]
            prob = result['probabilities'][i]
            prob_details.append(f"{label}: {prob:.3f}")
        print(f"All probabilities: {prob_details}")

    except Exception as e:
        print(f"Green image test failed: {e}")


initialize_model()


# ───────────────────────────────────────────────────────────────────────────────
# 2) Helper: waiting placeholder
# ───────────────────────────────────────────────────────────────────────────────
def create_waiting_result():
    return """
    <div style="
      background: #e0f2fe; border:2px solid #0284c7; border-radius:20px;
      padding:40px; text-align:center;">
      <div style="font-size:4rem;">📷</div>
      <div style="color:#0284c7;font-size:1.8rem;font-weight:600;">
        Ready for Analysis
      </div>
      <div style="color:#075985;font-size:1.2rem;">
        Upload a clear image of the skin area to begin
      </div>
    </div>
    """


# ───────────────────────────────────────────────────────────────────────────────
# 3) Create main result card
# ───────────────────────────────────────────────────────────────────────────────
def create_main_result(predicted_class, confidence):
    prediction_text = LABELS[predicted_class]
    if predicted_class == 0:
        color, icon = "#DC2626", "🔴"
        urgency, action = "HIGH PRIORITY", "Consult a dermatologist immediately"
    elif predicted_class == 1:
        color, icon = "#EA580C", "🟠"
        urgency, action = "ATTENTION REQUIRED", "Consult a dermatologist immediately"
    elif predicted_class == 2:
        color, icon = "#D97706", "🟡"
        urgency, action = "ROUTINE MONITORING", "Regular skin check recommended"
    else:
        color, icon = "#059669", "🟢"
        urgency, action = "NO IMMEDIATE CONCERN", "Continue regular skin surveillance"

    conf_pct = confidence * 100 if confidence <= 1 else confidence

    return f"""
    <div style="
        background:#fff; border:2px solid {color}; border-radius:20px;
        padding:40px; text-align:center; box-shadow:0 20px 25px -5px rgba(0,0,0,0.1);
    ">
      <div style="font-size:4rem;margin-bottom:20px;">{icon}</div>
      <div style="
          color:{color}; font-size:2.5rem; font-weight:700;
          margin-bottom:15px; text-transform:uppercase;
      ">{prediction_text}</div>
      <div style="
          background:{color}; color:#fff;
          padding:12px 24px; border-radius:50px;
          font-size:1.1rem; font-weight:600; display:inline-block;
      ">{urgency}</div>
      <div style="
          font-size:2rem; font-weight:600; color:#1f2937;
          margin:25px 0 15px;
      ">Confidence: {conf_pct:.1f}%</div>
      <div style="
          background:#f1f5f9; border-radius:15px;
          padding:20px; margin-top:25px; border-left:5px solid {color};
      ">
        <div style="
            color:#475569; font-size:1.2rem; font-weight:500;
            margin-bottom:8px;
        ">Recommended Action:</div>
        <div style="
            color:#1e293b; font-size:1.1rem; font-weight:400;
        ">{action}</div>
      </div>
    </div>
    """


# ───────────────────────────────────────────────────────────────────────────────
# 4) Create detailed analysis section
# ───────────────────────────────────────────────────────────────────────────────
def create_detailed_result(probabilities, metadata, age, gender):
    colors = ["#DC2626", "#EA580C", "#D97706", "#059669"]
    prob_bars = ""

    # Ensure probabilities are in correct format and range
    probs = probabilities
    if isinstance(probs, (list, tuple)) and len(probs) == 4:
        # Convert to list and ensure all values are between 0 and 1
        probs = [float(p) for p in probs]
        probs = [max(0.0, min(1.0, p)) for p in probs]  # Clamp between 0 and 1

        # Normalize if sum is not 1 (with small tolerance)
        prob_sum = sum(probs)
        if abs(prob_sum - 1.0) > 0.01:  # If sum is not close to 1
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
            else:
                probs = [0.25, 0.25, 0.25, 0.25]  # Equal distribution as fallback
    else:
        # Fallback: equal distribution
        probs = [0.25, 0.25, 0.25, 0.25]

    print(f"Processed probabilities: {probs}, Sum: {sum(probs)}")

    for i, label in LABELS.items():
        prob = probs[i]
        pct = prob * 100  # Convert to percentage (0-100)
        bar_width = max(pct, 1)  # Minimum 1% for visibility
        color = colors[i]

        prob_bars += f"""
        <div style="margin-bottom:20px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-weight:600;color:#374151;">{label}</span>
            <span style="font-weight:700;color:{color};">{pct:.1f}%</span>
          </div>
          <div style="background:#e5e7eb;height:12px;border-radius:6px;overflow:hidden;">
            <div style="background:{color};height:100%;width:{bar_width}%;border-radius:6px;"></div>
          </div>
        </div>
        """

    if metadata:
        meta_section = f"""
        <div style="
            background:#f8fafc;border:1px solid #e2e8f0;
            border-radius:15px;padding:25px;margin-top:30px;
        ">
          <h4 style="font-size:1.3rem;color:#1e293b;margin-bottom:20px;">
            📋 Patient Information Used
          </h4>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div style="
                background:#fff;padding:15px;border-radius:10px;
                border:1px solid #e2e8f0;text-align:center;
            ">
              <div style="color:#64748b;font-size:0.9rem;margin-bottom:5px;">Age</div>
              <div style="color:#1e293b;font-size:1.4rem;font-weight:600;">{age} years</div>
            </div>
            <div style="
                background:#fff;padding:15px;border-radius:10px;
                border:1px solid #e2e8f0;text-align:center;
            ">
              <div style="color:#64748b;font-size:0.9rem;margin-bottom:5px;">Gender</div>
              <div style="color:#1e293b;font-size:1.4rem;font-weight:600;">{gender}</div>
            </div>
          </div>
        </div>
        """
    else:
        meta_section = f"""
        <div style="
            background:#fef3c7;border:1px solid #f59e0b;
            border-radius:15px;padding:20px;margin-top:30px;
            text-align:center;
        ">
          <div style="color:#92400e;font-size:1.1rem;">
            ℹ️ Tip: Including age and gender can improve accuracy
          </div>
        </div>
        """

    return f"""
    <div style="
        background:#fff;border-radius:20px;padding:30px;
        margin:20px 0;box-shadow:0 10px 25px -5px rgba(0,0,0,0.1);
        border:1px solid #e2e8f0;
    ">
      <h3 style="
          font-size:1.8rem;color:#1e293b;font-weight:700;
          margin-bottom:25px;text-align:center;
          border-bottom:2px solid #e2e8f0;padding-bottom:15px;
      ">📊 Detailed Analysis</h3>
      {prob_bars}
      {meta_section}
    </div>
    """


# ───────────────────────────────────────────────────────────────────────────────
# 5) Prediction function
# ───────────────────────────────────────────────────────────────────────────────
def predict_skin_lesion(image, age, gender, use_meta):
    if image is None:
        return create_waiting_result(), ""

    metadata = None
    if use_meta:
        metadata = (float(age) / 100.0, 1.0 if gender.lower() == "female" else 0.0)

    try:
        out = predict_image(model, image, metadata, device)
        print(f"Raw prediction output: {out}")
        print(f"Output type: {type(out)}")

        # Handle different return formats properly
        if isinstance(out, dict):
            # Dictionary format: {'predicted_class': int, 'probabilities': array, 'confidence': float}
            pc = out["predicted_class"]
            probs = out["probabilities"]
            conf = out["confidence"]
            print(f"Dict format - Class: {pc}, Confidence: {conf}, Probs: {probs}")

        elif isinstance(out, tuple) and len(out) == 2:
            # Tuple format: (predicted_class, confidence_percentage)
            pc, conf = out
            print(f"Tuple format - Class: {pc}, Confidence: {conf}")

            # Convert confidence to 0-1 range if it's in percentage (>1)
            if conf > 1:
                conf = conf / 100.0

            # Create probability distribution
            probs = [0.0] * 4
            probs[pc] = conf

            # Distribute remaining probability equally among other classes
            remaining_prob = 1.0 - conf
            other_prob = remaining_prob / 3.0
            for i in range(4):
                if i != pc:
                    probs[i] = other_prob

        else:
            # Fallback: assume it's just the predicted class
            pc = int(out) if not isinstance(out, int) else out
            conf = 0.75  # Default confidence
            probs = [0.0] * 4
            probs[pc] = conf

            # Distribute remaining probability
            remaining_prob = 1.0 - conf
            other_prob = remaining_prob / 3.0
            for i in range(4):
                if i != pc:
                    probs[i] = other_prob

        # Ensure probabilities are valid (sum to 1, all positive)
        probs = [max(0.0, min(1.0, p)) for p in probs]  # Clamp between 0 and 1
        prob_sum = sum(probs)
        if prob_sum > 0:
            probs = [p / prob_sum for p in probs]  # Normalize to sum to 1

        # Ensure confidence is in 0-1 range
        conf = max(0.0, min(1.0, conf))

        print(f"Final - Class: {pc}, Confidence: {conf}, Probabilities: {probs}")
        print(f"Probability sum: {sum(probs)}")

        main_html = create_main_result(pc, conf)
        detail_html = create_detailed_result(probs, metadata, age, gender)
        return main_html, detail_html

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()

        # Return error message
        error_html = f"""
        <div style="background:#fef2f2;border:2px solid #dc2626;border-radius:20px;padding:40px;text-align:center;">
            <div style="font-size:4rem;">❌</div>
            <div style="color:#dc2626;font-size:1.8rem;font-weight:600;">Prediction Error</div>
            <div style="color:#7f1d1d;font-size:1.2rem;">Error: {str(e)}</div>
        </div>
        """
        return error_html, ""


# ───────────────────────────────────────────────────────────────────────────────
# 6) App-wide CSS for layout
# ───────────────────────────────────────────────────────────────────────────────
css = """
.gradio-container { background: #f3f4f6; }
.input-section, .results-section {
  background:#fff; border-radius:20px; padding:30px; margin:10px;
  box-shadow:0 10px 25px -5px rgba(0,0,0,0.1);
}
.image-upload { border:3px dashed #cbd5e1 !important; border-radius:20px !important; }
.analyze-button { width:100% !important; font-size:1.2rem !important; }
.model-info-section {
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: 20px;
  padding: 30px;
  margin: 20px 10px;
  box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
  border: 2px solid #cbd5e1;
}
"""

# ───────────────────────────────────────────────────────────────────────────────
# 7) Build the Gradio interface
# ───────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="🔬 AI Skin Cancer Detection") as demo:
    # Centered title
    gr.Markdown("<h1 style='text-align:center;'>🔬 AI Skin Cancer Detection</h1>")
    gr.Markdown("<p style='text-align:center; font-size: 1.5em;'>Upload a lesion photo and let the AI do the rest</p>")

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["input-section"]):
            gr.Markdown("### 📥 Patient & Image Input")
            image_input = gr.Image(label="Upload skin image", type="pil", elem_classes=["image-upload"])
            with gr.Row():
                age_input = gr.Slider(0, 100, value=35, label="Age (years)")
                gender_input = gr.Radio(["Male", "Female"], label="Gender", value="Male")
            use_meta = gr.Checkbox(label="Use patient info (recommended)", value=True)
            predict_btn = gr.Button("🔍 Analyze Lesion", elem_classes=["analyze-button"])

            gr.Markdown("### 🖼️ Try Sample Images")
            gr.Examples(
                examples=[
                    ["sample_photo_1.jpg", 30, "Male", True],
                    ["sample_photo_2.jpg", 75, "Female", True],
                    ["sample_photo_3.jpg", None, None, False],
                ],
                inputs=[image_input, age_input, gender_input, use_meta]
            )

        with gr.Column(scale=1, elem_classes=["results-section"]):
            main_out = gr.HTML(value=create_waiting_result())
            detail_out = gr.HTML()

    # Model Information Section - Collapsible
    with gr.Accordion("🧠 Tell me more about the model used", open=False):
        gr.HTML("""
        <div class="model-info-section">
            <h3 style="
                color: #1e293b;
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 25px;
                text-align: center;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
            ">🔬 AI Model Technical Details</h3>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin: 30px 0;">

                <!-- Architecture Section -->
                <div style="
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
                    border-left: 5px solid #667eea;
                ">
                    <h4 style="
                        color: #667eea;
                        font-size: 1.4rem;
                        font-weight: 600;
                        margin-bottom: 15px;
                        display: flex;
                        align-items: center;
                    ">
                        🏗️ Model Architecture
                    </h4>
                    <ul style="
                        color: #475569;
                        font-size: 1.05rem;
                        line-height: 1.7;
                        margin: 0;
                        padding-left: 20px;
                    ">
                        <li><strong>Base Architecture:</strong> MobileNetV4-ConvMedium</li>
                        <li><strong>Model Type:</strong> Fusion Neural Network</li>
                        <li><strong>Input Modalities:</strong> Images + Patient Demographics</li>
                        <li><strong>Output Classes:</strong> 4-way Classification</li>
                        <li><strong>Framework:</strong> PyTorch Deep Learning</li>
                        <li><strong>Optimization:</strong> Mobile-optimized for efficiency</li>
                    </ul>
                </div>

                <!-- Training Data Section -->
                <div style="
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
                    border-left: 5px solid #10b981;
                ">
                    <h4 style="
                        color: #10b981;
                        font-size: 1.4rem;
                        font-weight: 600;
                        margin-bottom: 15px;
                        display: flex;
                        align-items: center;
                    ">
                        📊 Training Dataset Breakdown
                    </h4>
                    <div style="
                        background: #f0fdfa;
                        border: 1px solid #a7f3d0;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 15px;
                        text-align: center;
                    ">
                        <div style="
                            font-size: 2.5rem;
                            font-weight: 800;
                            color: #065f46;
                            margin-bottom: 5px;
                        ">147,965</div>
                        <div style="
                            font-size: 1rem;
                            color: #047857;
                            font-weight: 600;
                        ">Total Training Images</div>
                    </div>

                    <div style="margin-top: 20px;">
                        <div style="font-weight: 600; color: #1e293b; margin-bottom: 10px;">Dataset Composition:</div>
                        <ul style="margin: 0; padding-left: 20px; color: #475569; font-size: 1rem; line-height: 1.6;">
                            <li><strong>36,482 medical images</strong> from ISIC 2019 dataset</li>
                            <li><strong>MIDAS dataset - Multimodal Image Dataset for AI-based Skin Cancer </strong> with ground truth biopsy diagnosis</li>
                            <li><strong>111,490 non-lesion images</strong> for robust training</li>
                        </ul>
                    </div>
                </div>

                <!-- Image Sources Section -->
                <div style="
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
                    border-left: 5px solid #f59e0b;
                ">
                    <h4 style="
                        color: #f59e0b;
                        font-size: 1.4rem;
                        font-weight: 600;
                        margin-bottom: 15px;
                        display: flex;
                        align-items: center;
                    ">
                        📷 Image Sources & Types
                    </h4>
                    <div style="
                        color: #475569;
                        font-size: 1.05rem;
                        line-height: 1.7;
                    ">
                        <div style="font-weight: 600; color: #1e293b; margin-bottom: 10px;">Medical Images:</div>
                        <ul style="margin: 0 0 15px 0; padding-left: 20px;">
                            <li><strong>Dermoscopic photographs</strong> - Professional dermatology images</li>
                            <li><strong>Clinical photos</strong> - Captured at 15-10 cm distance</li>
                            <li><strong>Mobile phone images</strong> - Real-world smartphone photos</li>
                            <li><strong>Biopsy-confirmed diagnoses</strong> - Ground truth validation</li>
                        </ul>

                        <div style="font-weight: 600; color: #1e293b; margin-bottom: 10px;">Non-lesion Images:</div>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li><strong>Nature photographs</strong> - Diverse outdoor scenes</li>
                            <li><strong>Animals and flowers</strong> - Natural texture variety</li>
                            <li><strong>Human body parts</strong> - Other skin regions without lesions</li>
                            <li><strong>General objects</strong> - Common everyday items</li>
                        </ul>
                    </div>
                </div>

                <!-- Clinical Validation Section -->
                <div style="
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
                    border-left: 5px solid #dc2626;
                ">
                    <h4 style="
                        color: #dc2626;
                        font-size: 1.4rem;
                        font-weight: 600;
                        margin-bottom: 15px;
                        display: flex;
                        align-items: center;
                    ">
                        🏥 Clinical Features
                    </h4>
                    <ul style="
                        color: #475569;
                        font-size: 1.05rem;
                        line-height: 1.7;
                        margin: 0;
                        padding-left: 20px;
                    ">
                        <li><strong>Multi-modal fusion:</strong> Combines image analysis with patient demographics</li>
                        <li><strong>Mobile-optimized:</strong> Designed for smartphone compatibility</li>
                        <li><strong>Clinical-grade training:</strong> Based on medical datasets with expert validation</li>
                        <li><strong>Robust classification:</strong> Extensive negative examples prevent false positives</li>
                        <li><strong>Real-world applicability:</strong> Tested with various lighting and distance conditions</li>
                        <li><strong>Professional validation:</strong> Ground truth confirmed through biopsy results</li>
                    </ul>
                </div>
            </div>

            <!-- Technical Specifications -->
            <div style="
                background: #f8fafc;
                border: 2px solid #e2e8f0;
                border-radius: 15px;
                padding: 25px;
                margin-top: 30px;
            ">
                <h4 style="
                    color: #1e293b;
                    font-size: 1.3rem;
                    font-weight: 600;
                    margin-bottom: 20px;
                    text-align: center;
                ">⚙️ Technical Specifications</h4>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px;">
                    <div style="text-align: center; padding: 15px;">
                        <div style="color: #667eea; font-size: 2rem; font-weight: 700;">224×224</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Input Resolution</div>
                    </div>
                    <div style="text-align: center; padding: 15px;">
                        <div style="color: #10b981; font-size: 2rem; font-weight: 700;">4</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Output Classes</div>
                    </div>
                    <div style="text-align: center; padding: 15px;">
                        <div style="color: #f59e0b; font-size: 2rem; font-weight: 700;">RGB</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Color Channels</div>
                    </div>
                    <div style="text-align: center; padding: 15px;">
                        <div style="color: #dc2626; font-size: 2rem; font-weight: 700;">2</div>
                        <div style="color: #64748b; font-size: 0.9rem;">Metadata Features</div>
                    </div>
                </div>
            </div>

            <!-- Research Purpose -->
            <div style="
                background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
                border: 2px solid #8b5cf6;
                border-radius: 15px;
                padding: 20px;
                margin-top: 25px;
                text-align: center;
            ">
                <div style="
                    color: #5b21b6;
                    font-size: 1.1rem;
                    font-weight: 600;
                    line-height: 1.6;
                ">
                    🔬 <strong>Research & Educational Purpose:</strong><br>
                    This model demonstrates the practical application of state-of-the-art deep learning techniques 
                    to medical image analysis, showcasing how AI can assist in dermatological screening while 
                    emphasizing the importance of professional medical consultation for definitive diagnosis.
                </div>
            </div>
        </div>
        """)

    # About the Creator Section - Simple and Modest
    with gr.Accordion("👨‍💻 About the Creator", open=False):
        gr.HTML("""
        <div style="
            background: #f8fafc;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border: 1px solid #e2e8f0;
            text-align: center;
            max-width: 500px;
            margin: 20px auto;
        ">
            <h3 style="
                color: #374151;
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 20px;
            "> The app was created by Alex Kagan - Medical Data Scientist </h3>

            <div style="
                color: #6b7280;
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 15px;
            ">
                <strong>For contact, reach me via my LinkedIn profile </strong>
            </div>

            <div style="
                margin-top: 20px;
            ">
                <a href="https://www.linkedin.com/in/alex-kagan-317a375a/" 
                   target="_blank" 
                   style="
                       color: #0066cc;
                       text-decoration: none;
                       font-size: 1rem;
                       padding: 8px 16px;
                       border: 1px solid #0066cc;
                       border-radius: 5px;
                       display: inline-block;
                       transition: all 0.3s ease;
                   "
                   onmouseover="this.style.backgroundColor='#0066cc'; this.style.color='white';"
                   onmouseout="this.style.backgroundColor='transparent'; this.style.color='#0066cc';">
                    🔗 LinkedIn Profile
                </a>
            </div>
        </div>
        """)

    # Medical Disclaimer
    gr.HTML("""
    <div style="
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 20px;
        padding: 25px;
        margin: 40px 10px 20px 10px;
        text-align: center;
    ">
        <div style="
            color: #92400e;
            font-size: 1.1rem;
            font-weight: 500;
            line-height: 1.6;
        ">
            ⚠️ <strong>IMPORTANT MEDICAL DISCLAIMER:</strong><br>
            This AI tool is designed for proof of concept and research purposes only. 
            For now, this tool is not a substitute for professional medical diagnosis or treatment. 
            Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment decisions.
        </div>

        <div style="
            margin-top: 20px; 
            padding-top: 20px; 
            border-top: 2px solid #f59e0b;
            color: #92400e;
            font-size: 1rem;
            font-weight: 500;
            line-height: 1.5;
        ">
            🔒 <strong>PRIVACY NOTICE:</strong><br>
            Uploaded images are processed temporarily in memory and are not saved or stored permanently. 
            All images are automatically deleted when your session ends. No personal data is retained.
        </div>
    </div>
    """)

    # Event handlers
    predict_btn.click(
        fn=predict_skin_lesion,
        inputs=[image_input, age_input, gender_input, use_meta],
        outputs=[main_out, detail_out]
    )

    # REMOVED: Auto-predict when image is uploaded
    # This allows users to adjust age/gender before analysis
    # image_input.change(
    #     fn=predict_skin_lesion,
    #     inputs=[image_input, age_input, gender_input, use_meta],
    #     outputs=[main_out, detail_out]
    # )

if __name__ == "__main__":
    demo.launch(share=True)