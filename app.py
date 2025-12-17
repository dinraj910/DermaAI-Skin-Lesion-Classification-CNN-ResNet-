from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = tf.keras.models.load_model("models/skin_lesion_resnet_cam.keras")

CLASS_NAMES = ["Benign", "Malignant"]
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    """Preprocess image for model prediction"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image file")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))  # Model expects 384x384
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict skin lesion classification"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return render_template("error.html", error="No image file provided"), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template("error.html", error="No file selected"), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return render_template("error.html", error="Invalid file type. Please upload an image."), 400
        
        # Save file with unique name
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Preprocess and predict
        img = preprocess_image(filepath)
        raw_prediction = model.predict(img, verbose=0)
        
        # Handle multi-output models (e.g., model returns list of outputs)
        if isinstance(raw_prediction, list):
            # Take first output if multiple outputs
            raw_prediction = raw_prediction[0]
        
        # Now raw_prediction should be a numpy array
        pred_array = np.squeeze(raw_prediction)  # Remove single dimensions
        
        # Handle both single output (sigmoid) and two-class output (softmax)
        if pred_array.ndim == 0:
            # Single scalar value (sigmoid)
            pred_value = float(pred_array)
            label = "Malignant" if pred_value > 0.5 else "Benign"
            confidence_value = pred_value if pred_value > 0.5 else (1 - pred_value)
        elif pred_array.size == 1:
            # Single element array (sigmoid)
            pred_value = float(pred_array.flat[0])
            label = "Malignant" if pred_value > 0.5 else "Benign"
            confidence_value = pred_value if pred_value > 0.5 else (1 - pred_value)
        else:
            # Multi-class output (softmax): [benign_prob, malignant_prob]
            pred_flat = pred_array.flatten()
            confidence_value = float(np.max(pred_flat))
            label = "Malignant" if int(np.argmax(pred_flat)) == 1 else "Benign"
        
        confidence = f"{confidence_value*100:.1f}%"
        
        # Relative path for template
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename).replace("\\", "/")
        
        return render_template(
            "result.html",
            image=image_path,
            label=label,
            confidence=confidence,
            prediction_score=f"{confidence_value*100:.1f}"
        )
    
    except Exception as e:
        return render_template("error.html", error=f"An error occurred: {str(e)}"), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template("error.html", error="Internal server error"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
