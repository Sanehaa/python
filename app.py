from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image):
    """Enhance the image for better shape recognition."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Perform morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closed
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def classify_shape(contour):
    """Classify a shape based on its contour."""
    try:
        # Approximate the contour to reduce vertices
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Number of vertices
        vertices = len(approx)

        # Get bounding rectangle to check aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # Solidity: ratio of contour area to convex hull area
        contour_area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        # Shape classification logic
        if vertices == 3 and solidity > 0.8:
            return "Triangle"
        elif vertices == 4:
            if 0.9 <= aspect_ratio <= 1.1 and solidity > 0.9:
                return "Square"
            else:
                return "Rectangle"
        elif vertices > 5:
            # Additional check for circularity
            circularity = (4 * np.pi * contour_area) / (perimeter**2)
            if 0.7 <= circularity <= 1.3:  # Range for circular shapes
                return "Circle"
        return "Unknown"
    except Exception as e:
        logger.error(f"Error in classify_shape: {str(e)}")
        return "Unknown"

def detect_shapes(image_path):
    """Detect shapes in the image."""
    try:
        logger.info(f"Processing image: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        logger.info("Image loaded successfully")
        original = image.copy()

        # Preprocess the image
        processed = preprocess_image(image)
        logger.info("Image preprocessing completed")

        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Found {len(contours)} contours")

        detected_shapes = []

        for contour in contours:
            # Ignore small contours
            if cv2.contourArea(contour) < 1000:
                continue

            # Classify the shape
            shape = classify_shape(contour)

            # Get the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Draw the shape on the original image
            cv2.drawContours(original, [contour], -1, (0, 255, 0), 2)
            cv2.putText(original, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            detected_shapes.append((shape, (cX, cY)))

        # Save the processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
        cv2.imwrite(output_path, original)
        logger.info(f"Processed image saved to: {output_path}")

        return [shape[0] for shape in detected_shapes], 'processed_' + os.path.basename(image_path)

    except Exception as e:
        logger.error(f"Error in detect_shapes: {str(e)}\n{traceback.format_exc()}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                shapes, processed_filename = detect_shapes(filepath)
                return jsonify({
                    'shapes': shapes,
                    'processed_image': f'/uploads/{processed_filename}'
                })
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
