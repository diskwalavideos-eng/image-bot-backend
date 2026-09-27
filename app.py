import os
import cv2
import numpy as np
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/')
def home():
    return "Image Bot Backend is Live and Running smoothly with Robust Multi-Text Inpainting! 🎉", 200

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 0, 255) # Default Red
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

@app.route('/process', methods=['POST'])
def process_image_api():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    new_text = request.form.get('new_text', '')
    manual_font_size = int(request.form.get('font_size', 40))
    position = request.form.get('position', 'bottom-center')
    color_hex = request.form.get('color', '#ffffff')
    text_color = hex_to_bgr(color_hex)

    input_path = 'temp_in.jpg'
    output_path = 'temp_out.jpg'
    file.save(input_path)

    img = cv2.imread(input_path)
    if img is None:
        return 'Invalid image', 400

    img_h, img_w = img.shape[:2]
    cleaned_img = img.copy()

    # --- ROBUST MULTI-TEXT REMOVAL (INPAINTING) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Use bilateral filter to smooth backgrounds while keeping text edges sharp
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Adaptive Thresholding to catch multiple text locations across varying brightness
    thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # 3. Kernel to connect characters into words and lines (handles multiple text regions robustly)
    kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed_words = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_word)
    
    # 4. Dilate slightly to ensure full coverage of text boundaries and edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(closed_words, kernel_dilate, iterations=2)
    
    # Apply OpenCV Inpainting across all detected text regions safely
    cleaned_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    # ---------------------------------------------

    # If new text is provided, place it nicely onto the cleaned image
    if new_text and new_text.strip() != '':
        font_scale = max(0.5, manual_font_size / 30.0)
        thickness = max(1, int(font_scale * 2))

        (text_width, text_height), baseline = cv2.getTextSize(
            new_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        if position == 'top-left':
            x, y = 30, text_height + 30
        elif position == 'top-center':
            x, y = (img_w - text_width) // 2, text_height + 30
        elif position == 'top-right':
            x, y = img_w - text_width - 30, text_height + 30
        elif position == 'center-left':
            x, y = 30, (img_h + text_height) // 2
        elif position == 'center':
            x, y = (img_w - text_width) // 2, (img_h + text_height) // 2
        elif position == 'center-right':
            x, y = img_w - text_width - 30, (img_h + text_height) // 2
        elif position == 'bottom-left':
            x, y = 30, img_h - 40
        elif position == 'bottom-right':
            x, y = img_w - text_width - 30, img_h - 40
        else:
            x, y = (img_w - text_width) // 2, img_h - 40

        cv2.putText(
            cleaned_img,
            new_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, cleaned_img)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
