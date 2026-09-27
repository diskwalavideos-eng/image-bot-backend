import os
import cv2
import numpy as np
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/')
def home():
    return "Image Bot Backend is Live with Precise Multi-Text Inpainting! 🎉", 200

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 0, 255)
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

    # --- PRECISE MULTI-TEXT REMOVAL (NO BACKGROUND BLUR) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to catch text across various brightness
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    
    # Connect text characters into words/lines
    kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    closed_words = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_word)
    
    # Find contours and filter strictly for text blocks (ignores background & large objects)
    contours, _ = cv2.findContours(closed_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    total_area = img_w * img_h
    for cnt in contours:
        x, y, w, h = cnt_box = cv2.boundingRect(cnt)
        box_area = w * h
        # Strict constraints: text shouldn't be too huge, must have text-like proportions
        if 6 < h < 120 and 6 < w < (img_w * 0.95) and box_area < (total_area * 0.15):
            aspect_ratio = w / float(h)
            if aspect_ratio > 0.15: # Valid text aspect ratio
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
    # Small dilation so text edges are fully covered without expanding to background
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # Apply Inpainting only on detected text zones
    cleaned_img = cv2.inpaint(img, final_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    # --------------------------------------------------------

    # Place new text if provided
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
