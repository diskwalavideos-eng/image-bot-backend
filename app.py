import os
import cv2
import numpy as np
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/')
def home():
    return "Image Bot Backend is Live and Running smoothly with Inpainting! 🎉", 200

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

    # --- ADVANCED TEXT REMOVAL (INPAINTING) ---
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding to detect high-contrast text regions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    
    # Kernel to dilate the text regions slightly so inpainting covers edges cleanly
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=1)
    
    # Apply OpenCV Fast Marching Inpainting to remove existing text smoothly
    cleaned_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # ------------------------------------------

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

        # Optional subtle background pill/box behind new text for readability
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
