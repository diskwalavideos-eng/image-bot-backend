import os
import cv2
import easyocr
import numpy as np
from flask import Flask, request, send_file

app = Flask(__name__)

print('Initializing EasyOCR model once into RAM...')
reader = easyocr.Reader(['en'], gpu=False)
print('EasyOCR model ready!')

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

    results = reader.readtext(input_path)

    if results:
        mask = np.zeros(img.shape[:2], dtype='uint8')
        for bbox, text, prob in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        cleaned_img = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    else:
        cleaned_img = img

    if new_text and new_text.strip() != '':
        img_h, img_w = cleaned_img.shape[:2]
        detected_height = manual_font_size
        if position == 'auto-detected' and results:
            box = results[0][0]
            detected_height = int(abs(box[3][1] - box[0][1]))
            if detected_height < 15:
                detected_height = manual_font_size

        font_scale = max(0.5, detected_height / 30.0)
        thickness = max(1, int(font_scale * 2))

        (text_width, text_height), baseline = cv2.getTextSize(
            new_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        x, y = (img_w - text_width) // 2, img_h - 40

        if position == 'auto-detected' and results:
            x = int(results[0][0][0][0])
            y = int(results[0][0][0][1]) + text_height + 5
            if x + text_width > img_w: x = img_w - text_width - 20
            if x < 10: x = 10
            if y > img_h - 20: y = img_h - 40
        elif position == 'bottom-center':
            x, y = (img_w - text_width) // 2, img_h - 40

        cv2.putText(cleaned_img, new_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, cleaned_img)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)