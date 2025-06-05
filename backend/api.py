from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Gerekli modelleri burada import et
# örn: from model import detect_drowsiness

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Flask API is running!'



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        # base64 string -> bytes
        image_bytes = base64.b64decode(image_data.split(',')[1])
        # bytes -> PIL image
        image = Image.open(io.BytesIO(image_bytes))
        # PIL image -> OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Buraya tahmin fonksiyonunu koy:
        # örnek sonuç:
        result = {"status": "drowsy", "confidence": 0.92}

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
