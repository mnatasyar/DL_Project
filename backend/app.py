import os
import joblib
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_cors import CORS

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)

# Konfigurasi folder untuk menyimpan file
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load model dan scaler
model_path = 'mlp_model.pkl'
scaler_path = 'scaler.pkl'
cat_face_cascade_path = 'haarcascade_frontalcatface.xml'

mlp = joblib.load(model_path)
scaler = joblib.load(scaler_path)
cat_face_cascade = cv2.CascadeClassifier(cat_face_cascade_path)

# Route untuk mengakses file statis
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

def generate_unique_filename(original_filename, folder):
    """Membuat nama file unik dengan timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename, extension = os.path.splitext(original_filename)
    return f"{filename}_{timestamp}{extension}"

def predict_cat_health(image_path):
    # Membaca gambar baru
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah kucing
    faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return "Wajah tidak terdeteksi", None

    # Variabel untuk menyimpan hasil prediksi
    labels = []

    # Preprocessing data dan prediksi untuk setiap wajah
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        face_flattened = face_resized.flatten()
        face_scaled = scaler.transform([face_flattened])

        # Prediksi kesehatan kucing
        prediction = mlp.predict(face_scaled)
        label = "Sehat" if prediction[0] == 1 else "Sakit"
        labels.append(label)

        # Menambahkan bounding box pada gambar
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tentukan hasil akhir berdasarkan mayoritas prediksi
    final_label = "Sehat" if labels.count("Sehat") > labels.count("Sakit") else "Sakit"

    # Menyimpan gambar hasil prediksi dengan nama unik
    result_filename = generate_unique_filename('result.png', RESULT_FOLDER)
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f'Hasil Prediksi: {final_label}')
    plt.savefig(result_path, format='png')
    plt.close()

    return final_label, result_filename

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename, UPLOAD_FOLDER)
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            label, result_filename = predict_cat_health(file_path)

            if label == "Wajah tidak terdeteksi":
                return jsonify({"message": label}), 400

            return jsonify({
                "label": label,
                "image_url": f"results/{result_filename}",
                "original_image": f"uploads/{unique_filename}"
            })
        except Exception as e:
            return jsonify({"message": f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
