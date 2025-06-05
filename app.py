from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from pygame import mixer
import os
import threading
from datetime import datetime
import time
import queue
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global değişkenler
camera = None
frame_queue = queue.Queue(maxsize=2)
output_frame = None
lock = threading.Lock()
alarm_on = False
detection_active = False
drowsy_start_time = None
total_drowsy_time = 0
drowsy_events = 0
session_start_time = None
model = None
alarm_sound = None
CLOSED_DURATION_THRESHOLD = 5.0  # Gözlerin kapalı kalma süresi (saniye)

def init_sound():
    """Ses sistemini başlatır ve alarm sesini yükler."""
    global alarm_sound
    try:
        logger.info("Ses sistemi başlatılıyor...")
        mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        alarm_path = os.path.join(os.getcwd(), "alarm.wav")
        if not os.path.exists(alarm_path):
            logger.error(f"Alarm ses dosyası bulunamadı: {alarm_path}")
            return False
        alarm_sound = mixer.Sound(alarm_path)
        alarm_sound.set_volume(0.7)
        logger.info(f"Ses sistemi ve alarm sesi ({alarm_path}) başarıyla yüklendi.")
        return True
    except Exception as e:
        logger.error(f"Ses sistemi başlatılamadı: {e}")
        return False

def play_alarm():
    """Alarm sesini çalar."""
    global alarm_on, alarm_sound
    with lock:
        if alarm_on:
            logger.info("Alarm zaten çalıyor, tekrar başlatılmayacak.")
            return
        try:
            logger.info("Alarm çalmaya başlıyor...")
            alarm_sound.play(loops=-1)  # Döngüde çal
            alarm_on = True
            logger.info("Alarm çalıyor.")
        except Exception as e:
            logger.error(f"Alarm çalma hatası: {e}")
            alarm_on = False

def stop_alarm():
    """Alarm sesini durdurur."""
    global alarm_on, alarm_sound
    with lock:
        if not alarm_on:
            logger.info("Alarm zaten kapalı, durdurma yapılmayacak.")
            return
        try:
            logger.info("Alarm durduruluyor...")
            alarm_sound.stop()
            alarm_on = False
            logger.info("Alarm başarıyla durduruldu.")
        except Exception as e:
            logger.error(f"Alarm durdurma hatası: {e}")

def load_drowsiness_model():
    """Drowsiness modelini yükler."""
    global model
    try:
        logger.info("Model yükleniyor...")
        model_path = 'models/drowsiness_model.h5'
        if not os.path.exists(model_path):
            logger.error(f"Model dosyası bulunamadı: {model_path}")
            return False
        model = tf.keras.models.load_model(model_path)
        logger.info("Model başarıyla yüklendi!")
        return True
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        return False

def init_camera():
    """Kamerayı başlatır."""
    global camera
    try:
        if camera is not None:
            camera.release()
        logger.info("Kamera başlatılıyor...")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.warning("Kamera 0 açılamadı, alternatif kamera deneniyor...")
            camera = cv2.VideoCapture(1)
        if not camera.isOpened():
            logger.error("Hiçbir kamera bulunamadı!")
            return False
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        success, _ = camera.read()
        if not success:
            logger.error("Kamera frame okunamıyor!")
            camera.release()
            return False
        logger.info("Kamera başarıyla başlatıldı!")
        return True
    except Exception as e:
        logger.error(f"Kamera başlatma hatası: {e}")
        if camera is not None:
            camera.release()
        return False

def safe_read_frame():
    """Kameradan frame okur."""
    global camera
    if camera is None or not camera.isOpened():
        return False, None
    try:
        success, frame = camera.read()
        if not success:
            logger.warning("Frame okunamadı!")
        return success, frame
    except Exception as e:
        logger.error(f"Frame okuma hatası: {e}")
        return False, None

def capture_frames():
    """Kameradan frame'leri yakalar ve kuyruğa ekler."""
    global camera, frame_queue, detection_active
    while detection_active:
        if camera is None or not camera.isOpened():
            if not init_camera():
                time.sleep(1)
                continue
        success, frame = safe_read_frame()
        if not success:
            time.sleep(0.01)
            continue
        try:
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put_nowait(frame)
        except Exception as e:
            logger.error(f"Frame kuyruğu hatası: {e}")

def detect_drowsiness():
    """Gözlerin açık/kapalı durumunu tespit eder ve alarmı yönetir."""
    global output_frame, lock, alarm_on, detection_active, drowsy_start_time, total_drowsy_time, drowsy_events, frame_queue, model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    leye_cascade = cv2.CascadeClassifier('haarcascadefiles/haarcascade_lefteye_2splits.xml')
    reye_cascade = cv2.CascadeClassifier('haarcascadefiles/haarcascade_righteye_2splits.xml')
    closed_start_time = None

    while detection_active:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        left_eye = leye_cascade.detectMultiScale(gray)
        right_eye = reye_cascade.detectMultiScale(gray)

        rpred, lpred = [99], [99]

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_region = gray[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (24, 24))
            face_region = np.expand_dims(face_region.reshape(24, 24, 1) / 255.0, axis=0)
            face_pred = model.predict(face_region, verbose=0)
            if np.argmax(face_pred) == 3:  # Yawn
                cv2.putText(frame, "Yawn Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for (x, y, w, h) in right_eye:
            r_eye = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = np.expand_dims(r_eye.reshape(24, 24, 1) / 255.0, axis=0)
            rpred = model.predict(r_eye, verbose=0)
            rpred = [0] if np.argmax(rpred) == 0 else [1]
            break

        for (x, y, w, h) in left_eye:
            l_eye = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = np.expand_dims(l_eye.reshape(24, 24, 1) / 255.0, axis=0)
            lpred = model.predict(l_eye, verbose=0)
            lpred = [0] if np.argmax(lpred) == 0 else [1]
            break

        logger.info(f"Sağ göz: {rpred[0]}, Sol göz: {lpred[0]}")

        current_time = time.time()
        if rpred[0] == 0 and lpred[0] == 0:  # Her iki göz kapalı
            if closed_start_time is None:
                closed_start_time = current_time
                logger.info("Gözler kapalı, süre ölçümü başladı...")
            closed_duration = current_time - closed_start_time
            logger.info(f"Kapanık süre: {closed_duration:.2f} saniye, alarm_on: {alarm_on}")
            if closed_duration >= CLOSED_DURATION_THRESHOLD and not alarm_on:
                logger.info(f"Alarm tetiklenecek (süre: {closed_duration}s >= {CLOSED_DURATION_THRESHOLD}s)")
                play_alarm()
                drowsy_events += 1
                drowsy_start_time = current_time
                cv2.putText(frame, f"Closed > {CLOSED_DURATION_THRESHOLD}s - Alert!", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:  # En az bir göz açık
            logger.info("En az bir göz açık, alarm kontrol ediliyor...")
            closed_start_time = None
            if alarm_on:
                logger.info("Gözler açıldı, alarm durduruluyor...")
                stop_alarm()
                if drowsy_start_time:
                    total_drowsy_time += current_time - drowsy_start_time
                    drowsy_start_time = None
            cv2.putText(frame, "Open", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f'Goz Durumu: {"Kapali" if rpred[0] == 0 and lpred[0] == 0 else "Acik"}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Alarm: {"Acik" if alarm_on else "Kapali"}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        with lock:
            output_frame = frame.copy()

def generate():
    """Video akışını oluşturur."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, 'Kamera Baslatiliyor...', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', dummy_frame)
                frame_data = buffer.tobytes()
            else:
                _, buffer = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global detection_active, camera, session_start_time, model
    if not detection_active:
        if not init_sound():
            return jsonify({"status": "error", "message": "Ses sistemi başlatılamadı!"})
        if not init_camera():
            return jsonify({"status": "error", "message": "Kamera başlatılamadı!"})
        if not load_drowsiness_model():
            return jsonify({"status": "error", "message": "Model yüklenemedi!"})
        detection_active = True
        session_start_time = time.time()
        threading.Thread(target=capture_frames, daemon=True).start()
        threading.Thread(target=detect_drowsiness, daemon=True).start()
        return jsonify({"status": "success", "message": "Tespit başlatıldı"})
    return jsonify({"status": "error", "message": "Tespit zaten aktif!"})

@app.route('/stop_detection')
def stop_detection():
    global detection_active, camera, alarm_on, session_start_time
    detection_active = False
    if alarm_on:
        stop_alarm()
    if camera is not None:
        camera.release()
        camera = None
    while not frame_queue.empty():
        frame_queue.get_nowait()
    session_start_time = None
    return jsonify({"status": "success", "message": "Tespit durduruldu"})

@app.route('/status')
def get_status():
    global drowsy_events, total_drowsy_time, session_start_time
    current_session_time = int(time.time() - session_start_time) if session_start_time else 0
    return jsonify({
        "detection_active": detection_active,
        "alarm_on": alarm_on,
        "drowsy_events": drowsy_events,
        "total_drowsy_time": int(total_drowsy_time),
        "session_time": current_session_time
    })

@app.route('/reset_stats')
def reset_stats():
    global drowsy_events, total_drowsy_time
    drowsy_events = 0
    total_drowsy_time = 0
    return jsonify({"status": "success", "message": "İstatistikler sıfırlandı"})

if __name__ == '__main__':
    try:
        load_drowsiness_model()
        app.run(host='0.0.0.0', threaded=True)
    except Exception as e:
        logger.error(f"Uygulama başlatma hatası: {e}")