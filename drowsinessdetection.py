import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# SES VE ALARMIN HAZIRLANMASI
mixer.init()
sound = mixer.Sound('alarm.wav')

# HAARCASCADE SINIFLANDIRICILARININ YUKLENMESİ
face = cv2.CascadeClassifier('haarcascadefiles/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascadefiles/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascadefiles/haarcascade_righteye_2splits.xml')

# ETİKET VE MODELLERİN YÜKLENMESİ
lbl = ['Closed', 'Open', 'no_yawn', 'yawn']
model = load_model('models/drowsiness_model.h5')

path = os.getcwd()  # bir görüntüyü kaydetmek için kullanacağız.
cap = cv2.VideoCapture(0)  # webcam'i başlatıyoruz.
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0  # kaç kez göz tespit edildi?
thicc = 2
rpred = [99]
lpred = [99]
closed_start_time = None  # Gözlerin kapalı olduğu zamanı takip için
closed_duration_threshold = 5  # 5 saniye
is_alarm_playing = False  # Alarmın çalıp çalmadığını takip et

# webcamden görüntü alma ve işleme
while True:
    ret, frame = cap.read()  # webcamden bir kare alıyoruz.
    if not ret:
        break
    height, width = frame.shape[:2]

    # görüntüyü gri tonlamaya çeviriyoruz.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # yüz, sağ göz ve sol göz tespiti
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # ekranda bir bilgi kutusu açıyoruz.
    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # burada tespit edilen yüzzü gri bir çerçeve içine alıyoruz.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
        # Yüz bölgesini kırpıp esneme kontrolü için kullanıyoruz
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (24, 24))
        face_region = face_region / 255.0
        face_region = face_region.reshape(24, 24, -1)
        face_region = np.expand_dims(face_region, axis=0)
        face_pred = model.predict(face_region)
        face_label = np.argmax(face_pred, axis=1)[0]
        if face_label == 3:  # yawn
            cv2.putText(frame, "Yawn Detected", (10, 30), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # sağ göz işlenmesi
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)
        r_label = np.argmax(rpred, axis=1)[0]
        if r_label == 0:  # Closed
            rpred = [0]
        else:  # Open
            rpred = [1]
        break

    # sol gözün işlenmesi
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        l_label = np.argmax(lpred, axis=1)[0]
        if l_label == 0:  # Closed
            lpred = [0]
        else:  # Open
            lpred = [1]
        break

    # Gözlerin durumuna göre zaman kontrolü
    if rpred[0] == 0 and lpred[0] == 0:  # Her iki göz kapalı
        if closed_start_time is None:
            closed_start_time = time.time()
        else:
            closed_duration = time.time() - closed_start_time
            if closed_duration > closed_duration_threshold and not is_alarm_playing:
                cv2.putText(frame, "Closed > 5s - Alert!", (10, height-20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                try:
                    sound.play(-1)  # -1 ile sürekli çalmasını sağlıyoruz
                    is_alarm_playing = True
                except:
                    pass
                if thicc < 16:
                    thicc = thicc + 2
                else:
                    thicc = thicc - 2
                    if thicc < 2:
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    else:  # Gözler açık
        closed_start_time = None
        if is_alarm_playing:
            sound.stop()  # Gözler açıldığında alarmı durdur
            is_alarm_playing = False
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # görüntüyü işleme ve çıkış kontrolü
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# temizlik ve kapanış
cap.release()
cv2.destroyAllWindows()