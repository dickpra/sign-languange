import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

DATA_DIR = 'data-vidio-sibi-kalimat'
OUTPUT_PICKLE = 'data_video.pickle'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

data = []
labels = []

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            if video_path.endswith(('.mp4', '.avi', '.mov')):  # Cek jika file adalah video
                cap = cv2.VideoCapture(video_path)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    data_aux = []
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            hand_data = []
                            x_ = []
                            y_ = []

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y

                                x_.append(x)
                                y_.append(y)

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                hand_data.append(x - min(x_))
                                hand_data.append(y - min(y_))

                            while len(hand_data) < 42 * 2:
                                hand_data.append(0)

                            data_aux.extend(hand_data)

                        # Pastikan jika hanya ada satu tangan, kita tambahkan placeholder untuk tangan kedua
                        if len(results.multi_hand_landmarks) < 2:
                            data_aux.extend([0] * (42 * 2))

                        data.append(data_aux)
                        labels.append(class_dir)  # Label sesuai dengan nama folder

                    frame_count += 1

                cap.release()

# Menentukan panjang maksimal dari data
max_length = max(len(d) for d in data)

# Menambahkan padding pada data yang lebih pendek
data_padded = [d + [0] * (max_length - len(d)) for d in data]

with open(OUTPUT_PICKLE, 'wb') as f:
    pickle.dump({'data': data_padded, 'labels': labels}, f)
