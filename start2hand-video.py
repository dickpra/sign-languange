import pickle
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {
    0: 'Saya',
    1: 'Makan',
    2: 'Nasi',
    3: 'Rumah',
    4: 'Kamu',
}

def speak_text(text):
    try:
        tts = gTTS(text=text, lang='id')
        tts.save("output.mp3")
        os.system("afplay output.mp3")
    except Exception as e:
        print("Error in TTS:", e)

previous_prediction = None

while True:
    try:
        data_aux = []
        x_total = []
        y_total = []

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                x_ = []
                y_ = []

                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

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

                data_aux.extend(hand_data)
                x_total.extend(x_)
                y_total.extend(y_)

            # Sesuaikan jumlah fitur agar sesuai dengan model (126 fitur)
            if len(data_aux) > 126:
                data_aux = data_aux[:126]  # Potong jika terlalu panjang
            else:
                while len(data_aux) < 126:
                    data_aux.append(0)  # Tambahkan nol jika terlalu pendek

            if data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = str(prediction[0])

                # Menghitung bounding box yang mencakup kedua tangan
                if x_total and y_total:
                    x1 = int(min(x_total) * W) - 10
                    y1 = int(min(y_total) * H) - 10
                    x2 = int(max(x_total) * W) + 10
                    y2 = int(max(y_total) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                                cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()
