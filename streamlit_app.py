import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from gtts import gTTS
import os
from PIL import Image

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        os.system("afplay output.mp3")  # Use 'afplay' for macOS, 'mpg321' atau 'mpg123' untuk Linux
    except Exception as e:
        st.error(f"Error in TTS: {e}")

def process_frame(frame):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
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
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        
        return frame, predicted_character

    return frame, None

st.title('Hand Sign Recognition App')

if 'run' not in st.session_state:
    st.session_state.run = False

def start():
    st.session_state.run = True

def stop():
    st.session_state.run = False

start_button = st.button('Start', on_click=start)
stop_button = st.button('Stop', on_click=stop)

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    picture = st.camera_input("Take a picture")
    if not cap.isOpened():
        st.error("Failed to open camera. Please check your camera connection and permissions.")
    else:
        stframe = st.empty()
        previous_prediction = None
        while st.session_state.run:
            ret, frame = picture.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            frame, predicted_character = process_frame(frame)

            stframe.image(frame, channels='BGR')

            if predicted_character and predicted_character != previous_prediction:
                speak_text(predicted_character)
                previous_prediction = predicted_character

        picture.release()
        cv2.destroyAllWindows()
