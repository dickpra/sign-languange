from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import signal
import webbrowser
from threading import Timer
from googletrans import Translator
import sys
import time

# Fungsi untuk mendapatkan path template berdasarkan kondisi pyinstaller
def get_template_path():
    try:
        base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    except Exception as e:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Error in template path: {e}")
    return os.path.join(base_path, 'templates')

app = Flask(__name__, template_folder=get_template_path())

# Global variables
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = None
judul = "Hand Sign Recognition and Translation"
last_prediction = ""
sentence = []
camera_active = False
hasilisyarat = ""
prediction_start_time = None

def load_model(model_name):
    global model, judul
    try:
        model_path = get_resource_path(f'model/{model_name}')
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        judul = model_name.replace('model-', '').replace('.p', '').replace('-', ' ').capitalize()
    except Exception as e:
        print(f"Error in load_model: {e}")

def gen_frames():  
    global last_prediction, sentence, hasilisyarat, prediction_start_time
    translator = Translator()
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            data_aux = []
            x_ = []
            y_ = []
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []
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

                if len(results.multi_hand_landmarks) < 2:
                    data_aux.extend([0] * (42 * 2))

                if data_aux:
                    if model.n_features_in_ == 252:
                        while len(data_aux) < 252:
                            data_aux.append(0)
                    elif model.n_features_in_ == 168:
                        while len(data_aux) > 168:
                            data_aux = data_aux[:168]
                        while len(data_aux) < 168:
                            data_aux.append(0)

                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = str(prediction[0])
                    predicted_character = predicted_character.split(' ', 1)[-1]

                    current_time = time.time()

                    if predicted_character == last_prediction:
                        if prediction_start_time and (current_time - prediction_start_time) > 2:
                            if len(predicted_character) == 1:
                                sentence.append(predicted_character)
                            else:
                                sentence.append(predicted_character)
                            last_prediction = None
                            prediction_start_time = None
                    else:
                        last_prediction = predicted_character
                        prediction_start_time = current_time

                    if len(predicted_character) == 1:
                        combined_sentence = ''.join(sentence)
                    else:
                        combined_sentence = ' '.join(sentence)

                    hasilisyarat = combined_sentence
                    cv2.putText(frame, f'{predicted_character}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    print(combined_sentence)
                    print(hasilisyarat)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_camera')
def start_camera():
    global cap, camera_active
    if not camera_active:
        cap = cv2.VideoCapture(0)
        camera_active = True
    return jsonify({"status": "camera started"})

@app.route('/stop_camera')
def stop_camera():
    global cap, camera_active
    if camera_active:
        cap.release()
        camera_active = False
    return jsonify({"status": "camera stopped"})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bisindo_kalimat')
def bisindo_kalimat():
    load_model('model-bisindo-malang-vidio.p')
    return render_template('bisindo_kalimat.html')

@app.route('/bisindo_abjad')
def bisindo_abjad():
    load_model('model-abjad-malang.p')
    return render_template('bisindo_abjad.html')

@app.route('/sibi_kalimat')
def sibi_kalimat():
    load_model('model-sibi-kalimat-v1.p')
    return render_template('sibi_kalimat.html')

@app.route('/sibi_abjad')
def sibi_abjad():
    load_model('modelabjadsibi.p')
    return render_template('sibi_abjad.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_all', methods=['POST'])
def reset_all():
    global cap, camera_active, hasilisyarat
    if camera_active:
        cap.release()
        camera_active = False
    hasilisyarat = ""
    return jsonify({"status": "all reset"})

@app.route('/translate_result', methods=['POST'])
def translate_result():
    global hasilisyarat
    data = request.get_json()
    target_language = data.get("lang")

    try:
        translator = Translator()
        translation = translator.translate(hasilisyarat, dest=target_language)
        return jsonify({"translated_text": translation.text})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/reset_prediction', methods=['POST'])
def reset_prediction():
    global hasilisyarat, sentence
    sentence = []
    hasilisyarat = ""
    return jsonify({"status": "prediction reset"})

@app.route('/quit', methods=['POST'])
def quit_app():
    os.kill(os.getpid(), signal.SIGINT)
    return "Application Quit"

@app.route('/hasilisyarat')
def get_hasilisyarat():
    global hasilisyarat
    return jsonify({"hasilisyarat": hasilisyarat})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    except Exception as e:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Error in resource path: {e}")
    return os.path.join(base_path, relative_path)



if __name__ == '__main__':
    # Timer digunakan untuk membuka browser setelah server Flask siap
    Timer(1, open_browser).start()  # Delay 1 detik
    app.run(debug=True, use_reloader=False)
