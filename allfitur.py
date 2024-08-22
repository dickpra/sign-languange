import tkinter as tk
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
from googletrans import Translator
from gtts import gTTS
import threading
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def open_bisindo():
    main_window.withdraw()  # Menyembunyikan jendela utama
    try:
        model_path = get_resource_path('model-video/model-bisindo-malang-vidio.p')
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        start_camera_loop(model)
        # threading.Thread(target=start_camera_loop, args=(model,)).start()
    except Exception as e:
        print(f"Error in open_bisindo: {e}")

def open_sibi():
    main_window.withdraw()  # Menyembunyikan jendela utama
    try:
        model_path = get_resource_path('model-video/model-sibi-kalimat-v1.p')
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        start_camera_loop(model)
        # threading.Thread(target=start_camera_loop, args=(model,)).start()
        # threading.Thread(target=Translatetor).start()
    except Exception as e:
        print(f"Error in open_sibi: {e}")

def open_abjadsibi():
    main_window.withdraw()  # Menyembunyikan jendela utama
    try:
        model_path = get_resource_path('model-video/modelabjadsibi.p')
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        start_camera_loop(model)
        # threading.Thread(target=start_camera_loop, args=(model,)).start()
        # threading.Thread(target=Translatetor).start()
    except Exception as e:
        print(f"Error in open_sibi: {e}")

def open_abjadmalang():
    main_window.withdraw()  # Menyembunyikan jendela utama
    try:
        model_path = get_resource_path('model-video/model-abjad-malang.p')
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        start_camera_loop(model)
        # threading.Thread(target=start_camera_loop, args=(model,)).start()
        # threading.Thread(target=Translatetor).start()
    except Exception as e:
        print(f"Error in open_sibi: {e}")


def get_resource_path(relative_path):
    """Dapatkan path ke resource file, baik dalam mode development maupun saat dibuild ke executable"""
    try:
        base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    except Exception as e:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Error in resource path: {e}")
    return os.path.join(base_path, relative_path)



def start_camera_loop(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    translator = Translator()

    last_prediction = ""
    prediction_start_time = None
    sentence = []
    translation_text = ""
    bahasa_dipilih ="en"

    def speak_text(text, lang='id'):
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(get_resource_path("assets/output.mp3"))
            os.system("afplay " + get_resource_path("assets/output.mp3"))
        except Exception as e:
            print("Error in TTS:", e)

    def translate_text():
        nonlocal translation_text
        combined_sentence = ' '.join(sentence)
        if combined_sentence:
            print("Translating to:", selected_lang.get())  # Debugging output
            translated = translator.translate(combined_sentence, dest=selected_lang.get())
            translation_text = translated.text
            translation_var.set(f'Translation: {translation_text}')

    def play_voice():
        translated_text = translation_text.replace("Translation: ", "")
        if translated_text:
            speak_text(translated_text, selected_lang.get())
            print(translation_text)

    def reset_prediction():
        nonlocal sentence, translation_text, bahasa_dipilih
        sentence = []
        translation_text = ""
        bahasa_dipilih =""
        prediction_var.set("")
        translation_var.set("Translation: ")

    def quit_application():
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()  # Menutup jendela Tkinter
        main_window.deiconify()  # Tampilkan kembali jendela utama

    root = tk.Tk()
    root.title("Sign Language Translator")

    prediction_var = tk.StringVar()
    translation_var = tk.StringVar()
    selected_lang = tk.StringVar(value="en")

    languages = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese": "zh-cn",
        "Arabic": "ar",
    }

    lang_frame = tk.Frame(root)
    lang_frame.pack(pady=10, side=tk.TOP)

    def update_lang(*args):
        nonlocal bahasa_dipilih
        bahasa_dipilih = selected_lang.get()
        print("Language selected:", bahasa_dipilih)  # Debugging output

    # Label untuk memilih bahasa
    label = tk.Label(root, text="Pilih Bahasa", font=("Helvetica", 12))
    label.pack(pady=10)

    # Menghubungkan StringVar dengan fungsi update_lang
    selected_lang.trace("w", update_lang)

    # Menu opsi bahasa
    lang_menu = tk.OptionMenu(root, selected_lang, *languages.values())
    lang_menu.pack(pady=10, side=tk.TOP)

    # Tombol Translate
    translate_button = tk.Button(root, text="Translate", command=translate_text, bg='lightblue', font=("Helvetica", 12))
    translate_button.pack(pady=10, side=tk.TOP)

    # Tombol Play Voice
    voice_button = tk.Button(root, text="Play Voice", command=play_voice, bg='lightgreen', font=("Helvetica", 12))
    voice_button.pack(pady=10, side=tk.TOP)

    # Tombol Reset Prediction
    reset_button = tk.Button(root, text="Reset Prediction", command=reset_prediction, bg='lightcoral', font=("Helvetica", 12))
    reset_button.pack(pady=10, side=tk.TOP)

    # Label untuk menampilkan hasil terjemahan
    translation_label = tk.Label(root, textvariable=translation_var, font=("Helvetica", 12), width=30)
    translation_label.pack(pady=10, side=tk.TOP)

    def camera_loop():
        nonlocal last_prediction, prediction_start_time, sentence
        try:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                return

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            predicted_character = None

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
                    # Periksa berapa banyak fitur yang diperlukan oleh model
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

                    if predicted_character == last_prediction:
                        if prediction_start_time and (time.time() - prediction_start_time) > 2:
                            sentence.append(predicted_character)
                            last_prediction = None
                            prediction_start_time = None
                    else:
                        last_prediction = predicted_character
                        prediction_start_time = time.time()

                    cv2.putText(frame, predicted_character, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    prediction_var.set(f"{' '.join(sentence)}")

            result_frame = np.zeros((H, W, 3), dtype=np.uint8)

            instruction_text = "Diam dalam gerakan tangan selama 2 detik untuk membuat sebuah kalimat"
            result_frame_pil = Image.fromarray(result_frame)
            draw = ImageDraw.Draw(result_frame_pil)

            font_path = get_resource_path('assets/font/arial.ttf')
            font2 = ImageFont.truetype(font_path, 15)
            font = ImageFont.truetype(font_path, 25)
            font3 = ImageFont.truetype(font_path, 15)
            draw.text((10, 10), instruction_text, font=font2, fill=(255, 255, 255))

            combined_sentence = ' '.join(sentence)
            draw.text((10, 100), f"Hasil: {combined_sentence}", font=font, fill=(255, 255, 255))
            draw.text((10, 280), f"Bahasa dipilih: {bahasa_dipilih}", font=font3, fill=(255, 255, 255))
            draw.text((10, 300), f"Translater: {translation_text}", font=font, fill=(255, 255, 255))

            result_frame = np.array(result_frame_pil)

            combined_frame = cv2.hconcat([frame, result_frame])

            cv2.putText(frame, "Tekan q untuk keluar app", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            combined_frame = cv2.hconcat([frame, result_frame])

            cv2.imshow('Kamera dan Hasil Isyarat', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_application()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            root.after(10, camera_loop)



    root.after(10, camera_loop)
    root.mainloop()


import tkinter as tk

def create_main_window():
    global main_window
    main_window = tk.Tk()
    main_window.title("Pilihan Bahasa Isyarat")

    # Label untuk menampilkan teks
    label = tk.Label(main_window, text="Pilih salah satu bahasa isyarat:", font=("Helvetica", 12))
    label.pack(pady=20)

    # Tombol untuk Bahasa Isyarat Bisindo Malang
    bisindo_button = tk.Button(main_window, text="Bahasa Isyarat Bisindo Malang", command=open_bisindo, width=40, height=5, bg='lightblue', font=("Helvetica", 12))
    bisindo_button.pack(pady=20)

    # Tombol untuk Bahasa Isyarat Abjad Bisindo Malang
    sibi_button = tk.Button(main_window, text="Bahasa Isyarat Abjad Bisindo Malang", command=open_abjadmalang, width=40, height=5, bg='lightblue', font=("Helvetica", 12))
    sibi_button.pack(pady=20)

    # Tombol untuk Bahasa Isyarat SIBI Kalimat
    sibi_button = tk.Button(main_window, text="Bahasa Isyarat SIBI Kalimat", command=open_sibi, width=40, height=5, bg='lightgreen', font=("Helvetica", 12))
    sibi_button.pack(pady=20)

    # Tombol untuk Bahasa Isyarat SIBI Abjad
    sibi_button = tk.Button(main_window, text="Bahasa Isyarat SIBI Abjad", command=open_abjadsibi, width=40, height=5, bg='lightgreen', font=("Helvetica", 12))
    sibi_button.pack(pady=20)

    main_window.mainloop()

create_main_window()
