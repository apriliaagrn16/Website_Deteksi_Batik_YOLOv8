from ultralytics import YOLO
import time
import streamlit as st
import cv2
import threading

import settings

def load_model(model_path):
    model = YOLO(model_path)
    return model

def _display_detected_frames(conf, model, st_frame, image):
    image = cv2.resize(image, (320, int(320 * (9 / 16))))  # Mengurangi resolusi gambar lebih lanjut
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            
            def process_frame():
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf, model, st_frame, image)
                        time.sleep(0.1)  # Batasi FPS (10 frame per detik)
                    else:
                        vid_cap.release()
                        break

            thread = threading.Thread(target=process_frame)
            thread.start()
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

