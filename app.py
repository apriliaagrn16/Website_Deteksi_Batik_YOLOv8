# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import av
import cv2  # Make sure to import OpenCV
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# Local Modules
import settings
from ultralytics import YOLO
import helper

# Setting page layout
st.set_page_config(
    page_title="Batik Detection using YOLOv8",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Welcome to the BATIK Detection and Tracking Application!")

# Sidebar
with st.sidebar:
    selected = option_menu("BATIK OF IDN", ["Home", 'Detection'], 
        icons=['house', 'gear'], menu_icon="feather2", default_index=1)
    selected

if selected == "Home":
    st.write("""
        This application allows you to perform batik detection using the YOLOv8 model.

        For information: This application can only detect 15 types of batik (Batik-Bali, Batik-Betawi, Batik-Celup,
        Batik-Cendrawasih, Batik-Dayak, Batik-Geblek-Renteng, Batik-Insang, Batik-Kawung, Batik-Lasem, Batik-Megamendung,
        Batik-Pala, Batik-Parang, Batik-Poleng, Batik-Sekar-Jagad, Batik-Tambal)
        
        Use the sidebar to navigate through different functionalities:
        - **Home**: Overview of the application.
        - **Detection**: Choose a model and source to start detecting objects.
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.image("images/hasildetek.jpg", caption="Overview Image", use_column_width=True)
        
    with col2:
        st.image("images/Batikcoba1.jpeg", caption="Overview webcam", use_column_width=True)

elif selected == "Detection":
    st.sidebar.header("ML Model Config")

    confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
    model_path = Path(settings.DETECTION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Image/Webcam Config")
    source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

    source_img = None
    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image", use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image", use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects', key='detect_button'):
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image', use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")

    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)

    elif source_radio == settings.WEBCAM:
        st.header("WebRTC Object Detection")
        
        # Define the VideoTransformer class
        class VideoTransformer(VideoProcessorBase):
            def __init__(self):
                self.model = YOLO(settings.DETECTION_MODEL)
                self.confidence = 0.3
            
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
        
                # Perform object detection
                results = self.model(img)
                st.write("Results obtained")  # Debugging line
                
                # Draw bounding boxes on the frame
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].cpu().numpy()  # get box coordinates in (top, left, bottom, right) format
                        c = box.cls
                        conf = box.conf.item()
                        if conf >= self.confidence:
                            x1, y1, x2, y2 = map(int, b)
                            label = f"{self.model.names[int(c)]} {conf:.2f}"
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=VideoTransformer,
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence = confidence
        else:
            st.error("Please select a valid source type!")

    else:
        st.error("Please select a valid source type!")
else:
    st.error("Please select a valid menu option!")
