import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json
import tempfile


def load_model():
    # Load the emotion detection model from disk
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model.h5")
    return emotion_model


def detect_emotions(frame, emotion_model):
    # Create the face detector
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]

        # Draw the bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Create the Streamlit app


def main():
    st.title('Emotion Detection')

    # Create the sidebar
    st.sidebar.title('Options')

    # Choose video source: Webcam or Upload
    video_source = st.sidebar.selectbox('Video Source', ['Webcam', 'Upload'])

    # Load the emotion detection model
    emotion_model = load_model()

    if video_source == 'Webcam':
        st.sidebar.write('Emotion Detection from Webcam')

        # Start the webcam feed
        cap = cv2.VideoCapture(0)

        # Display the webcam feed in the main body
        video_placeholder = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = detect_emotions(frame, emotion_model)

            video_placeholder.image(processed_frame, channels='BGR')

        cap.release()

    elif video_source == 'Upload':
        uploaded_file = st.sidebar.file_uploader(
            'Upload a video file', type=['mp4', 'mov'])

        if uploaded_file is not None:
            st.sidebar.write('Emotion Detection from Uploaded Video')

            # Save the uploaded video file locally
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())

            # Start the video capture
            cap = cv2.VideoCapture(temp_file.name)

            # Display the video with emotions in the main body
            video_placeholder = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = detect_emotions(frame, emotion_model)

                video_placeholder.image(processed_frame, channels='BGR')

            cap.release()

            # Remove the temporary file
            temp_file.close()
            temp_file.unlink()


if __name__ == '__main__':
    main()
