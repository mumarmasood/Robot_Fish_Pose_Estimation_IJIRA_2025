# open camera and detect faces
import cv2
import numpy as np
import face_recognition
import os
import time
import sys

# inintialize the camera
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# recongize any human faces in the image
def recognize_faces(frame):
    # Convert the image from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    return face_locations, face_encodings


# call the function to recognize faces
def main():
    while True:
        # Capture a single frame from the camera
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture image")
            break

        # Call the function to recognize faces
        face_locations, face_encodings = recognize_faces(frame)

        # Draw rectangles around the detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        video_capture.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        video_capture.release()
        cv2.destroyAllWindows()
        sys.exit(1)

