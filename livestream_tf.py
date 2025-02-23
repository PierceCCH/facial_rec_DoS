import cv2
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

import time

TFLITE = True
TARGET_FACE = 'Allen'

def predict_face(img, model, interpreter, input_details, output_details):
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if (TFLITE == True):
        img_array = img_array.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

    else:
        predictions = model.predict(img_array)

    #print("Confidences:", predictions)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['Nhat', 'Pierce', 'Yaqi', 'Alp']
    return class_labels[predicted_class[0]]

def main():
    # Initialize the video capture object.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Set video width and height.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create a window to display the video.
    cv2.namedWindow("Webcam Feed", cv2.WINDOW_AUTOSIZE)
    
    interpreter = tf.lite.Interpreter(model_path = "face_recognition_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model = load_model('face_recognition_model.keras')

    totalCount = 0
    correctCount = 0
    acc = -1

    frame_count = 0
    start_time = 0

    while True:
        # Capture frame-by-frame.
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        x_mid = 0
        y_mid = 0
        side = 0

        frameCopy = frame.copy()

        for (x, y, w, h) in face:
            x_mid = x + w/2
            y_mid = y + h/2
            side = max(w, h)
            if (side < 224):
                side = 224
                cv2.rectangle(frame, (int(x_mid - side/2), int(y_mid - side/2)), (int(x_mid + side/2), int(y_mid + side/2)), (0, 255, 0), 4)

        if (len(face) != 0 and side == 224):
            faceOnly = frameCopy[int(y_mid - side/2):int(y_mid + side/2), int(x_mid - side/2):int(x_mid + side/2), :]
            try:
                pred = predict_face(faceOnly, model, interpreter, input_details, output_details)
                #print(pred)
                cv2.putText(frame, pred, (int(x_mid - side/2), int(y_mid - side/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                totalCount += 1
                if (pred == TARGET_FACE):
                    correctCount += 1
            except Exception as error:
                print('Error!!:', error)

        if (totalCount != 0):
            acc = correctCount/totalCount
            #print(f'Accuracy: {acc}')
            
        if (acc != -1):
            cv2.putText(frame, 'Accuracy: ' + str(acc), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame.
        cv2.imshow('Webcam Feed', frame)
        frame_count += 1
        
        if (frame_count == 1):
            start_time = time.time() 
        
        end_time = time.time()
        
        if (end_time != start_time):
            print('FPS:', frame_count/(end_time - start_time))

        # Press 'q' on the keyboard to exit the loop.
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture object and destroy windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()