import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.losses import sparse_categorical_crossentropy

import time

TFLITE = True
USE_MASK = False
SHOW_PGD = True
class_labels = ['Nhat', 'Pierce', 'Yaqi', 'Alp']

TARGET_FACE = 'Pierce'

def predict_face_attack(img, model):
    img = tf.expand_dims(img, 0)
    predictions = model(img)

    predicted_class = tf.argmax(predictions, 1)
    return predictions

def predict_face(img, model, interpreter, input_details, output_details):
    img = tf.expand_dims(img, 0)

    if (TFLITE == True):
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

    else:
        predictions = model(img)

    predicted_class = tf.argmax(predictions, 1)
    return predicted_class[0], class_labels[predicted_class[0]], predictions

def pgd_attack(model, x, y, pretrain = False, eps=0.3, alpha=0.007, iters=8):
    if pretrain == False:
        x_adv = tf.Variable(initial_value=x/255, trainable=True) # Initialize adversarial example as a copy of the input
    
        for i in range(iters):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x_adv)
                prediction = predict_face_attack(x_adv, model)
                loss = sparse_categorical_crossentropy(y, prediction)

            gradient = tape.gradient(loss, x_adv)
            x_adv = x_adv + alpha * tf.sign(gradient)
            x_adv = tf.clip_by_value(x_adv, x/255 - eps, x/255 + eps)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    
        x_adv = x_adv*255
    else:
        img_path = './masks/' + class_labels[y] + '_mask.jpeg'
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        x_adv = x + img_tensor
    return x_adv

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
            faceOnly = tf.convert_to_tensor(faceOnly, dtype=tf.float32)
            try:
                pred, _, _ = predict_face(faceOnly, model, interpreter, input_details, output_details)
                x_adv = pgd_attack(model, faceOnly, pred, USE_MASK)
                _, new_pred, _ = predict_face(x_adv, model, interpreter, input_details, output_details)
                
                if SHOW_PGD:
                    frame[int(y_mid - side/2):int(y_mid + side/2), int(x_mid - side/2):int(x_mid + side/2), :] = x_adv.numpy()
                cv2.putText(frame, new_pred, (int(x_mid - side/2), int(y_mid - side/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                totalCount += 1
                if (new_pred == TARGET_FACE):
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