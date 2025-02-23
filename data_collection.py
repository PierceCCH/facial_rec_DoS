import cv2
import os

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    args = parser.parse_args()

    saveName = args.name

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

    path = 'content/data/train'
    count = 0
    fileCount = 0
    dirname = saveName
    dir_path = os.path.join(path, dirname)

    if (os.path.exists(dir_path)):
        for path in os.listdir(dir_path):
            if (os.path.isfile(os.path.join(dir_path, path))):
                fileCount += 1
    else:
        os.makedirs(dir_path)
    
    saveCount = fileCount + 1

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
            if (side < 256):
                side = 256
                cv2.rectangle(frame, (int(x_mid - side/2), int(y_mid - side/2)), (int(x_mid + side/2), int(y_mid + side/2)), (0, 255, 0), 4)

        # Display the resulting frame.
        cv2.imshow('Webcam Feed', frame)
        
        if (len(face) != 0 and side == 256):
            if (count == 5):
                count = 0
                faceOnly = frameCopy[int(y_mid - side/2):int(y_mid + side/2), int(x_mid - side/2):int(x_mid + side/2), :]
                filename = os.path.join(dir_path, saveName + str(saveCount) + '.jpeg')
                cv2.imwrite(filename, faceOnly)
                print(filename)
                saveCount += 1
            count += 1
        # Press 'q' on the keyboard to exit the loop.
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture object and destroy windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()