import cv2
import datetime

def main():
    # Attempt to open the video capture
    cap = cv2.VideoCapture(0)  # Use index 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Load the Haar cascades for face and smile detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        original_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face_roi = gray[y:y+h, x:x+w]

            # Detect smiles within the face ROI
            smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.3, minNeighbors=25)
            for (x1, y1, w1, h1) in smiles:
                cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w1, y + y1 + h1), (0, 0, 255), 2)

                # Capture and save the image with a timestamped filename
                time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                file_name = f'selfie-{time_stamp}.png'
                cv2.imwrite(file_name, original_frame)
                print(f"Captured {file_name}")

                # Optional: Uncomment the next line if you want a brief pause after each capture
                # cv2.waitKey(1000)

        # Display the frame with overlays
        cv2.imshow('Cam Star', frame)
        if cv2.waitKey(10) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
