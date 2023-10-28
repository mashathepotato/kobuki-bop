import cv2
import musicalbeeps


time_in_correct_position = 0

def play_melody():
    player = musicalbeeps.Player(volume = 0.3,
                            mute_output = False)
    player.play_note("C", .5)
    player.play_note("E", .5)
    player.play_note("G", .5)

def image_capture():
    global time_in_correct_position

    # Initialize the camera and cascade classifiers
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    is_correct_position = False
    time_in_correct_position = 0

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to access the camera.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.4, minNeighbors=25, minSize=(30, 30))
        eye_glasses = eye_glasses_cascade.detectMultiScale(gray_frame, scaleFactor=1.4, minNeighbors=8, minSize=(30, 30))

        for (egx, egy, egw, egh) in eye_glasses:
            cv2.rectangle(frame, (egx, egy), (egx+egw, egy+egh), (0, 165, 255), 1)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 1)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

            if len(smiles) > 0:
                is_correct_position = True
                time_in_correct_position += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 0, 0), 1)
            else:
                is_correct_position = False
                time_in_correct_position = 0
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Camera", frame)
        if time_in_correct_position >= 15: 
            play_melody()
            break

        if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release the camera and close OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_capture()
