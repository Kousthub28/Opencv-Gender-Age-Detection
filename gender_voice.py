import cv2
import simpleaudio as sa
import time
import threading

# Load pre-trained models
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "deploy_gender.prototxt")
age_net = cv2.dnn.readNet("age_net.caffemodel", "deploy_age.prototxt")

# Labels
GENDERS = ['Male', 'Female']
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load audio
male_audio = sa.WaveObject.from_wave_file("male_voice.wav")
female_audio = sa.WaveObject.from_wave_file("female_voice.wav")

# Start webcam
cap = cv2.VideoCapture(0)

spoken = False
last_spoken_time = 0
cooldown = 5  # seconds between voice playbacks


# Function to play sound in a separate thread
def play_voice_async(gender):
    if gender == 'Male':
        play_obj = male_audio.play()
    else:
        play_obj = female_audio.play()
    play_obj.wait_done()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        spoken = False  # reset flag

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()

        # Prepare blob
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426337, 87.768914, 114.895847), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDERS[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_GROUPS[age_preds[0].argmax()]

        # Draw results
        label = f"{gender}, Age: {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Voice playback (non-blocking)
        current_time = time.time()
        if not spoken or (current_time - last_spoken_time) > cooldown:
            threading.Thread(target=play_voice_async, args=(gender,), daemon=True).start()
            spoken = True
            last_spoken_time = current_time

        break  # process only first face

    cv2.imshow("Gender and Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
