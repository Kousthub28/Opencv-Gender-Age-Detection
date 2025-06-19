from flask import Flask, render_template, Response
import cv2
import simpleaudio as sa
import time
import threading

app = Flask(__name__)

# Load models
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "deploy_gender.prototxt")
age_net = cv2.dnn.readNet("age_net.caffemodel", "deploy_age.prototxt")

# Constants
GENDERS = ['Male', 'Female']
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(25-32)', '(38-43)', '(48-53)', '(60-100)']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
male_audio = sa.WaveObject.from_wave_file("male_voice.wav")
female_audio = sa.WaveObject.from_wave_file("female_voice.wav")

spoken = False
last_spoken_time = 0
cooldown = 5

def play_voice_async(gender):
    if gender == 'Male':
        male_audio.play().wait_done()
    else:
        female_audio.play().wait_done()

def generate_frames():
    global spoken, last_spoken_time
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                spoken = False

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w].copy()

                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                             (78.426337, 87.768914, 114.895847), swapRB=False)

                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDERS[gender_preds[0].argmax()]

                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_GROUPS[age_preds[0].argmax()]

                label = f"{gender}, Age: {age}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Play voice
                current_time = time.time()
                if not spoken or (current_time - last_spoken_time) > cooldown:
                    threading.Thread(target=play_voice_async, args=(gender,), daemon=True).start()
                    spoken = True
                    last_spoken_time = current_time

                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
