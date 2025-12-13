# ---------- Import required libraries ----------
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# ---------- Create Flask app ----------
app = Flask(__name__)

# ---------- Load trained gesture model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "gesture_model.pkl"), "rb") as f:
    model, encoder = pickle.load(f)

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------- Open webcam ----------
cap = cv2.VideoCapture(0)

# ---------- Generator function for video streaming ----------
def generate_frames():
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # ----- Feature extraction -----
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])

                    features = np.array(features).reshape(1, -1)

                    # ----- Prediction -----
                    probs = model.predict_proba(features)[0]
                    idx = np.argmax(probs)

                    gesture = encoder.inverse_transform([idx])[0]
                    confidence = probs[idx] * 100

                    # ----- Display result -----
                    cv2.putText(
                        frame,
                        f"{gesture} ({confidence:.2f}%)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            # Convert frame to JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # Stream frame to browser
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ---------- Route for homepage ----------
@app.route("/")
def index():
    return render_template("index.html")

# ---------- Route for video feed ----------
@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ---------- Run Flask app ----------
if __name__ == "__main__":
    app.run(debug=True)


