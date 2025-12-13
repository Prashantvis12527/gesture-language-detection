import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "gesture_model.pkl"), "rb") as f:
    model, encoder = pickle.load(f)

# ================= MEDIAPIPE SETUP =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting Gesture Detection...")
print("Press Q to quit")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
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

                # -------- Feature extraction --------
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                features = np.array(features).reshape(1, -1)

                # -------- Prediction + confidence --------
                probs = model.predict_proba(features)[0]
                pred_index = np.argmax(probs)

                gesture = encoder.inverse_transform([pred_index])[0]
                confidence = probs[pred_index] * 100

                # -------- Confidence threshold --------
                if confidence > 70:
                    text = f"{gesture} ({confidence:.2f}%)"
                    color = (0, 255, 0)
                else:
                    text = "Detecting..."
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    color,
                    3
                )

        cv2.imshow("Gesture Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()