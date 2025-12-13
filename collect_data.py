import cv2
import mediapipe as mp
import numpy as np
import os

# ================= CONFIG =================
GESTURES = ["hello", "yes", "no", "thankyou"]
SAMPLES_PER_GESTURE = 200
# =========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

X = []
y = []

print("Gesture Data Collection")
print("Press ENTER to start collecting each gesture")
print("Press Q to quit camera window\n")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    for label in GESTURES:
        input(f"👉 Press ENTER to start gesture: '{label}'")

        print(f"Collecting samples for: {label}")
        count = 0

        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                X.append(landmarks)
                y.append(label)
                count += 1

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(
                frame,
                f"{label}: {count}/{SAMPLES_PER_GESTURE}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collecting Gesture Data", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()

# ================= SAVE DATA =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X = np.array(X)
y = np.array(y)

print("\nTotal samples collected:", len(X))

if len(X) == 0:
    print("❌ No data collected.")
else:
    np.save(os.path.join(BASE_DIR, "X.npy"), X)
    np.save(os.path.join(BASE_DIR, "y.npy"), y)
    print("✅ X.npy and y.npy saved in:", BASE_DIR)