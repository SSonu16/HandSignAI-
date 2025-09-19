"""
Sign Language → Text (Starter)
Single-file Python app with three modes:
 1) collect  : collect landmark samples for labels and save to dataset.npz
 2) train    : train a KNN classifier on the saved dataset and save model.joblib
 3) run      : run real-time recognition from webcam and display text overlay

Usage examples:
  python sign_language_to_text_starter.py collect
  python sign_language_to_text_starter.py train
  python sign_language_to_text_starter.py run

Dependencies:
  pip install opencv-python mediapipe scikit-learn joblib numpy

Notes:
 - This starter uses Mediapipe hand landmarks (21 points) as numeric features.
 - You will need to collect samples for each label you want (e.g., "hello", "thanks", "A").
 - Keep the camera framing consistent while collecting.

"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

DATA_PATH = "dataset.npz"
MODEL_PATH = "model.joblib"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(results):
    """Return a 63-length vector (x,y,z for 21 landmarks) or None"""
    if not results.multi_hand_landmarks:
        return None
    # Use first detected hand
    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(landmarks):
    """Normalize landmarks: translate wrist to origin and scale by max distance."""
    # landmarks: (63,) -> reshape (21,3)
    pts = landmarks.reshape(-1, 3)
    # wrist is landmark 0
    wrist = pts[0]
    pts -= wrist
    # scale
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist
    return pts.flatten()


def collect_mode():
    print("== COLLECT MODE ==")
    print("Instructions: Choose/enter a label (word). Press SPACE to capture a sample for that label.")
    print("Press 'q' to quit and save. Keep camera framing consistent. Collect 50+ samples per label for better accuracy.")

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    X = []
    y = []

    current_label = input("Enter label to collect samples for (e.g., hello): ").strip()
    if not current_label:
        print("No label entered. Exiting.")
        return

    samples_for_label = 0
    print(f"Collecting for label: '{current_label}'. Press SPACE to record a sample, 'n' to change label, 'q' to quit and save.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        if res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {current_label} | Samples: {samples_for_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Collect - Press SPACE to save sample", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # change label
            new_label = input("Enter new label: ").strip()
            if new_label:
                current_label = new_label
                samples_for_label = 0
        elif key == 32:  # SPACE
            lm = extract_landmarks(res)
            if lm is None:
                print("No hand detected. Try again.")
            else:
                lm = normalize_landmarks(lm)
                X.append(lm)
                y.append(current_label)
                samples_for_label += 1
                print(f"Saved sample #{samples_for_label} for label '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()

    # Save to dataset
    if os.path.exists(DATA_PATH):
        print("Appending to existing dataset.npz")
        existing = np.load(DATA_PATH, allow_pickle=True)
        X_old = existing['X'].tolist()
        y_old = existing['y'].tolist()
        X_all = X_old + X
        y_all = y_old + y
    else:
        X_all = X
        y_all = y

    np.savez(DATA_PATH, X=np.array(X_all, dtype=object), y=np.array(y_all, dtype=object))
    print(f"Saved dataset to {DATA_PATH}. Total samples: {len(X_all)}")


def train_mode():
    print("== TRAIN MODE ==")
    if not os.path.exists(DATA_PATH):
        print("No dataset found. Run collect mode first.")
        return
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']

    # Convert object arrays to numeric 2D array
    X_num = np.stack([np.array(x, dtype=np.float32) for x in X])
    y_num = np.array(y)

    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_num, y_num, test_size=0.2, random_state=42, stratify=y_num)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    dump(clf, MODEL_PATH)
    print(f"Trained KNN model and saved to {MODEL_PATH}. Test accuracy: {acc:.2f}")


def run_mode():
    print("== RUN MODE ==")
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run train mode first.")
        return
    clf = load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    recent_predictions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        pred_text = ""
        if res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            lm = extract_landmarks(res)
            if lm is not None:
                lm = normalize_landmarks(lm)
                try:
                    pred = clf.predict([lm])[0]
                    recent_predictions.append(pred)
                    # keep short history and take majority to stabilize
                    if len(recent_predictions) > 10:
                        recent_predictions.pop(0)
                    # majority vote
                    pred_text = max(set(recent_predictions), key=recent_predictions.count)
                except Exception as e:
                    pred_text = "?"
        else:
            # no hand
            recent_predictions = []

        # Display prediction on screen
        cv2.rectangle(frame, (0,0), (frame.shape[1], 40), (0,0,0), -1)
        cv2.putText(frame, f"Prediction: {pred_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("Sign Language -> Text", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sign_language_to_text_starter.py [collect|train|run]")
        sys.exit(1)
    mode = sys.argv[1].lower()
    if mode == 'collect':
        collect_mode()
    elif mode == 'train':
        train_mode()
    elif mode == 'run':
        run_mode()
    else:
        print("Unknown mode. Use collect, train, or run.")
