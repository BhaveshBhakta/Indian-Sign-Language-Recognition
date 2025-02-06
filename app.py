# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load trained model
# model = load_model('action.keras')
# actions = ['hello', 'thanks', 'yes', 'no', 'stop']
# sequence = []
# threshold = 0.8

# # MediaPipe setup
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# def extract_keypoints(results):
#     """Extract keypoints from the MediaPipe detection results."""
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
#     return np.concatenate([pose, face, lh, rh])

# def prob_viz(res, actions, input_frame):
#     """Visualize prediction probabilities."""
#     output_frame = input_frame.copy()
#     colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0), (0, 255, 255)]
    
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 200), 90 + num * 40), colors[num], -1)
#         cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
#     return output_frame

# def generate_frames():
#     """Capture frames, detect gestures, and return processed frames."""
#     global sequence
#     cap = cv2.VideoCapture(0)

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         try:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Convert image format
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False
#                 results = holistic.process(image)
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#                 # Draw landmarks
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#                 mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
#                 mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#                 mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

#                 keypoints = extract_keypoints(results)
                
#                 # Avoid appending empty keypoints
#                 if np.any(keypoints):
#                     sequence.append(keypoints)
#                     sequence = sequence[-30:]  # Maintain last 30 frames

#                 if len(sequence) == 30:
#                     input_sequence = np.expand_dims(sequence, axis=0)
#                     res = model.predict(input_sequence)[0]

#                     if res[np.argmax(res)] > threshold:
#                         detected_action = actions[np.argmax(res)]
#                         print(f"Detected Gesture: {detected_action}")
                    
#                     image = prob_viz(res, actions, image)

#                 # Encode the frame for web streaming
#                 ret, buffer = cv2.imencode('.jpg', image)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#         except Exception as e:
#             print(f"Error: {e}")

#         finally:
#             cap.release()
#             cv2.destroyAllWindows()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model('action.keras')
actions = ['hello', 'thanks', 'yes', 'no', 'stop']
sequence = []
threshold = 0.8

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Custom drawing styles to make the face detection less intrusive
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

def extract_keypoints(results):
    """Extract keypoints from the MediaPipe detection results."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame):
    """Visualize prediction probabilities."""
    output_frame = input_frame.copy()
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0), (0, 255, 255)]
    
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 200), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output_frame

def generate_frames():
    """Capture frames, detect gestures, and return processed frames."""
    global sequence
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert image format
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks (excluding full face mesh)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec)

                # **Optional:** Draw only the face outline instead of full face mesh
                if results.face_landmarks:
                    for i in range(0, 468, 10):  # Reduce face points to prevent covering face
                        landmark = results.face_landmarks.landmark[i]
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                keypoints = extract_keypoints(results)
                
                # Avoid appending empty keypoints
                if np.any(keypoints):
                    sequence.append(keypoints)
                    sequence = sequence[-30:]  # Maintain last 30 frames

                if len(sequence) == 30:
                    input_sequence = np.expand_dims(sequence, axis=0)
                    res = model.predict(input_sequence)[0]

                    if res[np.argmax(res)] > threshold:
                        detected_action = actions[np.argmax(res)]
                        print(f"Detected Gesture: {detected_action}")
                    
                    image = prob_viz(res, actions, image)

                # Encode the frame for web streaming
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
