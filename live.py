import time
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.transforms import transforms

from face_detection.detector import RetinaFace
from utils.gradcam import GradCAM


def live_emotion_detection(MODEL_PATH, device, emotion_labels):
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    # model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 7)
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    NUM_FRAMES = 5
    frame_count = 0
    fps = 0
    start_time = time.time()

    detector = RetinaFace(device=device)
    gradcam = GradCAM(model, device)

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        start_time_face = time.time()
        faces = detector.detect(frame)
        end_time_face = time.time()
        time_taken_ms_face = (end_time_face - start_time_face) * 1000

        # Loop through detected faces
        for i in faces:
            face, landmarks, certainty = i
            box = face.astype(np.int32)
            (x, y, x1, y1) = box
            x = max(0, x)
            y = max(0, y)
            x1 = min(frame.shape[1], x1)
            y1 = min(frame.shape[0], y1)

            roi_face = frame[y:y1, x:x1]

            # Preprocess face for emotion recognition
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(roi_face)
            input_tensor = input_tensor.to(device)

            # Get gradcam heatmap and overlay on face
            start_time_emo = time.time()
            heatmap, emotion_idx, confidence = gradcam.get_heatmap(input_tensor)
            end_time_emo = time.time()
            time_taken_ms_emo = (end_time_emo - start_time_emo) * 1000

            heatmap = cv2.resize(heatmap, (roi_face.shape[1], roi_face.shape[0]))
            heatmap[heatmap < 0.5] = frame[y:y1, x:x1][heatmap < 0.5]
            frame[y:y1, x:x1] = cv2.addWeighted(roi_face, 0.7, heatmap, 0.3, 0)

            # Draw box and label on face
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, f"{emotion_labels[emotion_idx]} ({confidence:.2f})", (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Update frame count and check if benchmarking is complete
        frame_count += 1
        if frame_count == NUM_FRAMES:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = NUM_FRAMES / elapsed_time
            frame_count = 0
            start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # print(f"Time taken emotion is {time_taken_ms_emo:.2f} ms and face: {time_taken_ms_face:.2f} ms")

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    class_dict = {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral"
    }
    MODEL_PATH = './models/mobilenetv3_final.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    live_emotion_detection(MODEL_PATH, device, class_dict)