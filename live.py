import cv2
import numpy as np
from batch_face import RetinaFace

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.nn.functional as F
from torchvision.transforms import transforms


def live_emotion_detection(MODEL_PATH, device, emotion_labels):
    detector = RetinaFace(gpu_id=0)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 7)
    )
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Resize frame
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # Detect faces in the frame
        faces, landmarks = detector.detect(frame, threshold=0.5, scale=1.0)

        # Loop through detected faces
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            (x, y, x1, y1) = box
            roi_face = frame[y:y1, x:x1]

            # Preprocess face for emotion recognition
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(roi_face)
            input_batch = input_tensor.unsqueeze(0)

            # Predict emotion
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(input_batch)
            output = F.softmax(output, dim=1)
            emotion = emotion_labels[torch.argmax(output).item()]

            # Draw box and label on face
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

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
    MODEL_PATH = './models/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    live_emotion_detection(MODEL_PATH, device, class_dict)