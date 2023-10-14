import cv2
import numpy as np

import torch
import torch.nn.functional as F

class GradCAM():
    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.model = model

    def get_heatmap(self, img):
        if len(img.shape) == 3:
            img = img.unsqueeze(0).to(self.device)
        else:
            img = img.to(self.device)

        features = self.model.features(img)
        adaptive_pool = self.model.avgpool(features)
        output = self.model.classifier(adaptive_pool.view(features.shape[0], -1))
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(F.softmax(output, dim=1)).item()
        _, idx = torch.max(output, dim=1)
        idx = idx.item()

        # Get the gradients of the output with respect to the last convolutional layer
        self.model.zero_grad()
        output[:, idx].backward()
        gradients = self.model.features[-1][0].weight.grad
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Get the activations of the last convolutional layer
        activations = features

        # Weight the channels by their corresponding gradients
        for i in range(gradients.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # Resize the heatmap to the size of the input image
        heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))

        # Convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap, prediction, confidence

