import os
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torch.cuda.amp import GradScaler, autocast
import torch.utils.tensorboard as tb
from sklearn.metrics import f1_score

from utils.dataloader import get_dataloaders


def train(train_loader, val_loader, model_path, device, epochs, val_epoch, patience, dropout=0, gamma=0.00001, reg=True, lr=0.003):
    now = datetime.datetime.now()
    log_dir = f'./logs/{now.strftime("%Y-%m-%d_%H-%M-%S")}_mobilenetv3small_lr{lr}_dr{dropout}{"_gamma"+str(gamma)+"_reg" if reg else ""}'
    model_path = os.path.join(model_path, f'mobilenetv3_{now.strftime("%Y-%m-%d_%H-%M-%S")}.pt')

    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT, dropout=dropout)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 7)
    model = model.to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    scaler = GradScaler()
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    writer = tb.SummaryWriter(log_dir)

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader):
            inputs, labels, distribution = data
            inputs, labels, distribution = inputs.to(device), labels.to(device), distribution.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.log_softmax(dim=1), distribution)
                if reg:
                    loss += gamma * torch.norm(model.classifier[3].weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        writer.add_scalar('training_loss', running_loss / len(train_loader), epoch)
        tqdm.write(f"Epoch {epoch+1} training loss: {running_loss / len(train_loader):.3f}")


        # Validation every val_epoch epochs
        if epoch % val_epoch != 0: continue

        val_loss = 0.0
        val_loss2 = 0.0
        val_f1 = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels, distribution = data
                inputs, labels, distribution = inputs.to(device), labels.to(device), distribution.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.log_softmax(dim=1), distribution)
                    loss2 = criterion2(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

                val_loss += loss.item()
                val_loss2 += loss2.item()
                val_f1 += f1

        val_loss /= len(val_loader)
        val_loss2 /= len(val_loader)
        val_f1 /= len(val_loader)
        writer.add_scalar('validation_KL', val_loss, epoch)
        writer.add_scalar('validation_CE', val_loss2, epoch)
        writer.add_scalar('validation_f1', val_f1, epoch)

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += val_epoch

        if epochs_without_improvement >= patience:
            print(f'Early stopping: validation f1 has not improved in {patience} epochs.')
            break

    writer.close()


if __name__ == '__main__':
    DATA_DIR = './data/RAF/Image/aligned/'
    TRAIN_LABELS_PATH = './data/RAF/EmoLabel/emo_label_train.csv'
    TEST_LABELS_PATH = './data/RAF/EmoLabel/emo_label_test.csv'
    TRAIN_DISTRIBUTION_PATH = './data/RAF/EmoLabel/emo_distribution_train.csv'
    TEST_DISTRIBUTION_PATH = './data/RAF/EmoLabel/emo_distribution_test.csv'
    MODEL_PATH = './models/'

    class_dict = {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral"
    }

    n_classes = len(class_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, TRAIN_LABELS_PATH, TEST_LABELS_PATH, TRAIN_DISTRIBUTION_PATH, TEST_DISTRIBUTION_PATH, batch_size=256, num_workers=4)

    train(train_loader, val_loader, MODEL_PATH, device, epochs=1000, val_epoch=10, patience=100)