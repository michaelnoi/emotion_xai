from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import FaceDataset


def get_dataloaders(data_dir, train_labels_path, test_labels_path, train_distribution_path, test_distribution_path, batch_size=32, num_workers=8):
    train_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    raf_facial_emotion_train = FaceDataset(train_labels_path, train_distribution_path, data_dir, split='train', transform=train_transform)
    raf_facial_emotion_val = FaceDataset(train_labels_path, train_distribution_path, data_dir, split='val', transform=val_transform)
    raf_facial_emotion_test = FaceDataset(test_labels_path, test_distribution_path, data_dir, split='test', transform=val_transform)

    train_dataloader = DataLoader(raf_facial_emotion_train, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(raf_facial_emotion_val, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(raf_facial_emotion_test, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


