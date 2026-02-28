import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model

DATA_DIR = "data/"
EPOCHS = 10
BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, test_loader = get_dataloaders(
    DATA_DIR,
    BATCH_SIZE
)

num_classes = len(train_loader.dataset.classes)

model = get_model(num_classes)

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(EPOCHS):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

torch.save(
    model.state_dict(),
    "models/model.pth"
)

print("Training finished")
