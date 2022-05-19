from datasets import SkeletonDataset
import torch
from torch.utils.data import DataLoader
from model import GCN
import torch.nn as  nn
from torchvision import transforms

model = []
loss_fn = []

test_dir = "C:\\Users\\user\\Desktop\\Pose_test"
test_dir = SkeletonDataset(test_dir)
test_loader = DataLoader(test_dir)
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
test_loop(test_loader, model, loss_fn)