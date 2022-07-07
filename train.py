import torch
from torch.utils.data import DataLoader
from model import GCN
import torch.nn as  nn
from torchvision import transforms

_A =\
    [
        [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], #11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1], #12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], #13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0], #14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], #15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], #16
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]  #17
    ]
    

def train(model, train_dataset, valid_datset, model_name="GCN", epochs = 50,batch_size= 10, learning_rate = 1e-3, early_stop_cnt=5):
    import os

    if not os.path.isdir(f"exp/{model_name}"):
        os.mkdir(f"exp/{model_name}")

    train_loader = DataLoader(train_dataset, batch_size = 10, shuffle= True)
    valid_loader = DataLoader(valid_datset, batch_size = 10, shuffle=True)
    
 
    A = torch.tensor(_A).float()   
    print(f"learning_rate : {learning_rate}")
    print(f"batch_size : {batch_size}")
    print(f"epochs : {epochs}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_val_loss = 999999
    early_stop = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------------------")
        #def train_loop()
        size = len(train_loader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            
            pred = model(X)
            loss=  loss_fn(pred,y)

            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * X.shape[0]
                print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_batch, (X, y) in enumerate(valid_loader):
                pred = model(X)
                val_loss += loss_fn(pred, y)
            print(f'val_loss: {val_loss:>7f}')
            if best_val_loss > val_loss:
                early_stop = 0
                best_val_loss = val_loss
                print(f"Save best model : {best_val_loss} > {val_loss}")
                torch.save(model.state_dict(), f"exp/{model_name}/BestModel.pth")
            else :
                early_stop += 1

        
        if early_stop > early_stop_cnt :
            print('early stop')
            break

        print("---------------------------------")
    print("Done!")
    torch.save(model.state_dict(), "exp/{model_name}/fianl_gcn.pth")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        
        pred = model(X)
        loss=  loss_fn(pred,y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

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

