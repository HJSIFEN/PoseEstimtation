import torch.nn as nn
import torch

class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, A):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.A = A 
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, X):
        return self.fc(torch.matmul(self.A, X))


class GCN(nn.Module):
    def __init__(self, num_feature, num_class, A):
        super(GCN, self).__init__()

        self.input = nn.Sequential(
            GCN_layer(num_feature, 6, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
        )
        self.gcn1 = nn.Sequential(
            GCN_layer(6, 16, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            GCN_layer(16, 16, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            GCN_layer(16, 32, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
        )
        self.gcn2 = nn.Sequential(
            GCN_layer(32, 64, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            GCN_layer(64, 64, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            GCN_layer(64, 128, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
        )
        self.gcn3 = nn.Sequential(
            GCN_layer(128, 256, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            GCN_layer(256, 256, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            GCN_layer(256, 512, A),
            nn.BatchNorm1d(18),
            nn.ReLU(),
        )
                
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*18, 500),
            nn.Dropout(0.2),
            nn.Linear(500, 100),
            nn.Dropout(0.2),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.output(x)
        return x
