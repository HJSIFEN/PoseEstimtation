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

        self.feature_extractor = nn.Sequential(
            GCN_layer(num_feature, 4, A),
            nn.ReLU(),
            GCN_layer(4, 8, A),
            nn.ReLU(),
            GCN_layer(8, 16, A),
            nn.ReLU(),
            GCN_layer(16, 16, A),
            nn.ReLU(),
            GCN_layer(16, 16, A),
            nn.ReLU(),
            GCN_layer(16, 32, A),
            nn.ReLU(),
            GCN_layer(32, 32, A),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*18, num_class),
        )

    def forward(self, X):
        return self.feature_extractor(X)