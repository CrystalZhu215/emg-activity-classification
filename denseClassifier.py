import torch.nn as nn

class DenseClassifier(nn.Module):

    def __init__(self, feat_in, feat_out, feat_hidden=30):
        super(DenseClassifier, self).__init__()

        self.fc1 = nn.Linear(feat_in, feat_hidden)
        self.fc2 = nn.Linear(feat_hidden, feat_out)

    def forward(self, x):
        f1 = nn.functional.softmax(self.fc1(x), dim=0)
        output = self.fc2(f1)
        return output
