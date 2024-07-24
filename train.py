import numpy as np
import pandas as pd

from denseClassifier import DenseClassifier
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

df = pd.read_csv('data/train/train_data.csv')
train_x = df[['Chan_1', 'Chan_2', 'Chan_3', 'Chan_4', 'Chan_5']].to_numpy(dtype=np.float32)
# Labels are set to be zero-indexed
train_y = df['Label'].to_numpy() - 1
train_data = list(zip(train_x, train_y))

dl = DataLoader(train_data, batch_size=64, shuffle=True)
size = len(dl.dataset)
print('Features:', np.shape(train_x))
print('Labels:', np.shape(train_y))

num_classes = 21
model = DenseClassifier(5, 21)
model.train()

loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001)

for batch, (features, labels) in enumerate(dl):
    output = model(features)
    loss = loss_fn(output, labels)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(features)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")