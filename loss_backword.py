import torch
import torch.nn as nn
from torch.optim import Adam

net = nn.Linear(1, 1, bias=False)

#net.weight.data.fill_(2.0)
net.weight.data = torch.tensor([[2.0]])

optimizer = Adam(net.parameters(), lr=0.1)

x = torch.tensor([[1.0]])
y_true = torch.tensor([[5.0]])

y_pred = net(x)
loss = (y_pred - y_true).pow(2).mean()
print('Initial loss:', loss.item())

optimizer.zero_grad()
y_pred = net(x) 
# Manually compute the gradient for verification
# loss = (y_pred - y_true)^2
# Let net(x) = net.weight * x = 2.0 * 1.0 = 2.0
# So,
# loss(x) = (net(x) - y_true)^2
# loss'(x) = 2 * (net(x) - y_true) * net'(x)
#           = 2 * (1*net.weight - y_true) * 1
#           = 2 * (2.0 - 5.0) * 1
#           = -6.0
loss.backward()
print('Gradient:', net.weight.grad.item())  

optimizer.step()
print('Updated weight:', net.weight.data.item())

y_pred = net(x)
print('New prediction:', y_pred.item())