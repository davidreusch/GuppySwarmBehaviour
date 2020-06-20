import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
inputs = [torch.randn(1, 3) for _ in range(5)]

print(inputs)
print(inputs[0].size())

lstm = nn.LSTM(3,4)

hidden = (torch.randn(1, 1, 4),
          torch.randn(1, 1, 4))

for i in inputs:
    print(i.view(1,1,-1))

for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    
"""
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print(inputs.size())
hidden = (torch.randn(1, 1, 4), torch.randn(1, 1, 4))  # clean out hidden state
print(hidden)
out, hidden = lstm(inputs, hidden)
print(out)
print()
print(hidden)

test = torch.randn(4,2)
print(test)

test = torch.randn(4,1)
print(test)
test = test.view(2,-1)
print(test)

sm = nn.Softmax(dim=0)
print(sm(test))

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print("input: ", input)
target = torch.empty(3, dtype=torch.long)
print("target: ", target)
target = target.random_(2,5)
print("target: ", target)
output = loss(input, target)
print("output: ", output)
output.backward()

tensor = (torch.tensor([[1,2,3,4], [5,6,7,8]]))
print(tensor)
print(tensor.size())

print(torch.tensor((1,2)))
"""
