import torch


x = torch.ones(2,2, requires_grad=False)

x.requires_grad = True

print(x)

y = x + 2

print(y)

z = y * 3
print(z)

z.backward();

print(x.grad)

#out = z.mean()

#print(out)

#out.backward()

#print(x.grad)
