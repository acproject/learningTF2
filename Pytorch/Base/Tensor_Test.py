import torch
a = torch.Tensor(2, 2)
'''
tensor([[-6.3497e+09,  4.5874e-41],
        [-6.3497e+09,  4.5874e-41]])
'''
b = torch.DoubleTensor(2, 2)
'''
tensor([[4.6575e-310,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00]], dtype=torch.float64)
'''
c = torch.Tensor([[1, 2], [3, 4]])
'''
tensor([[1., 2.],
        [3., 4.]])
'''
d = torch.zeros(2, 2)
'''
tensor([[0., 0.],
        [0., 0.]])
'''
e = torch.ones(2, 2)
'''
tensor([[1., 1.],
        [1., 1.]])
'''
f = torch.eye(2, 2)
'''
tensor([[1., 0.],
        [0., 1.]])
'''
g = torch.randn(2, 2)
'''
tensor([[-2.5370,  0.4256],
        [ 1.9137, -0.5371]])
'''
h = torch.arange(1, 6, 2) #(start, end, step)
'''
tensor([1, 3, 5])
'''
i = torch.linspace(1, 6, 2) #(start, end, steps)
'''
tensor([1., 6.])
'''
j = torch.randperm(4)
'''
tensor([2, 1, 3, 0])
'''
k = torch.tensor([1, 2, 4])
'''
tensor([1, 2, 4])
'''

ii = torch.linspace(1, 6, 10) #(start, end, steps)
'''
tensor([1.0000, 1.5556, 2.1111, 2.6667, 3.2222, 3.7778, 4.3333, 4.8889, 5.4444,
        6.0000])
'''
hh = torch.arange(1, 6, 10)
'''
tensor([1])
'''
print(a)
print("\n")
print(b)
print("\n")
print(c)
print("\n")
print(d)
print("\n")
print(e)
print("\n")
print(f)
print("\n")
print(g)
print("\n")
print(h)
print("\n")
print(i)
print("\n")
print(j)
print("\n")
print(k)
print("\n")
print(ii)
print("\n")
print(hh)
print(a.numel())
