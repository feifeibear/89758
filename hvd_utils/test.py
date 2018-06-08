import torch


torch.manual_seed(123)
x = torch.randn(10, 10) #FloatTensor([[1, 2, 3], [4, 5, 6]])
x = x.cuda()
x_len = 1;
for dim in x.size():
    x_len *= dim
print("x_len : ", x_len)
ratio = 0.001

x_flatten = x.view(-1)

max_val = torch.max(x_flatten)
mean_val = torch.mean(x_flatten)
thd = mean_val + 0.5 * (max_val - mean_val)
x_sparse = x_flatten > thd
print(x_sparse)
x_idx = torch.nonzero(x_sparse)
print(x_idx)
