import numpy as np
from torch import nn
import torch
from d2l import torch as d2l
import sys, os 
sys.path.append(r"D:\Book\202603-202606\DeepLearning\code")
from utils.ch03 import Animator
num_inputs = 784
num_hiddens = 256
num_outputs = 10
batch_size = 256
lr = 0.1
num_epochs = 10
# 读取数据
data_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 参数初始化
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens,requires_grad=True)*0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params = [W1, b1, W2, b2]
# 定义优化器
trainer = torch.optim.SGD(params, lr = lr)
# 先定义一个类，方便后面求三条直线
class Accumulator:  
    def __init__(self, n):
        self.data = [0.0] * n   # list * n 生成n个0的列表（初始化）
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):  # 魔术方法，后面可以直接调用metric[idx]（定义的Accumulator类）获取第idx个变量的值
        return self.data[idx]

# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return H@W2 + b2
# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 训练函数
## 计算预测正确率
def accuracy(y_hat, y):  # 返回一个批量预测正确的总数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    cmp = (y_hat.type(y.dtype) == y) # dtype() 方法用于返回张量的元素类型，type() 方法用于将张量转换为指定的元素类型
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的数量
# 计算在指定数据集上模型的精度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval() # 开启评估模式，禁用Dropout等层的训练行为
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

# 计算每一轮epoch的训练精度
def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train() # 开启训练模式，启用Dropout等层的训练行为

    metric = Accumulator(3)  # 训练损失总和，训练准确度总和，样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()  # 计算梯度
            updater.step()  # 更新参数

        else:
            l.sum().backward()
            updater()
    metric.add(float(l.sum()), accuracy(y_hat, y), y.numel()) # 损失值求和，一个batch的准确个数，样本数
    return metric[0] / metric[2], metric[1] / metric[2]
# 训练函数
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim = [1, num_epochs], ylim = [0.3, 0.9],
                        legend = ['train_loss', 'train_acc', 'test_acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)  # 返回一个二元元组
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

if __name__ == '__main__':
    train(net, data_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()