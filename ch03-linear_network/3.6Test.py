import torch
from IPython import display
from d2l import torch as d2l
import sys, os

# 根据原代码设置绝对路径并引入 Animator
sys.path.append(r"D:\Book\202603-202606\DeepLearning\code")  # 绝对路径设置到父目录就好了
from utils.ch03 import Animator

# ==========================================
# 1. 初始化及加载数据
# ==========================================
# 引入3.5节中提到的Fashion-mnist数据集，设置batchsize为256
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# 因为每个样本都是 28 x 28 的灰度图，所以每个样本的像素数量为 784。
# 在这里使用正态分布初始化权重，偏置项初始化为0
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# ==========================================
# 2. 定义核心操作与模型
# ==========================================
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里用了广播机制，每一行的元素都除以这一行的和

def net(X):
    # 返回 softmax(XW + b)
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # 只有对正确预测的类别取对数

# ==========================================
# 3. 精度评估与辅助类
# ==========================================
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 这里是针对使用非独热编码的情况，进行转换
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 将预测结果的数据类型转换为与真实标签 y 相同（确保类型一致）
    return float(cmp.type(y.dtype).sum())  # 返回的是预测正确的总数

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n   # list * n 生成n个0的列表（初始化）

    def add(self, *args):  # 当前列表累加值和args（新传入的可变参数值进行配对）
        # 传入的args是一个元组，*args是可变长度参数语法，在这个版本代码里面，args为（预测正确的总数、预测总数）
        # * 是python的一个语法糖，可以直接传入add（num1, num2, ...）
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):  # 重置所有变量为0.0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):  # 魔术方法，用于实现索引操作，（这里可以直接调用metric[idx]）
        return self.data[idx]

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module): # isinstance(obj, class_or_tuple) 判断对象obj是否是类class_or_tuple的实例
        net.eval()  # pytorch中nn.Module的eval方法，将模型设置为评估模式。
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():  # 这个with语句用于在评估模式下，关闭自动求导机制，以节省内存，代码块结束完毕之后自动清理资源
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # numel() 方法用于返回张量中元素的个数
    return metric[0] / metric[1]

# ==========================================
# 4. 训练与优化逻辑
# ==========================================
def train_epoch_ch3(net, train_iter, loss, updater):  # loss是作为损失函数的参数名传入的
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):    # net是PyTorch的Module类的实例，需要设置为训练模式
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y) # 损失函数l
        if isinstance(updater, torch.optim.Optimizer): # updater是PyTorch的Optimizer类的实例，需要调用zero_grad()和step()方法
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward() # 使用求和然后计算梯度
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)  # 测试集的准确率
        animator.add(epoch + 1, train_metrics + (test_acc,)) # 元组的拼接，本质上传入的是一个三元素的元组
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss  # 设置检查点
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 设置学习率为0.1，使用小批量随机梯度下降优化
lr = 0.1 
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# ==========================================
# 5. 预测函数
# ==========================================
def predict_ch3(net, test_iter, n=6):  # 预测六张
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# ==========================================
# 6. 主程序：执行训练和预测
# ==========================================
if __name__ == "__main__":
    # 训练十个迭代周期
    num_epochs = 10
    print(f"开始训练模型，共计 {num_epochs} 个 epochs...")
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    
    # 训练完成后，测试预测能力
    print("模型训练完成，正在生成预测结果...")
    predict_ch3(net, test_iter)

    # 在脚本环境（非Jupyter环境）下，必须调用 plt.show() 才能使图像窗口停留并显示
    d2l.plt.show()