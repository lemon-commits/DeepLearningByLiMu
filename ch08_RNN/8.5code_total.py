import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# ==========================================
# 第一部分：被 d2l 隐藏的工具类（解包）
# ==========================================

class Timer:
    """记录多次运行时间的工具类"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

class Accumulator:
    """在 n 个变量上累加的工具类（常用来累加 Loss 和 预测正确的数量）"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]

def sgd(params, lr, batch_size):
    """小批量随机梯度下降（最基础的优化器）"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# ==========================================
# 第二部分：RNN 核心组件定义
# ==========================================

def get_params(vocab_size, num_hiddens, device):
    """初始化 RNN 模型的所有参数"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 按照均值为0，标准差为0.01的正态分布初始化权重
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens)) # 输入到隐状态的权重
    W_hh = normal((num_hiddens, num_hiddens)) # 上一隐状态到当前隐状态的权重
    b_h = torch.zeros(num_hiddens, device=device) # 隐藏层偏置

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs)) # 隐状态到输出的权重
    b_q = torch.zeros(num_outputs, device=device) # 输出层偏置

    # 附加梯度计算
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)  # 需要更新参数
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    """初始化隐藏状态（在时间步 t=0 时，没有过去的记忆，所以全填 0）"""
    # 返回一个元组，以匹配将来可能更复杂的模型（如LSTM有两个隐状态）
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    """定义前向传播的计算逻辑"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    
    # inputs的形状：(时间步数量, 批量大小, 词表大小)
    # X 遍历 inputs，相当于模型一步步按时间顺序读取当前字符
    for X in inputs:
        # 核心公式：结合当前输入 X 和上一步的记忆 H
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # 完成了每个时间步的隐藏状态更新
        # 根据当前记忆 H 计算输出
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
        
    # 把所有时间步的输出拼接在一起，形成一个巨大的二维张量
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    """包装类：把上面散装的函数封装成一个可以调用的模型对象"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):  # 魔术方法，无需显示写方法名即可调用：net(X, state)
        # 将传入的整数索引 X 转换为独热编码（One-hot）向量
        # 注意这里 X.T 转置了，确保时间步维度在最前面
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    
    # 动态获取隐藏层大小，方便外部调用
    @property  # 
    def num_hiddens(self):  # 只读属性，可以直接调用：net.num_hiddens
        return self.params[1].shape[0]

# ==========================================
# 第三部分：训练与预测的辅助函数
# ==========================================

def grad_clipping(net, theta):
    """梯度裁剪：防止 RNN 梯度爆炸"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params # 我们从零实现走这个分支
        
    # 计算所有参数梯度的 L2 范数（长度）
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def predict_ch8(prefix, num_preds, net, vocab, device):
    """基于前缀字符串 prefix，预测接下来的 num_preds 个字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    # 使用 lambda 动态获取最新预测的字符作为下一步的输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))  # 获取最近的一个字符
    
    # 预热期：用已知的前缀更新记忆（但不预测）
    for y in prefix[1:]:
        _, state = net(get_input(), state)  # 拿到最新的H，类会先把单个字符变成one-hot编码量，再输入到模型中去
        outputs.append(vocab[y])
        
    # 预测期：利用最新的记忆开始无中生有
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # 这里是逐个字符去预测的 outputs中内存放的是token的索引
        
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# ==========================================
# 第四部分：核心训练逻辑
# ==========================================

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练一个 Epoch"""
    state, timer = None, Timer()
    metric = Accumulator(2)  # 记录 [总损失累加, 总词元数量]
    
    for X, Y in train_iter:
        # 处理隐状态：如果是新的 epoch 或使用了随机采样，重新初始化记忆
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 截断反向传播，防止 OOM
            if isinstance(net, nn.Module) and not isinstance(state, tuple):  # 不是第一次初始化的隐藏状态，需要截断反向传播
                state.detach_()  # 带_表示原地修改，每一个batch对应的num_steps,只在当前batch更新梯度并进行优化
            else:
                for s in state:
                    s.detach_()
                    
        # 处理标签，展平以对应模型的输出顺序
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        y_hat, state = net(X, state)
        # 计算当前 batch 的平均损失
        l = loss(y_hat, y.long()).mean()
        
        # 反向传播与优化
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1) # 梯度裁剪
            updater.step()
        else:
            # 从零实现走这个分支
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1) # 我们用的自定义 sgd 函数已经除过 batch_size 了
            
        # 记录数据：当前批次总损失，当前批次总词元数
        metric.add(l * y.numel(), y.numel())
        
    # 返回：困惑度 (Perplexity), 训练速度 (Tokens / sec)
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """总训练循环（去除了 d2l 的动画画图功能，改为纯控制台打印）"""
    loss = nn.CrossEntropyLoss()
    
    # 定义更新器（这里调用我们解包出来的 sgd）
    def updater(batch_size):
        return sgd(net.params, lr, batch_size)
    
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        
        # 每隔 10 个 epoch 打印一次预测结果看看效果
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Perplexity: {ppl:.1f} | Speed: {speed:.1f} tokens/sec")
            print(f" -> 预测生成: {predict_ch8('time traveller', 50, net, vocab, device)}")

# ==========================================
# 第五部分：运行入口
# ==========================================
if __name__ == "__main__":
    # 配置超参数
    batch_size = 32
    num_steps = 35
    num_hiddens = 512
    num_epochs = 500
    lr = 1.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 读取数据集 (保留 d2l，因为不涉及模型原理)
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 实例化模型
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)

    # 打印初始状态预测（全是随机的，结果是乱码）
    print("未训练时的预测:", predict_ch8('time traveller ', 10, net, vocab, device))

    # 开始训练
    print("\n--- 开始训练 ---")
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)