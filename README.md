# Pytorch-LiMu

d2l.synthetic_data(w, b, num) w为(n,1)的形状，b为(1,) 生成形状为（num，n）的训练数据和（num，1）的标签

### timer = d2l.Timer()
功能
- 计时器工具：用于记录代码执行的时间。
- 在深度学习中，用于评估训练、推理等阶段的耗时。
方法
- start(): 开始计时（通常自动调用）。
- stop(): 暂停计时并返回耗时。
- cumsum(): 获取累计的耗时。
- avg(): 获取平均耗时。

> 本身不算太复杂，不介意的话可以自己写一个时间类

### train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
功能
- 加载 Fashion-MNIST 数据集：
    - Fashion-MNIST 是一个经典的深度学习数据集，用于图像分类任务。
    - 通过 d2l.load_data_fashion_mnist 函数，快速加载训练集和测试集。
参数
- batch_size：每个批次加载的样本数。
返回值
- train_iter：训练数据的迭代器（DataLoader），每次迭代返回一个批次的数据和标签。
- test_iter：测试数据的迭代器。

> 针对fashion_mnist数据集的一个快速读取数据的方法，内部是封装了dataset和dataloader等方法

### d2l.sgd([W, b], lr, batch_size)
功能
- 实现随机梯度下降（SGD）优化算法：
    - sgd 函数是一个手动实现的优化算法，用于更新模型参数W和b。
参数
- [W, b]：需要优化的模型参数列表。
- lr：学习率，决定参数更新的步长。
- batch_size：每个批次样本的大小，用于梯度归一化。

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
功能
- 针对第二节线性回归专门写的一个方法，内部封装了很多函数，一键完成训练、测试一级可视化作图等过程

参数
- net: 训练的模型
- train_iter, test_iter: 训练数据（的迭代器）
- loss: 损失函数
- num_epochs: 训练轮次
- trainer: 优化器

> 问题在于d2l版本不对会无法正常运行，要从d2l.train_ch3改为train_ch3，但是这样一来又会报错train_ch3内部调用的相关函数缺乏定义，得一个个找到定义放到前面完成编译才能成功运行。这么一来的话...性价比不高