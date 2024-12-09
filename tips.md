
```
next(iter(data_iter))
```
往往用于在测试时从测试数据中手动取一批数据

```
net = nn.Sequential(nn.Linear(2, 1))
net(train_data)
```
任何的nn中的模型都可以直接调用forward的魔法方法

```
net.train() #模型设置为训练模式
net.eval() #模型设置为评估模式
```
影响：
Dropout 层：
在训练模式下启用 Dropout，它会随机“丢弃”一部分神经元，防止过拟合。
Batch Normalization 层：
在训练模式下使用当前批次的均值和方差进行归一化，而不是全局统计值。

```
"""计算预测正确的数量"""
y_hat = y_hat.argmax(axis=1)
cmp = y_hat.type(y.dtype) == y
```
y_hat是二维张量
y_hat.argmax(axis=1) 会找到每个样本预测分数最大的类别的索引，返回形状为 (batch_size,) 的张量。
使用 type(y.dtype) 将 y_hat 转换为与 y 相同的数据类型，确保比较不会因为数据类型不一致而出错。