**用于在测试时从测试数据中手动取一批数据**
```
next(iter(data_iter))
```

**任何的nn中的模型都可以直接调用forward的魔法方法**
```
net = nn.Sequential(nn.Linear(2, 1))
net(train_data)
```

**模式切换**
```
net.train() #模型设置为训练模式
net.eval() #模型设置为评估模式
```
影响：
Dropout 层：
在训练模式下启用 Dropout，它会随机“丢弃”一部分神经元，防止过拟合。
Batch Normalization 层：
在训练模式下使用当前批次的均值和方差进行归一化，而不是全局统计值。

**计算预测正确的数量样例**
```
y_hat = y_hat.argmax(axis=1)
cmp = y_hat.type(y.dtype) == y
```
y_hat是二维张量
y_hat.argmax(axis=1) 会找到每个样本预测分数最大的类别的索引，返回形状为 (batch_size,) 的张量。
使用 type(y.dtype) 将 y_hat 转换为与 y 相同的数据类型，确保比较不会因为数据类型不一致而出错。

**K则交叉**
```
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

**reshape和view的区别**
view：
只能用于内存连续的张量。如果原张量在内存中不是连续存储的，会报错。
如果张量的内存不连续，可以通过 contiguous() 方法先将其转换为内存连续的形式，再使用view。
reshape：
不要求原张量是内存连续的。
如果张量在内存中不连续，reshape 会尝试创建一个新的张量或调整存储方式来满足需求，因此更灵活。

view：
不会分配新的内存，而是通过调整原张量的 视图（view） 来实现形状改变。
更加高效，因为它只是对张量的元数据（如形状和步幅）进行了修改。
reshape：
可能需要分配新的内存。如果原张量可以通过调整元数据直接实现形状改变，则行为类似于 view；否则，它会创建一个新的张量，分配新的内存并复制数据。


**权重衰退**
需要从数学角度分析
权重衰退是通过优化器中的 weight_decay 参数来设置的。常见的优化器如 Adam、SGD 等都可以设置 weight_decay 参数。

示例：使用 Adam 优化器并启用权重衰退
```
import torch
import torch.optim as optim

# 创建一个简单的模型
model = torch.nn.Linear(10, 1)

# 使用 Adam 优化器，启用权重衰退（weight_decay）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# 假设有一些训练数据
inputs = torch.randn(64, 10)
labels = torch.randn(64, 1)

# 损失函数
criterion = torch.nn.MSELoss()

# 训练过程中的一个步骤
optimizer.zero_grad()
output = model(inputs)
loss = criterion(output, labels)
loss.backward()
optimizer.step()
```
在上面的代码中，weight_decay=0.01 表示权重衰退系数为 0.01，这将对所有模型参数应用 L2 正则化。

**权重衰退的注意事项**
默认情况下没有权重衰退：
如果你没有显式设置 weight_decay 参数，它的默认值是 0，这意味着没有使用权重衰退。

在不同优化器中的使用：
SGD、Adam、RMSprop 等优化器都支持 weight_decay 参数，你可以通过它来开启 L2 正则化。
例如，对于 SGD 优化器，你也可以这样设置：
`optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)`

影响训练过程：
权重衰退会影响模型参数的更新，使得优化过程更倾向于保持较小的权重值，从而减少过拟合的风险。
适当的 weight_decay 值可以提升模型的泛化能力，但过大的 weight_decay 可能会导致欠拟合，过小的 weight_decay 可能没有明显的正则化效果。

对不同层的不同权重衰退：
如果你希望不同层使用不同的权重衰退系数，可以为不同的参数组设置不同的 weight_decay 值。例如：
```
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'weight_decay': 0.01},
    {'params': model.layer2.parameters(), 'weight_decay': 0.1}
], lr=0.001)
```

**dropout**
`nn.Dropout(dropout1)` dropout1为丢弃的比例如0.1

**数值稳定性和模型初始化**
需要进行数学推理

**批量归一化没必要和dropout混合使用**