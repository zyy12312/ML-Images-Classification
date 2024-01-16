import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboardX import SummaryWriter
from A1.dataset import MyDataset
from A1.model import *
from torch import nn
# 准备数据集
from torch.utils.data import DataLoader
from torchvision import transforms as T
print("算法2，VGG卷积神经网络")
transform = T.Compose([
    T.Resize(32),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(32),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[0.560, 0.550, 0.494], std=[0.203, 0.196, 0.193])  # 标准化至[-1, 1]，规定均值和标准差
])
train_data = MyDataset("/Users/zhengxiaozhu/Desktop/MachineLearning/FinalExam/dataset/train_all",
                       transform=transform)  # 读到的这个文件夹一定要是直接包含了所有数据集或者所有测试集的文件夹
test_data = MyDataset("/Users/zhengxiaozhu/Desktop/MachineLearning/FinalExam/dataset/test_all", transform=transform)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))  # 160
print("测试数据集的长度为：{}".format(test_data_size))  # 40

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
# 数据加载完毕

# 创建网络模型
model = Model()

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 交叉熵
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 优化器选择随机梯度下降优化器，model.parameters()是我们的模型参数

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")  # ../logs_train

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:  # 在本轮中，对每一个数据进行处理
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)  # 计算损失函数

        # 优化器优化模型, global_step=1
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 利用刚刚得到的grad来进行优化

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  # 只有训练的次数是100的倍数时才输出一次，这样输出的信息不会那么多
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # 画图，标题：train_loss，y轴：，x轴
    # print("-------第 {} 轮训练结束，开始测试-------".format(i + 1))
    # 测试步骤开始（每轮训练都做一次测试，确保程序在跑，以及精度在逐渐提高）
    model.eval()
    total_test_loss = 0  # 将测试集上的误差初始化
    total_accuracy = 0  # 将测试集上的精度初始化
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    with torch.no_grad():  # 不需要调整梯度，也不需要用梯度来优化
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)  # 计算损失函数
            total_test_loss = total_test_loss + loss.item()  # 误差值累加
            accuracy = (outputs.argmax(1) == targets).sum()  # 精度值累加
            total_accuracy = total_accuracy + accuracy
            if i + 1 == 10:
                for j in range(4):
                    p=outputs.argmax(1)[j]
                    y=targets[j]
                    if p == 1 and y == 1:
                        TP = TP + 1
                    elif p == 1 and y == 0:
                        FP = FP + 1
                    elif p == 0 and y == 0:
                        TN = TN + 1
                    elif p == 0 and y == 1:
                        FN = FN + 1

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)  # 画图 , global_step=i
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)  # 画图
    total_test_step = total_test_step + 1
    # torch.save(model, "model_{}.pth".format(i))  # 用文件名为model_XX.pth的文件来保存每一轮训练出来的模型
    # print("模型已保存")

writer.close()
print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
# 打开 tensorboard：在terminal里面输入 tensorboard --logdir "/Users/zhengxiaozhu/PycharmProjects/pythonProject_pytorch/A2/logs"
