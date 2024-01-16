import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
# from tensorboardX import SummaryWriter
from dataset import MyDataset
from model import *
from torch import nn
# 准备数据集
from torch.utils.data import DataLoader
from torchvision import transforms as T

print("算法1，自定义卷积神经网络，使用十折交叉检验")
transform = T.Compose([
    T.Resize(32),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(32),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[0.560, 0.550, 0.494], std=[0.203, 0.196, 0.193])  # 标准化至[-1, 1]，规定均值和标准差
])
myData = MyDataset("/Users/zhengxiaozhu/Desktop/MachineLearning/FinalExam/dataset/dataset_all",
                   transform=transform)  # 读到的这个文件夹一定要是直接包含了所有数据集或者所有测试集的文件夹

# length 长度
train_data_size = 180
test_data_size = 20

# 利用 DataLoader 来加载数据集
myDataloader = DataLoader(myData, batch_size=4, shuffle=True)
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
writer = SummaryWriter("logs")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))
    # print("本轮测试集的下标是：%d~%d,%d~%d" % (12 * i, 12 * (i + 1), 8 * (24 - i), 8 * (25 - i)))
    # 训练步骤开始
    model.train()
    count = -1
    for data in myDataloader:  # 在本轮中，对每一个数据进行处理
        count = count + 1  # count用来统计已经读取了多少个data（总共应该有200/4=50个），初值为0，每读取一个data，count值+1
        if i * 3 <= count < (i + 1) * 3 or (24 - i) * 2 <= count < (25 - i) * 2:  # 跳过这些样本，因为这些样本要作为测试集。
            # print("跳过测试集", count)
            continue
            # 每轮作为测试集的样本下标分别是：[0~12，192~200][12~24，184~192]...[108~120，120~128]，即头取12个，尾取8个
            # 由于每个data里面包含四个样本，所以对应的count值应该是：[0~3,48~50][3~6,46~48]...[27~30,30~32]
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)  # 计算损失函数

        # 优化器优化模型
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
        count = -1
        for data in myDataloader:
            count = count + 1  # count用来统计已经读取了多少个data,每读取一个data，count值+1
            if not (i * 3 <= count < (i + 1) * 3 or (24 - i) * 2 <= count < (25 - i) * 2):  # 跳过不是测试集的样本
                continue
            # print("测试：",count)
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)  # 计算损失函数
            total_test_loss = total_test_loss + loss.item()  # 误差值累加
            accuracy = (outputs.argmax(1) == targets).sum()  # 精度值累加
            total_accuracy = total_accuracy + accuracy
            # if i + 1 == 10:
            for j in range(4):
                p = outputs.argmax(1)[j]
                y = targets[j]
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
    print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
    writer.add_scalar("test_loss", total_test_loss, total_test_step)  # 画图
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)  # 画图
    total_test_step = total_test_step + 1
    # torch.save(model, "model_{}.pth".format(i))  # 用文件名为model_XX.pth的文件来保存每一轮训练出来的模型

writer.close()
# 打开 tensorboard：在terminal里面输入 tensorboard --logdir "/Users/zhengxiaozhu/PycharmProjects/pythonProject_pytorch/A1_2/logs"
