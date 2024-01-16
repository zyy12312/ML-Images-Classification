from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/Users/zhengxiaozhu/PycharmProjects/pythonProject_pytorch/test/logs1")
#y=x
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()


# tensorboard --logdir "/Users/zhengxiaozhu/PycharmProjects/pythonProject_pytorch/test/logs"