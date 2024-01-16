from torch import nn

# 搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(  # 新特征是一个1*4的向量，通道为1
            nn.Conv2d(1, 3, 5, 1, 2),
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# model=Model()
# print(model)

# if __name__ == '__main__':
#     Model = Model()
#     input = torch.ones((64, 3, 32, 32))
#     output = Model(input)
#     print(output.shape)
