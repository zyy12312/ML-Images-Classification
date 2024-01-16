# normMean = [0.56065917, 0.5504985, 0.4945692]
# normStd = [0.2039005, 0.19660968, 0.19380517]
import os
import numpy as np
import torch
os.environ['KMP_DUPLIairplaneE_LIB_OK'] = 'True'
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# 过滤警告信息
import warnings
import texture
warnings.filterwarnings("ignore")


class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, path_dir, transform=None):  # 初始化一些属性
        self.path_dir = path_dir  # 文件路径
        self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
        self.images = os.listdir(self.path_dir)  # 把路径下的所有文件放在一个列表中

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        image_index = self.images[index]  # 根据索引获取图像文件名称，如'forest86.tif'
        img_path = os.path.join(self.path_dir, image_index)  # 获取到一张图像的路径或目录
        # 如'/Users/zhengxiaozhu/Desktop/MachineLearning/FinalExam/dataset/dataset_all/forest86.tif'
        img_texture = texture.test(img_path)  # img_texture是图片的纹理特征(一维矩阵）
        # img = Image.open(img_path).convert('RGB')  # 读取图像

        # 根据目录名称获取图像标签（airplane或forest）
        label = img_path.split('\\')[-1].split('.')[0]
        # 把字符转换为数字airplane-0，forest-1
        label = 1 if 'forest' in label else 0

        # if self.transform is not None:
        #     img_texture = self.transform(img_texture)
        return img_texture, label

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)


# from torchvision import transforms as T
#
# transform = T.Compose([
#     T.Resize(32),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
#     T.CenterCrop(32),  # 从图片中间切出224*224的图片
#     T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
#     T.Normalize(mean=[0.560, 0.550, 0.494], std=[0.203, 0.196, 0.193])  # 标准化至[-1, 1]，规定均值和标准差
# ])

# dataset = MyDataset("/Users/zhengxiaozhu/Desktop/MachineLearning/FinalExam/dataset/dataset_all",transform=transform)  # 读到的这个文件夹一定要是直接包含了文件的文件夹
# dataset = MyDataset("/Users/zhengxiaozhu/Desktop/MachineLearning/FinalExam/dataset/dataset_all")
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# for x, y in dataloader:  # x是一个list，里面的每个元素都是list
#     x2=torch.cat(x,dim=0)
#     print(x2.shape)  # 输出"torch.Size([4, 3, 32, 32])"，其中4代表4张图片，3代表3通道，32，32是图片尺寸
#    print(y.shape)  # torch.Size([4]) 4代表4张图片

