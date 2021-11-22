from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as Data
from torchvision import transforms
import os
import json
import random

def read_datesets(root:str):
    random.seed(0)    # 保证随机结果可复现
    list_read = [i0 for i0 in os.listdir(root) if os.path.isdir(os.path.join(root, i0))]   # 遍历数据集文件，生成包含类别名称的列表
    list_read.sort()    # 名称排序
    dict_read = dict((k, v) for v, k in enumerate(list_read))  #列表中内容和索引转成字典，用于获取标签
    dict_save = dict((k, v) for k, v in enumerate(list_read))  #列表中内容和索引转成字典,预测使用
    # 保存成json文件
    with open('type.json','w') as f:
        json.dump(dict_save,f)
    train_path = []  # 存储训练集的所有图片路径
    train_label = []  # 存储训练集图片对应索引信息
    val_path = []  # 存储验证集的所有图片路径
    val_label = []  # 存储验证集图片对应索引信息
    every_num = []  # 存储每个类别的样本总数
    supported = ["jpg", "png"]  # 支持的文件后缀类型
    for i1 in list_read:
        #获取每类别的路径
        type_path = os.path.join(root, i1)
        #获取supported支持的所有图片路径
        images = [os.path.join(root, i1, i) for i in os.listdir(type_path) if i.split('.')[-1] in supported]
        # 获取该类别对应的索引
        image_class = dict_read[i1]
        # 记录该类别的样本数量
        every_num.append(len(images))
        # 按比例随机采样验证样本
        val = random.sample(images, k=int(len(images) *0.1))
        #判断是否在val列表里，在的话放验证集，不在放训练集
        for i2 in images:
            if i2 in val:
                val_path.append(i2)
                val_label.append(image_class)
            else:
                train_path.append(i2)
                train_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_num)))
    print("{} images for training.".format(len(train_path)))
    print("{} images for validation.".format(len(val_path)))
    return train_path, train_label, val_path, val_label

#设置训练集验证集的transform，用于生成tensor
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),#随机剪裁并缩放到224大小
                                 transforms.RandomHorizontalFlip(),#随机旋转
                                 transforms.ToTensor(),#转tensor
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),#标准化处理
    "val": transforms.Compose([transforms.Resize(256),#缩放到256大小
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#pytorch官方数据集定义方法
class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, index):
        img = Image.open(self.images_path[index])#打开图片
        if img.mode != 'RGB':#图片不是RGB通道就报错
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[index]))
        img=self.transform(img)#对图片使用transform
        label = self.images_class[index]#设置图片的标签
        return img, label

root = 'img'#数据集路径
train_path, train_label, val_path, val_label = read_datesets(root)#获取4个列表
#定义训练集
train_data_set = MyDataSet(images_path=train_path,
                           images_class=train_label,
                           transform=data_transform["train"]
                           )
#定义验证集
val_data_set = MyDataSet(images_path=val_path,
                           images_class=val_label,
                           transform=data_transform["val"])

train_loader = Data.DataLoader(train_data_set,batch_size=4,shuffle=True,num_workers=0)#导入训练集
val_loader = Data.DataLoader(val_data_set,batch_size=2,shuffle=False,num_workers=0)#导入预测集
