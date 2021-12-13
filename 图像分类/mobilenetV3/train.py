import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import MobileNetV3
import math
import torch.optim.lr_scheduler as lr_scheduler
from data_read import get_data
#图片路径，batch_size，测试集比例
train_loader,val_loader,val_path=get_data('img',batchsize=4,rat=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#设置使用的设备
net = MobileNetV3(num_classes=3).to(device)#载入网络
epochs=20#训练次数
best_acc = 0.0#最高的准确率初始化
loss_function = nn.CrossEntropyLoss()#损失函数设定
optimizer = optim.Adam(net.parameters(), lr=0.0005)#优化器设定
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 -0.1) + 0.1#自适应学习率公式
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)#自适应调整学习率设定

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        optimizer.zero_grad()
        output = net(images.to(device))
        loss = loss_function(output, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss =running_loss+loss.item()
        scheduler.step()#学习率更新
    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]#获取输出最大值的索引
            acc= acc+torch.eq(predict_y, val_labels.to(device)).sum().item()#索引和标签对比
    val_accurate = acc / len(val_path)
    print('[epoch {}] train_loss: {}  val_accuracy: {}'.format(epoch + 1, round((running_loss),3), round(val_accurate,3)))
    torch.save(net.state_dict(), f'{round((running_loss),3)}_{round(val_accurate,3)}.pkl')

