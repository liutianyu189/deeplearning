import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import EfficientNetv2
import math
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from data_read import val_loader,train_loader,val_path


def main(args):
    val_num = len(val_path)  # 获取验证集数量
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#设置使用的设备
    net = EfficientNetv2(num_classes=args.num_classes,).to(device)#载入网络

    # 迁移学习，权重载入
    # model_weight_path = "./mobilenet_v3.pth"
    # pre_weights = torch.load(model_weight_path)
    # model_dict = net.state_dict()#OrderedDict类型
    # keys = []
    # i = 0
    # for k, v in pre_weights.items():
    #     keys.append(k)
    # for k1, v1 in model_dict.items():
    #     if v1.size() == pre_weights[keys[i]].size():
    #         model_dict[k1] = pre_weights[keys[i]]
    #         i = i + 1
    # net.load_state_dict(model_dict)

    epochs=args.epochs#训练次数
    best_acc = 0.0#最高的准确率初始化
    loss_function = nn.CrossEntropyLoss()#损失函数设定
    optimizer = optim.Adam(net.parameters(), lr=args.lr)#优化器设定
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 -args.lfc) + args.lfc#自适应学习率公式
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)#自适应调整学习率设定
    train_steps = len(train_loader)


    for epoch in range(epochs):
        #训练
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
            train_bar.desc = "train epoch[{}/{}]".format(epoch + 1,epochs)#进度条显示优化
            scheduler.step()#学习率更新
        #验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]#获取输出最大值的索引
                acc= acc+torch.eq(predict_y, val_labels.to(device)).sum().item()#索引和标签对比
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)#进度条显示优化
        val_accurate = acc / val_num
        print('[epoch {}] train_loss: {}  val_accuracy: {}'
              .format(epoch + 1, round((running_loss / train_steps),3), round(val_accurate,3)))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), args.save_path)
    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)#类别数
    parser.add_argument('--epochs', type=int, default=30)#训练次数
    parser.add_argument('--save_path', type=str, default='./efficientnetv2.pth')#保存路径
    parser.add_argument('--lr', type=float, default=0.0001)#初始学习率
    parser.add_argument('--lfc', type=float, default=0.1)#自适应学习因子
    opt = parser.parse_args()
    main(opt)