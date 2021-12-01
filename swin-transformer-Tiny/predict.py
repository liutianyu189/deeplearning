import json
import torch
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from models import SwinTransformer

supported = ["jpg", "png"]  # 支持的文件后缀类型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU设置
json_file = open('type.json', "r")
class_indict = json.load(json_file)
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def main(args):

    img_path =args.img_path
    if img_path.split('.')[-1] in supported:#判断是否是单张图片还是文件夹
        img = Image.open(img_path)
        plt.imshow(img)
        img = data_transform(img)#图片缩放并转tensor
        img = torch.unsqueeze(img, dim=0)#图片加个维度，因为网络输入是4维
        model = SwinTransformer(num_classes=args.num_classes).to(device)
        model_weight_path = args.model_weight_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device))#载入模型参数
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()#去除结果的第一个维度
            predict = torch.softmax(output, dim=0)#softmax处理
            predict_cla = torch.argmax(predict).numpy()#获取最大值索引
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla])#在图片上显示
        plt.title(print_res)
        plt.show()

    elif os.path.isdir(img_path):#判断是否是文件夹
        for i in os.listdir(img_path):
            img = Image.open(os.path.join(img_path,i))
            plt.imshow(img)
            img = data_transform(img)  # 图片缩放并转tensor
            img = torch.unsqueeze(img, dim=0)  # 图片加个维度，因为网络输入是4维
            model = SwinTransformer(num_classes=args.num_classes).to(device)
            model_weight_path = args.model_weight_path
            model.load_state_dict(torch.load(model_weight_path, map_location=device))  # 载入模型参数
            model.eval()
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()  # 去除结果的第一个维度
                predict = torch.softmax(output, dim=0)  # softmax处理
                predict_cla = torch.argmax(predict).numpy()  # 获取最大值索引
            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla])  # 在图片上显示
            plt.title(print_res)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)  # 类别数
    parser.add_argument('--img_path', type=str, default='./img/level1')  # 读取图片路径,可设置为图片或者包含图片的文件夹
    parser.add_argument('--model_weight_path', type=str, default='./st.pth')  # 读取模型路径
    opt = parser.parse_args()
    main(opt)