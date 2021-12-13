import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import shffulnetv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU设置
json_file = open('type.json', "r")
class_indict = json.load(json_file)
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = Image.open('1.jpg')
plt.imshow(img)
img = data_transform(img)#图片缩放并转tensor
img = torch.unsqueeze(img, dim=0)#图片加个维度，因为网络输入是4维
model = shffulnetv2(num_classes=3).to(device)
model.load_state_dict(torch.load('11.51_1.0.pkl', map_location=device))#载入模型参数
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()#去除结果的第一个维度
    predict = torch.softmax(output, dim=0)#softmax处理
    predict_cla = torch.argmax(predict).numpy()#获取最大值索引
print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla])#在图片上显示
plt.title(print_res)
plt.show()


