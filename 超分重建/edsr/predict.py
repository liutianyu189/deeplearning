import torch
from PIL import Image
from model import edsr
import numpy as np
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPU设置
trans=transforms.ToTensor()
img = Image.open('test.jpg')
img=trans(img)
img = torch.unsqueeze(img,0)
model = edsr().to(device)
model.load_state_dict(torch.load('logs/15.26666.pkl', map_location=device))#载入模型参数
model.eval()
with torch.no_grad():
    out = model(img.to(device)).cpu()
    out=torch.squeeze(out)

    out=(out*255).numpy()
    out = out.transpose([1, 2, 0])
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    out=Image.fromarray(out)
    out.save('out.jpg')


















