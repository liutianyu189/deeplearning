import torch
from torchvision import transforms
from PIL import Image
from  model import UNet
import numpy as np
import json

with open('palette.json', "rb") as f:
    pallette_dict = json.load(f)
    pallette = []
    for v in pallette_dict.values():
        pallette += v

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trans = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()])
model =UNet(n_class=3).to(device)#修改为类别数+1
model.load_state_dict(torch.load('0.672.pkl', map_location=device))#修改为模型路径
img = Image.open('1.jpg')
tensor = trans(img).unsqueeze(axis=0)
model.eval()
with torch.no_grad():
    res = model(tensor.to(device))
    res = res.argmax(1).squeeze(0)
    res = res.to("cpu").numpy().astype(np.uint8)
    mask = Image.fromarray(res)
    mask.putpalette(pallette)
    mask.save("test_result.png")
