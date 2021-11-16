import torch
from torchvision import transforms
from PIL import Image
from  model import fcn
import time
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trans = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()])
model =fcn(1).to(device)
model.load_state_dict(torch.load('fcn.pkl', map_location=device))
stime=time.time()
for i in os.listdir('img_pre'):
    img = Image.open(f'img_pre/{i}')
    tensor = trans(img).unsqueeze(axis=0)
    model.eval()
    with torch.no_grad():
        res = model(tensor.to(device))
        res = res.squeeze().cpu()
        res = torch.sigmoid(res)
        res[res > 0.5] = 1
        res[res <= 0.5] = 0
        res = transforms.ToPILImage()(res)
        res.save(f'result/{i}')
etime=time.time()
print(etime-stime)