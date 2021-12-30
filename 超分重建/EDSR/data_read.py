import os
from PIL import Image
import torch.utils.data as D
from torchvision import transforms

def get_img(root1, root2, root3, root4):
    img1 = [os.path.join(root1, i) for i in os.listdir(root1) if i.split('.')[1] in ["jpg", "png","bmp"]]
    img2 = [os.path.join(root2, j) for j in os.listdir(root2) if j.split('.')[1] in ["jpg", "png","bmp"]]
    img3 = [os.path.join(root3, k) for k in os.listdir(root3) if k.split('.')[1] in ["jpg", "png","bmp"]]
    img4 = [os.path.join(root4, l) for l in os.listdir(root4) if l.split('.')[1] in ["jpg", "png","bmp"]]
    return img1, img2, img3, img4

class MyDataset(D.Dataset):
    def __init__(self, imgLR_path, imgHR_path):
        self.trans = transforms.ToTensor()
        self.imgLR_path = imgLR_path
        self.imgHR_path = imgHR_path
    def __len__(self):
        return len(self.imgHR_path)
    def __getitem__(self, idx):
        imgLR = Image.open(self.imgLR_path[idx]).convert('RGB')
        imgHR = Image.open(self.imgHR_path[idx]).convert('RGB')
        imgLR =self.trans(imgLR)
        imgHR = self.trans(imgHR)
        return imgLR, imgHR

def get_dataloader(batchsize, root1, root2,root3,root4):
    train_imgLR, train_imgHR, test_imgLR, test_imgHR= get_img(root1, root2,root3,root4)
    train_loader = D.DataLoader(MyDataset(train_imgLR, train_imgHR), batch_size=batchsize, shuffle=True,
                                pin_memory=True, num_workers=0)
    valid_loader = D.DataLoader(MyDataset(test_imgLR, test_imgHR), batch_size=batchsize,
                                pin_memory=True, num_workers=0)
    return train_loader, valid_loader

