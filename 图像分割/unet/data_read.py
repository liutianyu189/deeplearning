import os
from PIL import Image
import torch.utils.data as D
from torchvision import transforms

trans = transforms.Compose([
    transforms.Resize((512,512)),#修改输送到网络的图像大小
    transforms.ToTensor()])

def get_img(root1, root2):
    img_train = [os.path.join(root1, i) for i in os.listdir(root1) if i.split('.')[1] in ["jpg", "png"]]
    img_test = [os.path.join(root2, j) for j in os.listdir(root2) if j.split('.')[1] in ["jpg", "png"]]
    return img_train, img_test

class MyDataset(D.Dataset):
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        label = Image.open(self.mask_path[idx])
        img =trans(img)
        label = trans(label)
        label=label.squeeze()
        label=255*label
        label=label.long()
        return img, label

def get_dataloader(batchsize, root1, root2,root3,root4):
    img_train, img_test= get_img(root1, root2)
    label_train,label_test=get_img(root3,root4)
    train_loader = D.DataLoader(MyDataset(img_train, label_train), batch_size=batchsize, shuffle=True,
                                pin_memory=True, num_workers=0)
    valid_loader = D.DataLoader(MyDataset(img_test, label_test), batch_size=batchsize,
                                pin_memory=True, num_workers=0)
    return train_loader, valid_loader

