import torch
import torch.nn as nn
from tqdm import tqdm
from at_unet import AttU_Net
import torch.optim as optim
import os
import cv2
import numpy as np
from PIL import Image
import torch.utils.data as D
from torchvision import transforms
from PIL import ImageFile



ImageFile.LOAD_TRUNCATED_IMAGES = True
trans = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()])


def get_img(root1, root2,root3,root4):
    img_train = []
    mask_train = []
    mask_test = []
    img_test = []
    for i in os.listdir(root1):
        if i.split('.')[1]=='png' or i.split('.')[1]=='jpg':
                img = Image.open(os.path.join(root1, i))
                img_train.append(img)
    for j in os.listdir(root2):
        if j.split('.')[1]=='png' or j.split('.')[1]=='jpg':
                mask = Image.open(os.path.join(root2, j))
                mask_train.append(mask)
    for k in os.listdir(root3):
        if k.split('.')[1]=='png' or k.split('.')[1]=='jpg':
                img = Image.open(os.path.join(root3, k))
                img_test.append(img)
    for l in os.listdir(root4):
        if l.split('.')[1]=='png' or l.split('.')[1]=='jpg':
                mask = Image.open(os.path.join(root4, l))
                mask_test.append(mask)

    return img_train, mask_train,img_test,mask_test

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def rotate(xb, yb, angle):
    img_w, img_h = xb.shape[:2]
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb

def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def data_augment(xb, yb):
    xb = np.array(xb)
    yb = np.array(yb)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)
    if np.random.random() < 0.25:
        xb = blur(xb)
    if np.random.random() < 0.2:
        xb = add_noise(xb)
    return Image.fromarray(xb), Image.fromarray(yb)

class MyDataset(D.Dataset):
    def __init__(self, img, mask, transform, train=True):
        self.img = img
        self.mask = mask
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        mask = self.mask[idx]
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask

def get_dataloader(BATCH_SIZE, root1, root2,root3,root4):
    img_train, mask_train,img_test,mask_test = get_img(root1, root2,root3,root4)
    train_loader = D.DataLoader(MyDataset(img_train, mask_train, trans), batch_size=BATCH_SIZE, shuffle=True,
                                pin_memory=True, num_workers=0)
    valid_loader = D.DataLoader(MyDataset(img_test, mask_test, trans, train=False), batch_size=BATCH_SIZE,
                                pin_memory=True, num_workers=0)

    return train_loader, valid_loader

train_loader, valid_loader = get_dataloader(4, './img_train', './label_train','./img_test','./label_test')
model=AttU_Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()
# criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
best_loss = float('inf')
model.to(device)

epochs=30
es = 0
for epoch in range(epochs):
    model.train()
    t_loss = 0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        img, mask = data
        optimizer.zero_grad()
        y_pred = model(img.to(device))
        loss = criterion(y_pred, mask.to(device))
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        scheduler.step(epochs)

    model.eval()
    v_loss = 0
    with torch.no_grad():
        val_bar = tqdm(valid_loader)
        for img1, mask1 in val_bar:
            v_img = img1
            v_mask = mask1
            output = model(v_img.to(device))
            loss1 = criterion(output, v_mask.to(device))
            v_loss += loss1.item()
    if v_loss < best_loss:
        es = 0
        best_loss = v_loss
        torch.save(model.state_dict(), f'./model/AtUnet.pkl')
    else:
        es += 1
        if es > 30:
            break
    scheduler.step(v_loss)
    print(f'epoch {epoch} | train loss:{t_loss / 9:.4f} | valid loss:{v_loss:.4f}')

