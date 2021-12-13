import torch
import torch.nn as nn
from tqdm import tqdm
from  model import fcn
import torch.optim as optim
from data_read import get_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#（batch_size,训练集图像，测试集图像，训练集标签，测试集标签）
train_loader, valid_loader = get_dataloader(4, 'img_train', 'img_test','label_train','label_test')

#修改为类别数+1
model=fcn(num_class=2).to(device)

criterion =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
epochs=30

for epoch in range(epochs):
    model.train()
    t_loss = 0
    for step_train, data_train in enumerate(tqdm(train_loader)):
        img_train, label_train = data_train
        label_train=label_train.long()
        optimizer.zero_grad()
        pre = model(img_train.to(device))
        loss = criterion(pre, label_train.to(device))
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        scheduler.step(epochs)

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for step_test, data_test in enumerate(tqdm(valid_loader)):
            img_test ,label_test=data_test
            label_test=label_test.long()
            output = model(img_test.to(device))
            loss2 = criterion(output, label_test.to(device))
            v_loss += loss2.item()
            scheduler.step(v_loss)
        torch.save(model.state_dict(), f'{round(v_loss, 3)}.pkl')
    print('[epoch {}] train_loss: {}  val_loss: {}'
          .format(epoch + 1, round((t_loss), 3), round(v_loss, 3)))

