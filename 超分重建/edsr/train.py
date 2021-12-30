from model import edsr
from math import log10
import torch.nn as nn
import torch.optim as optim
import torch
from data_read import get_dataloader
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize=32
train_loader, valid_loader = get_dataloader(batchsize, 'train_imgLR', 'train_imgHR','test_imgLR','test_imgHR')
num=len(train_loader)
model = edsr().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)
epochs=150

for epoch in range(epochs):
	model.train()
	t_loss = 0
	for step_train, data_train in enumerate(tqdm(train_loader)):
		imgLR, imgHR = data_train
		optimizer.zero_grad()
		out = model(imgLR.to(device))
		loss = criterion(out, imgHR.to(device))
		t_loss += loss.item()
		loss.backward()
		optimizer.step()

	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, t_loss))
	model.eval()
	with torch.no_grad():
		for step_test, data_test in enumerate(tqdm(valid_loader)):
			imgLRt, imgHRt = data_test
			result = model(imgLRt.to(device))
			MSE = criterion(result, imgHRt.to(device))
			psnr = 10 * log10(1 / MSE.item())
			# sum_psnr += psnr
		print("**Average PSNR: {} dB".format(psnr))
		torch.save(model.state_dict(), f'logs/{round(psnr, 5)}.pkl')




