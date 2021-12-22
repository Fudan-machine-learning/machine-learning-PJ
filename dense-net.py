import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from util import load_fabric_data, extract_label_grouping, extract_label_grouping, load_fabric_images
import cv2
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗⼝形状设置成输⼊的⾼和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): 
        return x.view(x.shape[0], -1)

class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]
        
    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return data, label

    def __len__(self):
        return self.length

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim = 1)
        return X

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk

def Densenet():
    net = nn.Sequential(
        nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlock_%d" % i, DB)
        num_channels = DB.out_channels
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" %i, transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_channels, 15)
    ))
    return net

def evaluate_accuracy(data_iter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
    n += y.shape[0]
    return acc_sum / n

def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            
            print('X.dtype: ',X.dtype)
            
            print('y.dtype: ',y.dtype)
            
            X = X.to(device)
            y = y.to(device)
            X = X.permute(0, 3, 1, 2)
            X = np.float32(X)
            X = torch.from_numpy(X)
            y_hat = net(X)
            print('y_hat.dtype: ',y_hat.dtype)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

if __name__=="__main__":
    batch_size = 2
    lr, num_epochs = 0.001, 5
    net = Densenet()
    path = "fabric_data/label_json/**/**.json"
    fids, fdata = load_fabric_data(path)
    ftype1, ftype2 = extract_label_grouping(fdata)
    path = "fabric_data/temp/"
    labels, imgs = load_fabric_images(path, fids, fdata, ftype1)
    print(len(labels))
    imgs = [cv2.resize(img,(200, 200)) for img in imgs]
    train_img, test_img, train_label, test_label = train_test_split(imgs, labels, test_size=0.2, random_state=1)
    train_img, test_img, train_label, test_label = np.array(train_img), np.array(test_img), np.array(train_label), np.array(test_label)
    train_images, test_images = train_img / 255.0, test_img / 255.0   
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_set = MyDataSet(data = train_img, label = train_label)
    test_set = MyDataSet(data = test_img, label = test_label)
    train_iter = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_iter = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 0)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

