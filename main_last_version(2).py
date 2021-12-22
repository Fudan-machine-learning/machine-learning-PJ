import itertools

import numpy as np
from matplotlib import image
import sys
import os
import torch
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import random
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
#device_ids = [0,1,2]

from util import load_fabric_data, extract_label_grouping, extract_label_grouping, load_fabric_images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#加载数据
fids, fdata = load_fabric_data("./fabric_data/label_json/**/**.json")
ftype = extract_label_grouping(fdata)

#加载图片和标签
path = './fabric_data/temp/'
labels, imgs = load_fabric_images(path, fids, fdata, ftype)

n_samples = len(imgs)
print("Number of samples:", n_samples)


imgs = [cv2.resize(img, (200, 200)) for img in imgs]
imgs, labels = shuffle(imgs, labels, random_state=0)

#训练集和测试集划分
train_images, test_images, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.2, random_state=1)

print("#Training data: {}\n#Testing data: {}\n#Class: {}".format(len(train_images), len(test_images),
                                                                 len(set(train_labels))))

#转为np数组
train_images, test_images, train_labels, test_labels = np.array(train_images), np.array(test_images), np.array(
    train_labels), np.array(test_labels)

train_images, test_images = train_images / 255.0, test_images / 255.0


#接下来对特定类别进行旋转数据增强

#找某一类别数据的函数
def find_indices(listOfLabel, idx):
    res = []#类别为idx的数据索引
    i = 0
    for i in range(0, len(listOfLabel)):
        if (listOfLabel[i] == idx):
            res.append(i)

    return res

#对类别为3 5 7 9 10 12进行数据增强
label_five = find_indices(train_labels, idx=5)
label_three = find_indices(train_labels, idx=3)
label_seven = find_indices(train_labels, idx=7)
label_nine = find_indices(train_labels, idx=9)
label_ten = find_indices(train_labels, idx=10)
label_telw = find_indices(train_labels, idx=12)

train_is_three = []
train_is_five = []
train_is_seven = []
train_is_nine = []
train_is_ten = []
train_is_telw = []

#得到需要旋转数据增强的images

for i in label_five:
    train_is_five.append(train_images[i])

for i in label_three:
    train_is_three.append(train_images[i])

for i in label_seven:
    train_is_seven.append(train_images[i])

for i in label_nine:
    train_is_nine.append(train_images[i])

for i in label_ten:
    train_is_ten.append(train_images[i])

for i in label_telw:
    train_is_telw.append(train_images[i])


#旋转数据增强
def rotate_a_bunch(a_list_of_image):
    res = []
    for i in range(0, len(a_list_of_image)):
        rotate_once1 = np.rot90(a_list_of_image[i], k=1, axes=(0, 1))
        a = rotate_once1[::-1]
        rotate_twice1 = np.rot90(rotate_once1, k=1, axes=(0, 1))
        b = rotate_twice1[::-1]
        rotate_third1 = np.rot90(rotate_twice1, k=1, axes=(0, 1))
        c = rotate_third1[::-1]
        rotate_once2 = np.rot90(a_list_of_image[i], k=1, axes=(1, 0))
        d = rotate_once2[::-1]
        rotate_twice2 = np.rot90(rotate_once2, k=1, axes=(1, 0))
        e = rotate_twice2[::-1]
        rotate_third2 = np.rot90(rotate_twice2, k=1, axes=(1, 0))
        f = rotate_third2[::-1]

        new_gen = [rotate_once1, rotate_once2, rotate_twice1, rotate_twice2, rotate_third1, rotate_third2]
        new_gen2 = [a, b, c, d, e, f]
        res = new_gen + res + new_gen2
    return np.array(res)

#生成新图片
generated_5 = rotate_a_bunch(train_is_five)
generated_3 = rotate_a_bunch(train_is_three)
generated_7 = rotate_a_bunch(train_is_seven)
generated_9 = rotate_a_bunch(train_is_nine)
generated_10 = rotate_a_bunch(train_is_ten)
generated_12 = rotate_a_bunch(train_is_telw)

#新图片对应的标签
generated_5_label = [5] * len(generated_5)
generated_3_label = [3] * len(generated_3)
generated_7_label = [7] * len(generated_7)
generated_9_label = [9] * len(generated_9)
generated_10_label = [10] * len(generated_10)
generated_12_label = [12] * len(generated_12)

#把数据增强生成的图片加到训练集中
train_images = np.concatenate((train_images, generated_3, generated_5, generated_7, generated_9, generated_12))
train_labels = np.concatenate(
    (train_labels, generated_3_label, generated_5_label, generated_7_label, generated_9_label, generated_12_label))

print(len(train_images))
print(len(train_labels))

#网络结构定义

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
net.add_module("global_avg_pool", GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 15)))

#网络定义结束
#net = torch.nn.DataParallel(net, device_ids=device_ids) # 声明所有可用设备


#接下来是训练所有用到的函数

#k折数据获取函数

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    print('train size:', X_train.shape, 'test size:', X_part.shape)
    return X_train, y_train, X_valid, y_valid

#k折交叉验证
def k_fold(k, net, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, device):
    train_l_sum, valid_l_sum = 0, 0
    train_loss = []
    val_loss = []
    train_accuracy = []
    valid_accuracy = []
    for i in range(k):
        train_img, train_label, val_img, val_label = get_k_fold_data(k, i, X_train, y_train)
        print('fold %d' % i)
        train_ls, valid_ls, val_acc, train_acc = train(net, train_img, train_label, val_img, val_label, batch_size, weight_decay,
                                             learning_rate,
                                             device,
                                             num_epochs)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print('fold %d, train loss %f, valid loss %f' % (i, train_ls[-1], valid_ls[-1]))
        if i == 0:
            train_loss = train_ls
            val_loss = valid_ls
            valid_accuracy = val_acc
            train_accuracy = train_acc
        else:
            train_loss = np.concatenate((train_loss, train_ls), axis=0)
            val_loss = np.concatenate((val_loss, valid_ls), axis=0)
            train_accuracy = np.concatenate((train_accuracy, train_acc), axis=0)
            valid_accuracy = np.concatenate((valid_accuracy, val_acc), axis=0)
    return train_loss, val_loss, train_accuracy, valid_accuracy



class myDataset(Dataset):  # 定义自己的数据类myDataset，继承的抽象类Dataset
    def __init__(self, data, label):
        self.data = data  # 读取csv文件，并且赋给他本身
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, mask):  # 定义自己的数据类，必须重写这个方法（函数）
        data = self.data[mask]
        label = self.label[mask]  # 获取数据的方式，按照索引进行的
        return data, label

    def __len__(self):  # 定义自己的数据类，必须重写这个方法（函数）
        return self.length  # 返回的数据的长度

#计算测试集的准确率
def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available()
                                                          else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        test_l_sum = 0
        loss = torch.nn.CrossEntropyLoss()
        batch_count = 0
        for X, y in data_iter:
            X = X.permute(0, 3, 1, 2)
            X = X.float()
            if isinstance(net, torch.nn.Module):
                net.eval()
                l = loss(net(X.to(device)), y.to(device))
                test_l_sum += l.cpu().item()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if 'is_training' in net.__code__.co_varnames:
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum.item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            batch_count += 1
    return acc_sum / n, test_l_sum / batch_count

#混淆矩阵定义
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

#画混淆矩阵函数
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                verticalalignment='center',
                horizontalalignment="center",
                color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('matrix.png', format='png')

#引入混淆矩阵的计算测试集准确率函数
def evaluate_accuracy_matrix(data_iter, net, device, conf_matrix):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        test_l_sum = 0
        loss = torch.nn.CrossEntropyLoss()
        batch_count = 0
        for X, y in data_iter:
            X = X.permute(0, 3, 1, 2)
            X = X.float()
            if isinstance(net, torch.nn.Module):
                net.eval()
                out = net(X.to(device))
                l = loss(net(X.to(device)), y.to(device))
                test_l_sum += l.cpu().item()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                prediction = torch.max(out, 1)[1]
                conf_matrix = confusion_matrix(prediction, labels=y, conf_matrix=conf_matrix)
                net.train()
            n += y.shape[0]
            batch_count += 1
    return acc_sum / n, test_l_sum / batch_count


#训练，包括生成混淆矩阵
def train_matrix(model, train_features, train_labels, test_features, test_labels, batch_size, weight_decay,
              learning_rate,
              device,
              num_epochs,
                 conf_matrix):
    model = model.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    train_data = myDataset(data=train_features, label=train_labels)
    test_data = myDataset(data=test_features, label=test_labels)
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    batch_count = 0
    train_Loss = []
    test_Loss = []
    Test_acc = []
    Train_acc = []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.float()
            X = X.to(device)
            y = y.to(device)
            X = X.permute(0, 3, 1, 2)
            y_hat = model(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc, test_loss = evaluate_accuracy_matrix(test_iter, model, device, conf_matrix)
        train_Loss.append(train_l_sum / batch_count)
        test_Loss.append(test_loss)
        Test_acc.append(test_acc)
        Train_acc.append(train_acc_sum / n)
        print('epoch %d, train_loss %.4f, test_loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / batch_count, test_loss, train_acc_sum / n, test_acc))
    return train_Loss, test_Loss, Test_acc, Train_acc

#训练，不生成混淆矩阵
def train(model, train_features, train_labels, test_features, test_labels, batch_size, weight_decay,
              learning_rate,
              device,
              num_epochs):
    model = model.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    train_data = myDataset(data=train_features, label=train_labels)
    test_data = myDataset(data=test_features, label=test_labels)
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    batch_count = 0
    train_Loss = []
    test_Loss = []
    Test_acc = []
    Train_acc = []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.float()
            X = X.to(device)
            y = y.to(device)
            X = X.permute(0, 3, 1, 2)
            y_hat = model(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc, test_loss = evaluate_accuracy(test_iter, model)
        train_Loss.append(train_l_sum / batch_count)
        test_Loss.append(test_loss)
        Test_acc.append(test_acc)
        Train_acc.append(train_acc_sum / n)
        print('epoch %d, train_loss %.4f, test_loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / batch_count, test_loss, train_acc_sum / n, test_acc))
    return train_Loss, test_Loss, Test_acc, Train_acc


print("#Training data: {}\n#Testing data: {}\n#Class: {}".format(len(train_images), len(test_images),
                                                                 len(set(train_labels))))

#参数
k, lr, num_epochs, weight_decay, batch_size = 10, 0.001, 10, 0.0001, 256

conf_matrix = torch.zeros(15, 15)

#k折交叉验证
train_l, valid_l, train_accuracy, valid_accuracy = k_fold(k, net, train_images, train_labels, num_epochs, lr,
                          weight_decay,
                          batch_size,
                          device)

#保存模型
torch.save(net.state_dict(), '/home/yzzhang/workspace/test/model/net.pth')

#最后整体train，再用测试集测试
train_Loss, test_Loss, test_acc, train_acc = train_matrix(net, train_images, train_labels, test_images, test_labels,
                                                       batch_size,
                                                       weight_decay,
                                                       lr,
                                                       device,
                                                       num_epochs,
                                                          conf_matrix)

#k折整个过程的loss变化
plt.figure(1)
plt.plot(range(len(train_l)), train_l, label='train_loss')
plt.plot(range(len(valid_l)), valid_l, label='valid_loss')
plt.legend(loc='right', fontsize=10)  # 标签位置
plt.savefig('fold_loss.png', format='png')

#k折整个过程的acc变化
plt.figure(2)
plt.plot(range(len(train_accuracy)), train_accuracy, label='train_acc')
plt.plot(range(len(valid_accuracy)), valid_accuracy, label='valid_acc')
plt.legend(loc='right', fontsize=10)  # 标签位置
plt.savefig('fold_acc.png', format='png')

#最后一遍训练测试集和训练集loss变化
plt.figure(3)
plt.plot(range(len(train_Loss)), train_Loss, label='train_loss')
plt.plot(range(len(test_Loss)), test_Loss, label='test_loss')
plt.legend(loc='right', fontsize=10)  # 标签位置
plt.savefig('loss.png', format='png')

#最后一遍训练测试集和训练集acc变化
plt.figure(4)
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc)
plt.savefig('acc.png', format='png')

#绘制混淆矩阵
plt.figure(5)
classes = ["未知", "逃花", "塞网", '破洞', '缝头', '水渍', '脏污', '白条', '花糊', '坯疵', '沙眼', '拖色', '网折印', '无疵点', '未对齐']
plot_confusion_matrix(conf_matrix.numpy(), classes=classes, normalize=False,
                                 title='Normalized confusion matrix')
