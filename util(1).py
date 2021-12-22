import json
import glob, os
from matplotlib import image
import numpy as np

def load_fabric_data(path):
    '''
    Loads data from fabric_data folder.
    Returns:
        (list, list)
        A list of id in the file name
        A list of dictionary file containing data from json (flaw_type, bbox)

    Sample usage: fid, fdata = load_fabric_data('fabric_data/label_json/**/**.json')
    '''
    fid = []
    fdata = []
    for filename in glob.iglob(path, recursive=True):
        fid.append(filename.split('/')[-1].split('.')[0])
        with open(filename) as f:
            fdata.append(json.load(f))
    return (fid, fdata)

def extract_label_grouping(fdata):
    '''
    Generates lists of labels according to different groupings.
    Type 1 grouping: original
    Type 2 grouping: 6-12 as group 6, 13 as group 7, 14 as group 8
    Type 3 grouping: only take 1,2,5 and 13
    '''
    ftype1 = [] #original
    for i in fdata:
        ftype1.append(i['flaw_type'])
    return ftype1

def load_fabric_images(path, fids, fdata, ftype):
    path += '**/**.jpg'
    labels = []
    imgs = []
    for filename in glob.iglob(path, recursive=True):
        #find info about the image
        fid = filename.split('/')[-1].split('.')[0]
        info = fdata[fids.index(fid)]
        filename_trgt = filename.replace("temp", "trgt")
        #get image
        size1 = os.stat(filename).st_size
        size2 = os.stat(filename_trgt).st_size
        if (size1 != 0) and (size2 != 0): 
            #load image
            img_data_temp = image.imread(filename)
            img_data_temp = img_data_temp[info['bbox']['y0']:info['bbox']['y1'], info['bbox']['x0']:info['bbox']['x1']]
            img_data_tgrt = image.imread(filename_trgt)
            img_data_tgrt = img_data_tgrt[info['bbox']['y0']:info['bbox']['y1'], info['bbox']['x0']:info['bbox']['x1']]
            if img_data_temp.shape == img_data_tgrt.shape:
                #append image
                imgs.append(np.concatenate([img_data_temp, img_data_tgrt], axis = 2))
                #append label
                labels.append(ftype[fids.index(fid)])

                #transpose
                img_data_tgrt_lr = np.fliplr(img_data_tgrt) # 左右翻转
                img_data_temp_lr = np.fliplr(img_data_temp) # 左右翻转
                imgs.append(np.concatenate([img_data_temp_lr, img_data_tgrt_lr], axis = 2))
                labels.append(ftype[fids.index(fid)])

                #transpose
                img_data_tgrt_ud = np.flipud(img_data_tgrt)  # 上下翻转
                img_data_temp_ud = np.flipud(img_data_temp)  # 上下翻转
                imgs.append(np.concatenate([img_data_temp_ud, img_data_tgrt_ud], axis=2))
                labels.append(ftype[fids.index(fid)])

    return (labels, imgs)
