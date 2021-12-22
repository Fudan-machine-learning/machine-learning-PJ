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
        #print("filename in load_fabric_data is ")
       # print(filename)
        filename = filename.replace('\\' ,'/')
        
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
    ftype2 = [] #6-12 as group 6, 13 as group 7, 14 as group 8
    ftype2_dict = {num:6 for num in range(6, 13)}
    ftype2_dict.update({num:num for num in range(6)})
    ftype2_dict[13] = 7
    ftype2_dict[14] = 8 
    for i in fdata:
        ftype1.append(i['flaw_type'])
        ftype2.append(ftype2_dict[i['flaw_type']])
    return ftype1, ftype2

def load_fabric_images(path, fids, fdata, ftype):
    path += '**/**.jpg'
    labels = []
    imgs = []

    print(path)
    #random.sample(list(glob.iglob(path, recursive=True)), 50)
    for filename in glob.iglob(path, recursive=True):
        #find info about the image
        filename = filename.replace('\\' ,'/')


        fid = filename.split('/')[-1].split('.')[0]
     
        info = fdata[fids.index(fid)]
        
        filename_trgt = filename.replace("temp", "trgt")
        #filename_trgt = r"C:/Users/Administrator/Desktop/PRML/Project/fabric_data/trgt" + "/" +fid +'.jpg'
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
    return (labels, imgs)
