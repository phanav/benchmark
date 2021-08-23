#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os, shutil
from os.path import join as joinpath
from os import listdir
import json


# In[15]:


"""
# Expected input format: an `images` folder with all images in flat structure, i.e. as direct children.
datadir = "/mnt/beegfs/home/vu/Codalab-MetaDL/data/resisc45/resisc45-resized-formatted-new"
imagedir = joinpath(datadir, "images")
resultdir = "/mnt/beegfs/home/vu/Codalab-MetaDL/result"

# path to metadata file with at least 2 columns: filename and label
# metadata_filepath = joinpath(datadir, "resisc45-filtered-metadata.csv")
metadata_filepath = joinpath(datadir, "labels.csv")

# names of these 2 columns
labelcolumn = "category"
filecolumn = "newfilename"

# result are saved in this folder inside the resultdir
dataname = "mini_resisc"
# prefix for output file
resultprefix = "mini_resisc-finetune"

random_seed = 2021
"""


# In[320]:


# Expected input format: an `images` folder with all images in flat structure, i.e. as direct children.
# datadir = "/mnt/beegfs/home/vu/Codalab-MetaDL/src/meta-album/meta-album-main/data/omniprint2"
datadir = "/mnt/beegfs/home/vu/Codalab-MetaDL/data/alldata/formatted-image/omniprint3-meta5bis_first_set"

imagedir = joinpath(datadir, "images")
resultdir = "/mnt/beegfs/home/vu/Codalab-MetaDL/result"

# path to metadata file with at least 2 columns: filename and label
# metadata_filepath = joinpath(datadir, "resisc45-filtered-metadata.csv")
metadata_filepath = joinpath(datadir, "labels.csv")

# names of these 2 columns
# labelcolumn = "CATEGORY"
# filecolumn = "FILE_NAME"

# result are saved in this folder inside the resultdir
dataname = "omniprint3_meta5bis_first_set"
# prefix for output file
resultprefix = "omniprint3_meta5bis_first_set-finetune-randomseed2022"

random_seed = 2022
print(f"{random_seed = }")


# In[321]:


info_filepath = joinpath(datadir, "info.json")

# if os.path.exists(info_filepath):
with open(info_filepath) as file:
    infofile = json.load(file)

labelcolumn = infofile['category_column_name']
filecolumn = infofile['image_column_name']             
# else:
#     labelcolumn = "CATEGORY"
#     filecolumn = "FILE_NAME"


# In[322]:



figdir = joinpath(resultdir, dataname, resultprefix, 'fig')
modeldir = joinpath(resultdir, dataname, resultprefix, 'model')
metricdir = joinpath(resultdir, dataname, resultprefix, 'metric')

for outputdir in (figdir, modeldir, metricdir):
    os.makedirs(outputdir, exist_ok=True)


# In[323]:


# import os, shutil
# from os.path import join as joinpath

from pathlib import Path

import sys, copy
import itertools, math

from functools import partial
import json

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# from tqdm import tqdm_notebook as tqdm
# from tqdm.autonotebook import tqdm
# from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics

import PIL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.autograd as autograd

import captum.attr
import scipy


# In[324]:


# libpath = "/mnt/beegfs/home/vu/Codalab-MetaDL/lib"
# if libpath not in sys.path: 
#     sys.path.append(libpath)
    
# import dropblock


# In[325]:


pd.options.mode.chained_assignment = None 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

pltparams = {
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    'figure.titlesize': 'x-large',
    'savefig.dpi': 600,
}
plt.rcParams.update(pltparams)

sns.set(font_scale = 1.2)


# ## Preprocess and explore data

# In[326]:


filtered_metadata = pd.read_csv(metadata_filepath, index_col=0)
# filtered_metadata = pd.read_csv(metadata_filepath)
filtered_frequency = filtered_metadata.value_counts(labelcolumn)
filtered_metadata.sample(5)


# In[327]:


plt.figure(figsize=(12, 4))
# ax = frequency[frequency >= 20].plot(kind="bar")
ax = filtered_frequency.plot(kind="bar")
ax.set_xticks([])
plt.suptitle(dataname)
# ax.figure.savefig(joinpath(figdir, f'{dataname}-distribution.png'), bbox_inches='tight')


# In[328]:


# class_weights = filtered_frequency.max()/filtered_frequency
# class_weights


# ## Load data

# In[329]:


def tensor_to_display(imagetensor):
    return imagetensor.numpy().transpose((1, 2, 0))

def show_images(metadata, imagedir, rows=2, columns=5, figsize=(16, 8), title=None):
    sns.set_style("dark")

    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten()
    
    for index, ax in enumerate(axes):
        imageinfo = metadata.iloc[index]
        image = PIL.Image.open(joinpath(imagedir, imageinfo[filecolumn]))
        width, height = image.size
#         print(width,height)
        ax.imshow(image)
        ax.set_title(imageinfo[labelcolumn])
#         plt.axis('off')

    if title is not None:
        plt.suptitle(title)
    
    return fig

fig = show_images(filtered_metadata.sample(10), imagedir, title=dataname)

# fig.savefig(joinpath(basedir, 'fig', f'{dataname}-sample-image.png'))


# In[330]:


def encode_label(metadata):
    labelcode = LabelEncoder()
    metadata.loc[:,"labelcode"] = labelcode.fit_transform(metadata[labelcolumn])
    return metadata, labelcode


# In[331]:


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, imagedir, transform=None):
        self.metadata = metadata
        self.imagedir = imagedir
        self.transform = transform
    
    def __getitem__(self, index):
        imageinfo = self.metadata.iloc[index]
        
        imagedata = PIL.Image.open(os.path.join(self.imagedir, imageinfo[filecolumn])) 
        transformed_image = self.transform(imagedata) if self.transform else imagedata
        label = imageinfo["labelcode"]
#         print(transformed_image.shape)

        return transformed_image, label

    def __len__(self):
        return len(self.metadata)
    

def get_dataloader(dataset, batchsize, ifshuffle):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=ifshuffle)
    return dataloader


# In[332]:


# filtered_metadata = filter_metadata(metadata, frequency)
processed_metadata, labelcode = encode_label(filtered_metadata)
train_metadata, valid_metadata = train_test_split(processed_metadata, test_size=0.5, stratify=processed_metadata[labelcolumn], random_state=2021)
# train_metadata, valid_metadata = train_test_split(processed_metadata, train_size=1/8, stratify=processed_metadata[labelcolumn], random_state=2021)
print(train_metadata.shape, valid_metadata.shape)

numberclass = valid_metadata["labelcode"].max() + 1
labelnames = labelcode.inverse_transform(range(numberclass))
print(numberclass)


# In[333]:


def save_train_test_split(train_metadata, test_metadata, output_path):
    train_metadata['partition'] = 'train'
    valid_metadata['partition'] = 'test'
    
    all_metadata = pd.concat([train_metadata, valid_metadata]).drop(columns='labelcode')
    all_metadata.to_csv(output_path, index=False)
    print(all_metadata.sample(5))
    
    return output_path

# save_train_test_split(train_metadata, valid_metadata, joinpath(datadir, 'sd128_v1_crop_test025.csv'))


# In[334]:


def train_transform(pil_image, imagesize):
    width, height = pil_image.size
    transform = transforms.Compose([
        transforms.CenterCrop(min(width, height)),
        transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize(imagesize),
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image)

def test_transform(pil_image, imagesize):
    transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.ToTensor(),
    ])
    return transform(pil_image)


def build_data(imagesize, train_metadata, valid_metadata, imagedir, train_transform, test_transform, batchsize=128):
#     imagesize = (128, 128)
#     transform0 = partial(transform_image, imagesize=(imagesize))

    train_dataset = ImageDataset(train_metadata, imagedir, train_transform)
    train_dataloader = get_dataloader(train_dataset, batchsize=batchsize, ifshuffle=True)

    valid_dataset = ImageDataset(valid_metadata, imagedir, test_transform)
    valid_dataloader = get_dataloader(valid_dataset, batchsize=batchsize, ifshuffle=False)

    return train_dataset, valid_dataset, train_dataloader, valid_dataloader


imagesize = (128, 128)
train_transform0 = partial(train_transform, imagesize=(imagesize))
test_transform0 = partial(test_transform, imagesize=imagesize)
train_dataset, valid_dataset, train_dataloader, valid_dataloader = build_data(
    imagesize, train_metadata, valid_metadata, imagedir, train_transform0, test_transform0)


# In[335]:


def display_tensor(dataloader, labelcode):
    inputs, labels = next(iter(dataloader))
    print(inputs.shape, labels.shape)
    for inpt, label in zip(inputs[:4], labels[:4]):
        plt.figure()
        plt.imshow(tensor_to_display(inpt))
        plt.title(labelcode.inverse_transform([label]))


# In[336]:


inputs, labels = next(iter(train_dataloader))
print(inputs.shape, labels.shape)
for inpt, label in zip(inputs[:4], labels[:4]):
    plt.figure()
    plt.imshow(tensor_to_display(inpt))
    plt.title(labelcode.inverse_transform([label]))


# In[337]:


# for i, (inpt, label) in enumerate(train_dataset):
#     if inpt.shape[0] != 3:
#         print('train index: ', i)
    
# for i, (inpt, label) in enumerate(valid_dataset):
#     if inpt.shape[0] != 3:
#         print('valid index:', i)
    


# ## Define training

# In[338]:


def onebatch(model, inputs, labels, lossfunc, device):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    loss = lossfunc(outputs, labels)
    
    predictions = torch.argmax(outputs, dim=1)  
#     correct_count = (predictions == labels).sum()

    return loss, predictions, labels

def score_predictions(predictions, targets):
    score = sklearn.metrics.balanced_accuracy_score(targets, predictions)
    return score

scorename = 'balanced_accuracy'


# In[339]:


def allbatch(model, dataloader, lossfunc, device, ifbackward, optim=None, batch_lr_scheduler=None):
    sum_loss = 0; predictions = []; targets = []
    losses = []
    sample_count = 0

    for inputs, labels in dataloader:
        if ifbackward: optim.zero_grad()
            
        batch_loss, batch_prediction, batch_target = onebatch(model, inputs, labels, lossfunc, device)
#         sum_loss += batch_loss.item()
        losses.append(batch_loss)
        sample_count += len(labels)
    
        predictions.append(batch_prediction)
        targets.append(batch_target)
            
        if ifbackward:
            batch_loss.backward()
            optim.step()
            
        if batch_lr_scheduler: batch_lr_scheduler.step()
    
    predictions = torch.cat(predictions).cpu()
    targets = torch.cat(targets).cpu()
    losses = torch.Tensor(losses)
    score = score_predictions(predictions, targets)
    
    return losses.sum().item()/sample_count, score


# In[340]:


def build_result(model, score, loss, epoch):
    result = dict(model=model, score=score, loss=loss, epoch=epoch)
    return result

def update_best_result(best_result, current_result):
    if current_result['score'] > best_result['score']:
        best_result['score'] = current_result['score']
        best_result['model'] = copy.deepcopy(current_result['model'])
        best_result['loss'] = current_result['loss']
        best_result['epoch'] = current_result['epoch']
        
    return  best_result


# In[341]:


metricnames = "train_loss train_score valid_loss valid_score".split(' ')

def init_metric(metricnames):
    metric = {metricname:[] for metricname in metricnames}
    return metric

def update_metric(metric, updatedict):
    for metricname, metricvalue in updatedict.items():
        metric[metricname].append(metricvalue)
    
    return metric
    
# def adjust_metric(dataset, *metrics):
#     scaled_metrics = (metric.item() / len(dataset) for metric in metrics)
#     return scaled_metrics


# In[342]:


def train(model, train_dataloader, valid_dataloader, train_dataset, valid_dataset, optim, lossfunc, 
          device, metricnames, 
          epoch_lr_scheduler=None, batch_lr_scheduler=None,
          epochs=5):

    metric = init_metric(metricnames)
    
#     for epoch in tqdm(range(epochs)):
    best_result = build_result(model, 0, 0, 0)
    for epoch in range(1, epochs+1):
        model.train()

        train_epoch_loss, train_epoch_score = allbatch(model, train_dataloader, lossfunc, device, 
                                                       ifbackward=True, optim=optim, batch_lr_scheduler=batch_lr_scheduler)
        metric = update_metric(metric, dict(train_loss=train_epoch_loss, train_score=train_epoch_score))
    
        
        model.eval()
        with torch.no_grad():
            valid_epoch_loss, valid_epoch_score = allbatch(model, valid_dataloader, lossfunc, device, 
                                                           ifbackward=False, optim=None)
        
        metric = update_metric(metric, dict(valid_loss=valid_epoch_loss, valid_score=valid_epoch_score))
        
        
        current_result = build_result(model, valid_epoch_score, valid_epoch_loss, epoch)
        best_result = update_best_result(best_result, current_result)
        
        if (epochs <= 10) or (epoch % (epochs//10) == 0):
            print(f"\nEpoch {epoch} / {epochs}")
            print(f" train loss = {train_epoch_loss} ; train score = {train_epoch_score}")
            print(f" valid loss = {valid_epoch_loss} ; valid score = {valid_epoch_score}")
            
        if epoch_lr_scheduler:
            epoch_lr_scheduler.step(valid_epoch_loss)
#             print('Epoch: ', epoch,' LR:', epoch_lr_scheduler.get_last_lr())

    # end epoch loop
    print("\nBest valid score loss epoch: ", best_result['score'], best_result['loss'], best_result['epoch'])
    return best_result, metric
        


# ## Model

# In[343]:


def save_model(model, modelpath):
    parent, filename = os.path.split(modelpath)
    if parent: os.makedirs(parent, exist_ok=True)
    torch.save(model, modelpath)
    
def save_result(result, resultpath):
    parent, filename = os.path.split(resultpath)
    if parent: os.makedirs(parent, exist_ok=True)
    with open(resultpath, "w") as file:
        json.dump(result, file)    

def load_model(modelpath):
    model = torch.load(modelpath)
    return model

def save_checkpoint(model, optimizer, loss, savepath, epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, savepath)
    return savepath

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss
    


# In[344]:


def show_setting():
    #require python >= 3.8
    print(f"{dataname = }")
    print(f"{imagesize = }")
    print(f"{modelname = }")
    print(f"{scorename = }")
    print(f"{metricnames = }")
    
def get_result_name():
#     resultname = f"{dataname}-{modelname}-imagesize_{imagesize[0]}x{imagesize[1]}-{scorename}"
    resultname = f"{dataname}-{modelname}-imagesize_{imagesize[0]}x{imagesize[1]}"
    return resultname


# In[345]:


def get_optim_param_dict(model, lrrange):
    paramgroups = {}
    namegroup = {}
    
    for name, param in model.named_parameters():
        groupname = '.'.join(name.split('.')[:3])
        
#         layergroup = namegroup.setdefault(groupname, [])
#         layergroup.append(name)
#         namegroup[groupname] = layergroup
    
        paramgroup = paramgroups.setdefault(groupname, [])
        paramgroup.append(param)
        paramgroups[groupname] = paramgroup
    
    lrgroup = np.geomspace(*lrrange, num=len(paramgroups))
#     optim_param = [dict(params=group, lr=lr, weight_decay=1e-3) for group, lr in zip(paramgroups.values(), lrgroup)]
    print(namegroup)
    return optim_param

def get_optim_param_dict_resnet(model, lrrange, weightdecay):
    paramgroups = {}
    namegroup = {}
    
    for name, param in model.named_parameters():
#         print(name)
        groupname = '.'.join(name.split('.')[:2])
        
        if groupname.startswith("fc"):
            groupname = "fc"
        
        layergroup = namegroup.setdefault(groupname, [])
        layergroup.append(name)
        namegroup[groupname] = layergroup
    
        paramgroup = paramgroups.setdefault(groupname, [])
        paramgroup.append(param)
        paramgroups[groupname] = paramgroup
    
    lrgroup = np.geomspace(*lrrange, num=len(paramgroups))
#     optim_param = [dict(params=group, lr=lr, weight_decay=0.1) for group, lr in zip(paramgroups.values(), lrgroup)]
    optim_param = [dict(params=group, lr=lr, weight_decay=weightdecay) for group, lr in zip(paramgroups.values(), lrgroup)]

    return optim_param
    
# optim_param = get_optim_param_dict(model, (1e-6, 1e-3))
# len(optim_param)


# In[346]:


def get_optimizer(model, lr=1e-4, weightdecay=1e-3):
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weightdecay)
#     optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)
#     optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightdecay)

    return optim

def get_lrrange_optimizer(model, lrrange, get_paramdict_func, weightdecay=1e-3):
#     optim = torch.optim.Adam(get_optim_param_dict(model, lrrange), lr=lrrange[-1], weight_decay=1e-3)
#     optim = torch.optim.Adam(get_paramdict_func(model, lrrange), lr=lrrange[-1])
#     optim = torch.optim.SGD(get_paramdict_func(model, lrrange), lr=lrrange[-1], weight_decay=0.1, momentum=0)
    optim = torch.optim.SGD(get_paramdict_func(model, lrrange, weightdecay), lr=lrrange[-1])

    return optim

def get_epoch_lr_scheduler(optimizer, epochs=None):
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     milestones = [int(epochs*fraction) for fraction in (3/4, 7/8)]
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, verbose=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
    return scheduler

def get_batch_lr_scheduler(optimizer, dataloader, epochs):
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3,step_size_up=10,mode="triangular2")
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(dataloader), epochs=epochs)
    return scheduler


# ## Run training

# In[347]:


def get_model_resnet_dropblock(outsize):
    model = models.resnet18(pretrained=True)
  
    drop_proba = dict(layer4=0.1, layer3=0.1)
    for layer, sublayer, module in itertools.product(drop_proba.keys(), [0, 1], "conv1 conv2".split()):
        modulename = f"""model.{layer}[{sublayer}].{module}"""
        command = f"""{modulename} = nn.Sequential(dropblock.DropBlock2D(drop_prob={drop_proba[layer]}, block_size=3), {modulename})"""
#         print(command)
        exec(command)
    
    for layer, proba in drop_proba.items():
        command = f"""conv_outchannels = model.{layer}[0].conv2[1].out_channels"""
        exec(command)
        modulename = f"""model.{layer}[0].bn2"""
        command = f"""{modulename} = nn.Sequential(dropblock.DropBlock2D(drop_prob=proba, block_size=3), {modulename})"""
        exec(command)  
    
    conv_outsize = model.fc.in_features
    model.fc = nn.Linear(conv_outsize, outsize)
    
    return model.to(device)


# In[348]:


def get_model_resnet_dropout(outsize, with_dropout=True):
    model = models.resnet18(pretrained=True)
    if with_dropout:
        drop_proba = dict(layer4=0.1, layer3=0.1)
    #     drop_proba = dict(layer4=0.2)

        for layer, sublayer, module in itertools.product(drop_proba.keys(), [0,1], "conv1 conv2".split()):
            modulename = f"""model.{layer}[{sublayer}].{module}"""
            command = f"""{modulename} = nn.Sequential(nn.Dropout({drop_proba[layer]}), {modulename})"""
    #         print(command)
            exec(command)

        for layer, proba in drop_proba.items():
            command = f"""conv_outchannels = model.{layer}[0].conv2[1].out_channels"""
            exec(command)
            modulename = f"""model.{layer}[0].bn2"""
            command = f"""{modulename} = nn.Sequential(nn.Dropout(proba), {modulename})"""
            exec(command)
                
#     prefixes = "conv1 bn1 layer1 layer2 layer3 layer4.0".split()
#     for name, params in model.named_parameters():
#         if any(name.startswith(prefix) for prefix in prefixes):
#             params.requires_grad = False

    conv_outsize = model.fc.in_features
    model.fc = nn.Linear(conv_outsize, outsize)

    return model.to(device)


# In[349]:


# models.resnet18()


# For omniprint, weight decay = 1, 100 epochs

# In[350]:


# max no droppout weight decay = 20
modelname = 'resnet18_sgd_weightdecay1_dropout0101_lr_1e-3'
resultname = resultprefix + '-' + modelname
print(resultname)

checkpoint_path = joinpath(modeldir, resultname + '-checkpoint.pth')

lossfunc = nn.CrossEntropyLoss(reduction='sum')
# lossfunc = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights.values).to(device), reduction='sum')

# model = get_model()
model = get_model_resnet_dropout(numberclass, with_dropout=True)
# model = get_model_resnet_dropblock(numberclass)
# best lr=1e-4

epochs = 20

optim = get_optimizer(model, lr=1e-3, weightdecay=1)
# optim = get_lrrange_optimizer(model, lrrange=(1e-6, 1e-3), get_optim_param_dict)
# optim = get_lrrange_optimizer(model, (1e-6, 1e-3), get_optim_param_dict_resnet, weightdecay=10)

lr_scheduler = get_epoch_lr_scheduler(optim, epochs)
# lr_scheduler = get_batch_lr_scheduler(optim, train_dataloader, epochs)


# In[351]:


result0, metric0 = train(model, train_dataloader, valid_dataloader, train_dataset, valid_dataset, optim, lossfunc, device, metricnames, 
                         epochs=epochs, epoch_lr_scheduler=lr_scheduler, batch_lr_scheduler=None)

save_checkpoint(result0['model'], optim, metric0['train_loss'][-1], checkpoint_path, epochs)
save_result(metric0, joinpath(metricdir, resultname + ".json"))

# save_model(result0['model'], joinpath(basedir, "model", f"{dataname}-{modelname}-imagesize_{imagesize[0]}x{imagesize[1]}.pth"))


# In[352]:


def get_old_result_metric(model, old_result_path):
    with open(old_result_path) as resultjson:
        metric0 = json.load(resultjson)
    best_index = np.argmax(metric0['valid_score'])
    
    result0 = build_result(model, metric0['valid_score'][best_index], metric0['valid_loss'][best_index], best_index+1)
    return result0, metric0
        


# In[353]:


# lossfunc = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights.values).to(device), reduction='sum')
# model = get_model()
# optim = get_optimizer(model, lr=1e-4)
# epochs=10

# model, optimizer, epoch, loss = load_checkpoint(checkpoint_path, model, optim)

# result1, metric1 = train(model, train_dataloader, valid_dataloader, train_dataset, valid_dataset, optim, lossfunc, device, metricnames, 
#                          epochs=epochs, epoch_lr_scheduler=None, batch_lr_scheduler=None)


# In[354]:


def merge_result(result0, metric0, result1, metric1):
    merged_metric = {metricname0: metricvalue0 + metricvalue1 
                     for (metricname0, metricvalue0), (metricname1, metricvalue1) 
                     in zip(metric0.items(), metric1.items())}
    merged_result = result0 if result0['score'] > result1['score'] else result1
    
    return merged_result, merged_metric


# In[355]:


final_result, final_metric = result0, metric0

# result0, metric0 = get_old_result_metric(model, joinpath(metricdir, resultname + '.json'))
# final_result, final_metric = merge_result(result0, metric0, result1, metric1)
# save_checkpoint(final_result['model'], optim, final_metric['train_loss'][-1], checkpoint_path, epoch=40)
# save_result(final_metric, joinpath(metricdir, resultname + ".json"))


# ## Analyze train metric

# In[356]:


def plot_train_metric(metric, title=None):
    sns.set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    
    axes[0].plot(metric['train_loss'], label='train_crossentropy')
    axes[0].plot(metric['valid_loss'], label='valid_crossentropy')
    axes[0].set_xlabel('epoch')
    axes[0].legend()

    axes[1].plot(metric['train_score'], label='train_score')
    axes[1].plot(metric['valid_score'], label='valid_score')
    axes[1].set_xlabel('epoch')
    axes[1].legend()
    
    if title: fig.suptitle(title)

    return fig


# In[357]:


# fig = plot_train_metric(metric0, resultname)
# sns.set_style("whitegrid")

fig = plot_train_metric(final_metric, resultname)
fig.savefig(joinpath(figdir, f"{resultname}.png"), bbox_inches='tight')


# ## Load result

# In[358]:


# imagesize=(128, 128)
# modelname = 'resnet18'
# resultname = get_result_name()

# checkpoint_path = joinpath(modeldir, resultname + '-checkpoint.pth')
# # model = get_model()
# model = get_model_resnet(numberclass)
# optim = get_optimizer(model, lr=1e-4)
# model, optimizer, epoch, loss = load_checkpoint(checkpoint_path, model, optim)
# model.eval();
# # result1, metric1 = train(model, train_dataloader, valid_dataloader, train_dataset, valid_dataset, optim, lossfunc, device, metricnames, 
# #                          epochs=epochs, epoch_lr_scheduler=None, batch_lr_scheduler=None)


# In[359]:


# def load_plot_metric(metricpath, resultname):
#     with open(metricpath) as resultjson:
#         metric = json.load(resultjson)
# #         metric['train_score'] = metric1['train_accuracy']
# #         metric['valid_score'] = metric1['valid_accuracy']

#     fig = plot_train_metric(metric, resultname)
    
#     return metric, fig
    
# # sns.set_theme()
# metric0, fig = load_plot_metric(joinpath(metricdir, resultname + '.json'), resultname)
# # fig.suptitle('sd128_v1_crop-mobilenetv2-imagesize128')
# fig.savefig(joinpath(figdir, resultname + "-balanced_accuracy.png"))


# ## All metrics

# In[360]:


def predict_batch(model, inputs):
    outputs = model(inputs)
#     _, predictions = torch.argmax(outputs, dim=1)
    return outputs

def get_prediction_target(model, dataloader):
    targets = []
    outputs = []
    with torch.no_grad():
        model.eval()
        for inpt, target in dataloader:
            outputs.append(predict_batch(model, inpt.to(device)))
            targets.append(target)
    outputs = torch.cat(outputs).cpu()
    targets = torch.cat(targets).cpu()
    
    return outputs, targets


# In[361]:


# %%time
# valid_outputs, valid_targets = get_prediction_target(final_result['model'].eval(), valid_dataloader)

valid_outputs, valid_targets = get_prediction_target(model.eval(), valid_dataloader)
valid_predictions = torch.argmax(valid_outputs, dim=1)
print(f"{valid_outputs.shape = } ; {valid_targets.shape = } ; {valid_predictions.shape = }")


# In[362]:


def report_metric(targets, predictions, labelnames):
    metric_report = pd.DataFrame(sklearn.metrics.classification_report(
        targets, predictions, output_dict=True, target_names=labelnames))
    return metric_report.T

metric_report = report_metric(valid_targets, valid_predictions, labelnames)
metric_report


# In[363]:


def score_per_class(outputs, targets, predictions, labelcode):
    numclass = targets.max() + 1
#     scores = dict(auroc=[], auprc=[])
    metrics = 'auroc auprc matthews_corr cohen_kappa'.split(' ')
    scores = {metric: [] for metric in metrics}
    
    for labelindex in range(numclass):
        binary_targets = (targets == labelindex)
        binary_predictions = (predictions == labelindex)
        selected_outputs = outputs[:,labelindex]
        
        scores['auroc'].append(sklearn.metrics.roc_auc_score(binary_targets, selected_outputs))
        scores['auprc'].append(sklearn.metrics.average_precision_score(binary_targets, selected_outputs))
        scores['matthews_corr'].append(sklearn.metrics.matthews_corrcoef(binary_targets, binary_predictions))
        scores['cohen_kappa'].append(sklearn.metrics.cohen_kappa_score(binary_targets, binary_predictions))
#         scores['balanced_accuracy'] = sklearn.metrics.balanced_accuracy_score(targets, predictions)
                                                    
    scores = pd.DataFrame(scores, index=labelcode.inverse_transform(range(numberclass)))
    return scores

scores = score_per_class(F.softmax(valid_outputs, dim=1), valid_targets, valid_predictions, labelcode)
scores


# In[364]:


def merge_metric(report, scores):
    merged = pd.merge(report, scores, how='left' ,left_index=True, right_index=True)
    return merged

all_metrics = merge_metric(metric_report, scores)
all_metrics.to_csv(joinpath(metricdir, resultname + '-all_metric.csv'))
all_metrics


# ## Load metric

# In[365]:


# imagesize=(128, 128)
# modelname = 'mobilenetv2'
# resultname = get_result_name()

# all_metrics = pd.read_csv(joinpath(metricdir, resultname + '-all_metric.csv'), index_col=0)
# # all_metrics = pd.read_csv(joinpath(basedir, 'result/sd128_v1_crop/sd128_v1_crop-mobilenetv2-imagesize_128x128-balanced_accuracy-all_metric.csv'), index_col=0)
# all_metrics


# ## Easiest and hardest labels

# In[366]:


class_metrics = all_metrics.dropna().drop(columns='support')
# class_metrics.sort_values(by='matthews_corr').iloc[:10]


# In[367]:


# # class_metrics.sort_values(by='matthews_corr')[:20].plot(kind='bar')
sns.set_theme()
# sns.set(font_scale = 1.2)

fig = plt.figure(figsize=(20,4))
fig = class_metrics['matthews_corr'].sort_values().plot(kind='bar', figsize=(20,4)).figure
fig.suptitle("Matthews correlation ascending")
fig.savefig(joinpath(figdir, resultname + '-matthewscorr_ascend.png'), bbox_inches='tight')


# ## Plot AUROC

# In[368]:


# def plot_auroc(outputs, targets, predictions, labelcode, labels, rows=2, columns=4, figsize=(8, 8)):
# #     fig, axes = plt.subplots(rows, columns, figsize=figsize)
# #     axes = axes.flatten()
# #     for ax, label in zip(axes, labels):
    
#     fig = plt.figure(figsize=figsize)
#     for label in labels:
#         labelindex = labelcode.transform([label])[0]
#         binary_targets = (targets == labelindex)
# #         binary_predictions = (predictions == labelindex)
#         selected_outputs = outputs[:,labelindex]

#         fpr, tpr, threshold = sklearn.metrics.roc_curve(binary_targets,  selected_outputs)
# #         auroc = sklearn.metrics.roc_auc_score(binary_targets,  selected_outputs)
#         auroc = sklearn.metrics.auc(fpr, tpr)
#     #         axes[axindex]
#         plt.plot(fpr,tpr, label=f"{label} AUROC={auroc:.2f}")
#         plt.legend(loc=4)
#         plt.plot([0,1], [0,1], linestyle='--', color='black')
#         plt.ylabel('True Positive Rate', fontsize=16)
#         plt.xlabel('False Positive Rate', fontsize=16)
        
#     return fig

# fig = plot_auroc(F.softmax(valid_outputs, dim=1), valid_targets, valid_predictions, labelcode, 
#                 class_metrics.sort_values(by='auroc').iloc[:5].index.values)


# In[369]:


# sns.set_theme()
# plt.figure(figsize=(16, 6))
# sns.violinplot(data=class_metrics['matthews_corr cohen_kappa f1-score auprc auroc'.split(' ')])
# # sns.violinplot(data=class_metrics)


# In[370]:


# #https://stackoverflow.com/questions/23556153/how-to-put-legend-outside-the-plot-with-pandas
# def plot_all_metrics(all_metrics, figsize=(16,4), title=None):
#     sns.set_theme()
#     fig = all_metrics.dropna().drop(columns='support').plot(kind='bar', figsize=figsize).legend(loc='lower left', bbox_to_anchor=(1.0, 0)).figure
#     if title: plt.title(title)
    
#     return fig
        
# fig = plot_all_metrics(all_metrics, title=resultname[:resultname.rfind('-')])
# # fig.savefig(joinpath(figdir, resultname[:resultname.rfind('-')] + '-all_metrics.png'), bbox_inches='tight')


# ## Compare metrics and confusion matrix

# In[371]:


# def get_plot_confusion_matrix(predictions, targets, labelcode, title=None):
#     confusion = sklearn.metrics.confusion_matrix(targets, predictions)
#     labelnames = labelcode.inverse_transform(range(targets.max().item() + 1))
    
#     confusion = pd.DataFrame(confusion, columns=labelnames, index=labelnames)
#     confusion.index.name = 'Actual'
#     confusion.columns.name = 'Predicted'
    
#     plt.figure(figsize = (12,10))
# #     plt.tight_layout(pad=2)
#     sns.set(font_scale=1.1)
#     fig = sns.heatmap(confusion, cmap="Blues", annot=True, annot_kws={"size": 12}).figure
    
#     if title: plt.suptitle(title)
    
#     return confusion, fig

# confusion, fig = get_plot_confusion_matrix(valid_predictions, valid_targets, labelcode, 
#                                            resultname[:resultname.rfind('-')] + "-confusion")
# # fig.savefig(joinpath(figdir, f"{resultname}-confusion.png"), bbox_inches='tight')
# #https://stackoverflow.com/questions/37427362/plt-show-shows-full-graph-but-savefig-is-cropping-the-image/37428142


# In[ ]:





# In[372]:


def get_confusion_matrix(targets, predictions, labelcode):
    confusion = sklearn.metrics.confusion_matrix(targets, predictions)
    labelnames = labelcode.inverse_transform(range(targets.max().item() + 1))

    confusion = pd.DataFrame(confusion, columns=labelnames, index=labelnames)
    confusion.index.name = 'Actual'
    confusion.columns.name = 'Predicted'
    
    return confusion

confusion = get_confusion_matrix(valid_targets.detach().cpu(), valid_predictions.detach().cpu(), labelcode)


# In[373]:


def plot_heatmap_4crop(weightmatrix, figsize=(20, 20), title=None):
    mid = len(weightmatrix) // 2
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
#     axes[0].plot(sns.heatmap(weightmatrix.iloc[:mid, :mid], cmap="Reds"))
    sns.heatmap(weightmatrix.iloc[:mid, :mid], cmap="Reds", ax=axes[0], square=True, annot=True)
    sns.heatmap(weightmatrix.iloc[:mid, mid:], cmap="Reds", ax=axes[1], square=True, annot=True)
    sns.heatmap(weightmatrix.iloc[mid:, :mid], cmap="Reds", ax=axes[2], square=True, annot=True)
    sns.heatmap(weightmatrix.iloc[mid:, mid:], cmap="Reds", ax=axes[3], square=True, annot=True)
    plt.tight_layout()
    
    if title: plt.suptitle(title)
    
    return fig

# fig = plot_heatmap_4crop(confusion, title=f"{resultname}-confusion-4crop")
# fig.savefig(joinpath(figdir, f"{resultname}-confusion.png"), bbox_inches='tight')


# ## GradCAM

# In[374]:


def get_attribution_layer(model):
#     layer = list(model.base.features.children())[-1]
    layer = model.layer4[1].conv2
    return layer

trained_model = final_result['model']
# trained_model = model
layer_gradcam = captum.attr.LayerGradCam(trained_model, get_attribution_layer(trained_model))


# In[375]:


def sample_inference(model, dataset, device, samplesize=1):
    sampleindex = np.random.randint(0, len(dataset), samplesize)
    inpts =[]; labels = []
    
    for index in sampleindex:
        inpt, label = dataset[index]
        inpts.append(inpt)
        labels.append(label)
    
    inpts = torch.stack(inpts).to(device)
    labels = torch.tensor(labels).to(device)
    
    predictions = predict_batch(model, inpts)
    
    return inpts, labels, predictions
    


# In[376]:


def plot_gradcam(layer_gradcam, inpt, label, prediction, labelcode, title=None):
    gradcam_attr = layer_gradcam.attribute(inpt.unsqueeze(0).to(device), label, relu_attributions=False)

    # upsample CAM to original image size
    gradcam_upsample = scipy.ndimage.interpolation.zoom(
      gradcam_attr.squeeze(0).cpu().detach().numpy(), 
      np.array(inpt.shape)/np.array(gradcam_attr.shape[1:]),
    )
#     gradcam_upsample = captum.attr.LayerAttribution.interpolate(
#         gradcam_attr.cpu().detach(), inpt.shape[-2:]).squeeze(0)
#     print(gradcam_upsample.shape)

    fig, ax = captum.attr.visualization.visualize_image_attr_multiple(
        np.transpose(gradcam_upsample, (1,2,0)), 
        tensor_to_display(inpt.cpu().detach()),
        methods = 'original_image blended_heat_map'.split(' '),
        signs='absolute_value absolute_value'.split(' '),
        alpha_overlay=0.6,
        cmap='viridis',
        titles=[f'truth: {labelcode.inverse_transform([label.item()])[0]}', 
                f'prediction: {labelcode.inverse_transform([prediction.item()])[0]}'],
        show_colorbar=True,
    )
    
    if title: fig.suptitle(title)
        
    return fig


# In[377]:


# sample_inputs, sample_labels, sample_predictions = sample_inference(trained_model, valid_dataset, device)
# fig = plot_gradcam(layer_gradcam, sample_inputs[0], sample_labels[0], sample_predictions[0], labelcode, title=resultname+'gradcam')

# selected_label = class_metrics['matthews_corr'].sort_values().index[7]
# # select_mask = ((valid_predictions == valid_targets) & (valid_targets == labelcode.transform([selected_label])[0]))
# select_mask =  (valid_targets == labelcode.transform([selected_label])[0])
# selected_index = np.where(select_mask == 1)[0]
# sample_index = np.random.choice(selected_index)

sample_index = np.random.randint(0, len(valid_predictions), 1)[0]
fig = plot_gradcam(layer_gradcam, valid_dataset[sample_index][0], valid_targets[sample_index], valid_predictions[sample_index], 
                   labelcode, title=resultname + '-gradcam')
fig.savefig(joinpath(figdir, f"{resultname}-gradcam.png"), bbox_inches='tight')


# In[ ]:




