#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import torch.optim as optim
# import pytorch_lightning as pl
import copy
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.axes_grid1 import ImageGrid
# from sklearn.metrics import ConfusionMatrixDisplay,classification_report
import shutil
import json
# import torchmetrics
# %load_ext jupyternotify

parser = argparse.ArgumentParser()
parser.add_argument('--name',nargs = 1)
args = parser.parse_args()
filename = args.name[0]
# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
print(device)


# In[3]:


if os.path.exists("./imgs"):
  shutil.rmtree('./imgs')
os.makedirs("./imgs")


# In[4]:


batch_size = 8
classes = ['Aloo_Fry', 'Banana_Chips', 'Bhindi', 'Dhokla', 'Dosa', 'Idli', 'Jalebi', 'Pav_Bhaji', 'Rasgulla', 'Samosa']


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')
# get_ipython().system('cp ./drive/MyDrive/calories_data/calorie_values.csv .')
# get_ipython().system('cp ./drive/MyDrive/fyp_weights/VGG_finalstate.pt .')
# get_ipython().system('cp ./drive/MyDrive/fyp_weights/mrcnn_1.pth .')


# In[9]:


from vision import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# In[10]:


from engine import train_one_epoch, evaluate
import utils
import transforms as T_mrcnn
def get_transform_mrcnn(train):
    transforms_mrcnn = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms_mrcnn.append(T_mrcnn.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms_mrcnn.append(T_mrcnn.RandomHorizontalFlip(0.5))
    return T_mrcnn.Compose(transforms_mrcnn)


# In[11]:


def get_transform(train = True):
  return transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Resize([224,224], antialias = True),
                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


# In[12]:


# device = xm.xla_device()
# print(device)
# our dataset has two classes only - background and person
num_classes = 103

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# In[13]:


model.load_state_dict(torch.load('./mrcnn_1.pth'))


# In[14]:


img = Image.open('./frontend_input/'+filename)
# img


# In[15]:


img_transformed = get_transform_mrcnn(train = False)(img,None)
print(len(img_transformed),type(img_transformed))
if img_transformed[0].shape[0] != 3:
  if os.path.exists("./outputs.txt"):
    os.remove("./outputs.txt")
  f = open(f"./outputs.txt", "w")
  f.write('{"items":[],"total":0}')
  f.close()
  exit()

# In[16]:


model.eval()
with torch.no_grad():
    print(img_transformed[0].shape)
    prediction = model([img_transformed[0].to(device)])


# In[17]:


def plot_img(images,normal=False,labels=None):
    npimg=images.numpy()
    if normal:
#     img*mean(0.5) + SD(0.5) => unnormalizing the image
        npimg=images.numpy()/2+0.5
    # print(images.shape)
    # npimg = np.transpose(npimg,(0,2,3,1))
    maxValue = np.amax(npimg)
    minValue = np.amin(npimg)
    # print(maxValue,minValue)
    # print(npimg.shape)
    fig = plt.figure()
    columns = npimg.shape[0]
    rows = 1
    for i in range(1, columns*rows +1):
      fig.add_subplot(rows, columns, i)
      # print(len(npimg[i-1].shape))
      if len(npimg[i-1].shape) == 3:
        cur = copy.deepcopy(npimg[i-1])
        if cur.shape[0]<=3:
          cur = np.transpose(cur,(1,2,0))
        plt.imshow(cur)
      else:
        plt.imshow(npimg[i-1])
      # plt.imshow(cur)
      plt.axis("off")
      # print(labels)
      if labels is not None:
        plt.title(labels[i-1])
    # plt.imshow(npimg)
    plt.show()


# In[20]:


# print(prediction[0]['masks'].shape)
masks = prediction[0]['masks']
# print(np.unique(masks.cpu().numpy()))
masks = masks>0.5
mask = copy.deepcopy(masks[0])
for i in range(1,masks.shape[0]):
  mask+=masks[i]
# print(mask.shape)
# plot_img(mask.cpu(),normal = True)


# In[21]:


# print(type(img_transformed[0]))
# print(img_transformed[0].shape,mask.shape)
final = img_transformed[0]*mask.cpu()
# print(type(final),final.shape)
final = final.cpu().numpy()
# final =  final/2+0.5
final = np.transpose(final,(1,2,0))
# print(final.shape)
im = Image.fromarray(np.uint8(final*255)).convert('RGB')
# im
# im.save("imgs/sample_all.png")
# imgplot = plt.imshow(final)
# Image.fromarray(final)
# Image.fromarray(((final.cpu().numpy())*255).astype(np.uint8))


# In[22]:


from PIL import ImageDraw
height = mask.shape[1]
width = mask.shape[2]
cropped_version = copy.deepcopy(mask[0].cpu().numpy())
cropped_version = Image.fromarray(cropped_version)
ct = 0
cropped_version = cropped_version.convert('L')
# print(cropped_version.size,cropped_version.mode)
for i in range(height):
  for j in range(width):
    if cropped_version.getpixel((j,i)) == 255:
      ct+=1
      ImageDraw.floodfill(cropped_version,(j,i),ct)
# print(ct)
# cropped_version


# In[23]:


st={''}
for i in range(height):
  for j in range(width):
    if cropped_version.getpixel((j,i))!=0:
      st.add(cropped_version.getpixel((j,i)))
st


# In[25]:


cropped_version = np.array(cropped_version)
obj_ids = np.unique(cropped_version)
print(obj_ids)
obj_ids = obj_ids[1:]
cropped_masks = (cropped_version == obj_ids[:, None, None])


# In[28]:


# print(cropped_masks.shape)
# plot_img(torch.tensor(cropped_masks))


# In[30]:


masked_imgs = torch.empty(0,3,cropped_masks.shape[1],cropped_masks.shape[2])
# print(masked_imgs.shape)
# print(cropped_masks.shape)
for mask in cropped_masks:
  mask = mask>0
  mask = torch.tensor(mask)
  mask  = torch.unsqueeze(mask, axis = 0)
  out = torch.unsqueeze(mask*img_transformed[0], axis = 0)
  # png = Image.fromarray()
  masked_imgs = torch.cat((masked_imgs,out))
# print(masked_imgs.shape)
# plot_img(masked_imgs)


# In[31]:


for i,masked_img in enumerate(masked_imgs):
  npim = masked_img.cpu().numpy()
  npim = np.transpose(npim,(1,2,0))
  # print(np.unique(npim))
  im = Image.fromarray((npim * 255).astype(np.uint8))
  im.save(f"imgs/im_single_{i+1}.png")


# In[32]:


def evaluation(dataloader,model):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        total,correct=0,0
        for data in dataloader:
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            m = nn.Sigmoid()
            outputs=m(outputs)
#             preds = output.data.max(dim=1,keepdim=True)[1]
            _,pred = torch.max(outputs.data, 1)
#             pred=outputs>=0.5
            pred=pred.flatten()
            for label in labels:
              y_true.append(label.item())
            for prediction in pred:
              y_pred.append(prediction.item())
            total+=labels.size(0)
            correct+=(pred==labels).sum().item()
            del inputs,labels,outputs,pred
            torch.cuda.empty_cache()
    print(correct,total)
    # plt.figure(figsize = (20,20))
    plt.tight_layout()
    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred,display_labels = classes,xticks_rotation = 'vertical')
    # print(classification_report(y_true,y_pred))
    plt.show()
    model.train()

    return 100*correct/total


# In[33]:


class VGG(nn.Module):
    def _init_(self,learning_rate=0.001):
        super()._init_()
        self.learning_rate = learning_rate
        # init a pretrained vggnet
        backbone = models.vgg16_bn(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        fc=[]
        fc.extend([
            # nn.Linear(in_features=25088,out_features=512),
          #  nn.ReLU(),
          #  nn.Dropout(),
           nn.Linear(in_features=25088,out_features=256),
           nn.ReLU(),
          #  nn.Dropout(),
           nn.Linear(in_features=256,out_features=10)
          ])
        self.classifier=nn.Sequential(*fc)
        for param in self.feature_extractor.parameters():
            param.requires_grad=False
    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x=self.classifier(representations)
        return x


# In[34]:


vgg=VGG().to(device)


# In[35]:


# vgg.load_state_dict(torch.load('vgg_cpu.pt'))
vgg = torch.load('vgg.pt')


# In[36]:


vgg.eval()
final = []
with torch.no_grad():
  for img in os.listdir('./imgs'):
    img_path = img
    img = Image.open('./imgs/'+img)
    img = get_transform(False)(img)
    img = torch.unsqueeze(img, axis = 0)
    outputs = vgg(img.to(device))
    print("Before Softmax:")
    print(outputs)
    outputs = nn.Softmax()(outputs)
    val,idx = torch.max(outputs.data,1)
    val = torch.squeeze(val)
    idx = torch.squeeze(idx)
    final.append((val,idx,img_path))


# In[37]:


print(final)


# In[39]:


df = pd.read_csv('./calorie_values.csv')


# In[40]:


df.head()


# In[41]:


sum = 0
for p,idx,path in final:
  # print(df.loc[df['Food Item']== classes[idx]])
  if p.item() <= 0.5:
    continue
  # print(p,classes[idx])
  dff = df.loc[df['Food Item'] == classes[idx]]
  dff_val = dff['Calories(KCal)'].iloc[0]
  print(f"{classes[idx]}: {dff_val} calories")
  sum += dff_val
  


# In[42]:


print(sum)


# In[43]:


sum = 0
result = {"items":[]}
for p,idx,path in final:
  # print(df.loc[df['Food Item']== classes[idx]])
  if p.item() <= 0.25:
    continue
  # print(p,classes[idx])
  dff = df.loc[df['Food Item'] == classes[idx]]
  dff_val = dff['Calories(KCal)'].iloc[0]
  print(f"{classes[idx]}: {dff_val} calories")
  result["items"].append((classes[idx],int(dff_val),float(p.item()),path))
  sum += dff_val
result["total"] = int(sum)


# In[44]:


result


# In[45]:


result_str = json.dumps(result)


# In[46]:


if os.path.exists("./outputs.txt"):
  os.remove("./outputs.txt")
f = open(f"./outputs.txt", "w")
f.write(result_str)
f.close()