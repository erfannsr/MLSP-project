import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms, transforms_224
from dataset.mydataset import MyDataset
import pandas as pd
import math

from torchsummary import summary

def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    device = args.device

    torch.backends.cudnn.benchmark=True
    test_dataset = MyDataset(txt_path=test_list, transform=transforms_224['test'], get_feature = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)
    print(f"Len Train dataset = {test_dataset_size}")
    corrects = 0
    acc = 0
    #model = torchvision.models.densenet121(num_classes=2)
    # model = model_selection(modelname='SupCON', num_out_classes=2, dropout=0.5)
    model = model_selection(modelname='resnet18', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    # if isinstance(model, torch.nn.DataParallel):
    #     print("is instance")
    #     model = model.module
    
    # print("model:")
    # print(dir(model.model))
    

    

    sum = 0
    # model.last_linear = nn.Sequential(
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(in_features=2048, out_features=2, bias=True),
    # )
    model = model.to(device)
    model.eval()

    # print(model)
    # exit()
    # summary(model,(3, 299, 299),batch_size=1,device="cuda")
    iteration = 0
    feature = []
    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.to(device)
            # print(image.size())
            labels = labels.to(device)
            outputs, _ = model(image) #,_
            # print(outputs.shape) 
            feature.append(outputs)
            #exit()
            _, preds = torch.max(outputs.data, 1)
            # print(f'size of data loader: {len(labels)}')
            # print(f'outputs.shape = {outputs.shape}')
            # print(outputs.data)
            # print(preds)
            # print(labels)
            # exit()
            # prob = nn.functional.softmax(outputs.data,dim=1)
    
            ############# calculate the difficult of sample ############## 

            corrects += torch.sum(preds == labels.data).to(torch.float32)
            iteration += 1
            if not (iteration % 100):
                print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
                # print(preds)
                # print(labels)
            # exit()
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))
    # print(feature)
    a = feature[0]
    for i in range(len(feature)):
        if i == 0:
            continue
        a = torch.cat([a,feature[i]],dim = 0)
    # torch.save(a, "20250424_train_data_800_sampes.pt")
    # b = torch.load("20250424_train_data_800_sampes.pt")
    torch.save(a, "29042025_getFeatureOutput.pt")
    b = torch.load("29042025_getFeatureOutput.pt")
    print(b.shape)



if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=28)
    parse.add_argument('--device', type=str, required=True, help="Check with nvidia-smi")
    #parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')
    parse.add_argument('--test_list', '-tl', type=str, default='FF++_DFDCP_DFD_CDF2.txt')
    #parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
    parse.add_argument('--model_path', '-mp', type=str, default='20230414_Task4_CDF_with_KD_MFD_SCL_Memory/best.pkl')
    
    main()

    print('Get fetures run successful.')