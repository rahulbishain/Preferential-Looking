import math, shutil, os, time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#from ITrackerData import ITrackerData
#from ITrackerModel import ITrackerModel
from get_external_module import get_module as getmod

'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

NOTE: This file has been slightly modified to be used with Preferential looking code - Rahul Bishain

'''

# Change there flags to control what happens.
doLoad = False # Load checkpoint at the beginning
doTest = False # Only run test, no training

workers = 8
epochs = 100
batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
acc = 0
best_acc = 1e20
acc_arr = []
lr = base_lr

count_test = 0
count = 0



def main():
    global args, best_acc, weight_decay, momentum

    external_module_name = "ITrackerModel"
    external_module_path = "../csail/ITrackerModel.py"
    model_module = getmod(external_module_name, external_module_path)

    model = model_module.ITrackerModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with error %.5f...' % (saved['epoch'], saved['best_acc']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_acc = saved['best_acc']
        else:
            print('Warning: Could not read checkpoint!')

    
    external_module_name = "ITrackerData"
    external_module_path = "../csail/ITrackerData.py"
    data_module = getmod(external_module_name, external_module_path)

    dataTrain = data_module.ITrackerData(split='train', imSize = imSize)
    dataVal = data_module.ITrackerData(split='test', imSize = imSize)
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # Quick test
    if doTest:
        return validate(val_loader, model, criterion, epoch)

    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, epoch)
        acc_arr.append(acc)

        # remember best prec@1 and save checkpoint
        is_best = acc < best_acc
        best_acc = min(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'acc_arr': acc_arr,
        }, is_best)

def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gazeLabel, recordNum, frameIndex) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda(async=True)
        imEyeL = imEyeL.cuda(async=True)
        imEyeR = imEyeR.cuda(async=True)
        faceGrid = faceGrid.cuda(async=True)
        gazeLabel = gazeLabel.cuda(async=True)
        
        imFace = torch.autograd.Variable(imFace)
        imEyeL = torch.autograd.Variable(imEyeL)
        imEyeR = torch.autograd.Variable(imEyeR)
        faceGrid = torch.autograd.Variable(faceGrid)
        gazeLabel = torch.autograd.Variable(gazeLabel)

        # zero gradients
        optimizer.zero_grad()

        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        gazeLabel = gazeLabel.squeeze()
        loss = criterion(output, gazeLabel)
        
        # losses.update(loss.cpu().data[0], imFace.cpu().size(0))
        losses.update(loss.item(), imFace.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                #   '[{3}/{4}]\t\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), 
                #    ','.join(map(str, recordNum.numpy())), ','.join(map(str, frameIndex.numpy())),
                   batch_time=batch_time, data_time=data_time, loss=losses))

def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    #------------------------------------------------------------------>>>>>>>>>>>>>>>>>>>>>
    # outfil = open('out1.csv','wb')
    final_output = torch.cuda.FloatTensor(0)
    final_loss = torch.cuda.FloatTensor(0)
    #------------------------------------------------------------------>>>>>>>>>>>>>>>>>>>>>

    Index = 0
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gazeLabel, recordNum, frameIndex) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda(async=True)
        imEyeL = imEyeL.cuda(async=True)
        imEyeR = imEyeR.cuda(async=True)
        faceGrid = faceGrid.cuda(async=True)
        gazeLabel = gazeLabel.cuda(async=True)
        
        imFace = torch.autograd.Variable(imFace, volatile = True)
        imEyeL = torch.autograd.Variable(imEyeL, volatile = True)
        imEyeR = torch.autograd.Variable(imEyeR, volatile = True)
        faceGrid = torch.autograd.Variable(faceGrid, volatile = True)
        gazeLabel = torch.autograd.Variable(gazeLabel, volatile = True)
        
        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)
        # final_output = torch.cat((final_output,torch.cuda.FloatTensor(output.data)))
        gazeLabel = gazeLabel.squeeze()
        loss = criterion(output, gazeLabel)

        # compute accuracy
        tmp, outputLabel = torch.max(output, 1)
        accuracy = sum(outputLabel == gazeLabel).float()/len(outputLabel)

        # final_loss = torch.cat((final_loss,torch.cuda.FloatTensor([loss.data])))

        # gazeLabel = gazeLabel.float()
        # lossLin = output - gazeLabel
        # lossLin = torch.mul(lossLin,lossLin)
        # lossLin = torch.sum(lossLin,1)
        # lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.item(), imFace.size(0))
        accuracies.update(accuracy.item(), imFace.size(0))

        # lossesLin.update(lossLin.data[0], imFace.size(0))
    
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # print('Epoch (val): [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
        #             epoch, i, len(val_loader), batch_time=batch_time,
        #            loss=losses,lossLin=lossesLin))

        print('Epoch (val)  : [{0}][{1}/{2}]\t'
                # '[{3}/{4}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format( epoch, i, len(val_loader), 
                # ','.join(map(str, recordNum.numpy())), ','.join(map(str, frameIndex.numpy())),
                batch_time=batch_time, data_time=data_time, accuracy=accuracies, loss=losses))

    
    # outfil.close()
    # return lossesLin.avg
    #print(final_output)
    return accuracies.avg


CHECKPOINTS_PATH = '../csail/'

def load_checkpoint(filename='checkpoint__2.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def save_checkpoint(state, is_best, filename='checkpoint__2.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print('DONE')
