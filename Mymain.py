"""
@author:achange
base on win10 + anaconda+python 3.6.x + pytorch 0.4.0
"""

# coding=utf-8
import argparse
import numpy
import os
import shutil
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pprint import pprint
import itertools


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def list2generator(listName):
    for g in listName:
        yield g


def get_images_name(folder):
    """Create a generator to list images name at evaluation time"""
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for f in onlyfiles:
        yield f


def pil_loader(path):
    """Load images from /eval/ subfolder and resized it as squared"""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            sqrWidth = numpy.ceil(numpy.sqrt(img.size[0] * img.size[1])).astype(int)
            return img.resize((sqrWidth, sqrWidth))


def train(train_loader, model, criterion, optimizer, epoch):  # criterion is loss function
    """Train the model on Training Set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = data
        if cuda:
            input_var, target_var = input.cuda(async=True), target.cuda(async=True)

        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # topk = (1,5) if labels >= 100 else (1,) # TO FIX
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o, target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0) * len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.item(), input.size(0)) # losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))


def validate(val_loader, model, criterion):
    """Validate the model on Validation Set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # Evaluate all the validation set
    for i, data in enumerate(val_loader):
        input, target = data
        if cuda:
            input_var, target_var = input.cuda(async=True), target.cuda(async=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        # print ("Output: ", output)
        # topk = (1,5) if labels >= 100 else (1,) # TODO: add more topk evaluation
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o, target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0) * len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses,
                top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))

    print(' * Prec@1 {top1}'
          .format(top1=numpy.asscalar(top1.avg.cpu().numpy())))
    return top1.avg


def test(test_loader, model, names, classes):
    """Test the model on the Evaluation Folder

    Args:
        - classes: is a list with the class name
        - names: is a generator to retrieve the filename that is classified
    """
    # switch to evaluate mode
    model.eval()
    # Evaluate all the validation set
    y_pred = []
    fw = open('./output/classRes.txt', 'a+', encoding='utf-8')
    for i, (input, _) in enumerate(test_loader):
        if cuda:
            input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)
        # Take last layer output
        if isinstance(output, tuple):
            output = output[len(output) - 1]
        lab = classes[numpy.asscalar(output.data.max(1, keepdim=True)[1].cpu().numpy())]
        # lab是str类型的！!！
        y_pred.append(lab)
        imgName = next(names)
        print(str(i) + ":  Images: " + imgName + ", Classified as: " + lab)
        fw.writelines("Images: " + imgName + ", Classified as: " + lab + '.\n')
    fw.close()
    return y_pred


def save_checkpoint(state, is_best, filename='vgg19.pkl'):  # 原本保存的文件名是:checkpoint.pth.tar
    torch.save(state, os.path.join(args.outf, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.outf, filename), os.path.join(args.outf, 'vgg19-model_best.pkl'))


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
    lr = args.lr * (0.1 ** (epoch // 30))
    print('current learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).to(device))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

##########     预定义的参数与变量    ##########

# 接收运行main.py时传递的参数  ----使用argparse模块
'''该模块正确用法：
parser.add_argument('--data', default='./input/', metavar='DIR',
                    help='path to dataset')
'''
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
args = parser.parse_args()
args.data = 'E:/Data/dianshi/datasets'
args.outf = './output'
# net in ['alexnet', 'densenet121', 'densenet161', 'densenet169',
#                'densenet201', 'inception_v3', 'resnet101', 'resnet152',
#                'resnet18', 'resnet34', 'resnet50', 'squeezenet1_0',
#                'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
#                'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
args.arch = 'vgg19'
args.workers = 0
args.epochs = 90
args.start_epoch = 0
args.batch_size = 6
args.lr = 0.001
args.momentum = 0.9
args.weight_decay = float(1e-4)
args.print_freq = 10
args.resume = ''
args.evaluate = False
args.train = True
args.test = False
args.fine_tuning = False
args.world_size = 1
args.distributed = False

# 是否使用CUDA?
cuda = torch.cuda.is_available() #->可用则用
print("=> using cuda: {cuda}".format(cuda=cuda))
print("=> distributed training: {dist}".format(dist=args.distributed))
best_prec1 = torch.FloatTensor([0])
train_sampler = None

############ DATA PREPROCESSING ############

# 定义数据集的路径：
traindir = os.path.join(args.data, 'train')
testdir = os.path.join(args.data, 'test')

### 加载数据集涉及的变量定义：

size = (224, 224)
# 内嵌的VGG19模型图片必须resize->  224*224,  并且  Normalize on RGB Value
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# Train -> Preprocessing -> Tensor
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

# pprint(train_dataset.class_to_idx)  --  pprint(train_dataset.imgs)
labels = len(train_dataset.classes)  # Get number of labels labels=classes = 3 三分类问题。

# Pin memory  固定存储器？？
if cuda:
    pin_memory = True
else:
    pin_memory = False  # use this default

# Data loading code
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)

# if args.test: # 如果需要进行测试, 加载测试集
# Testing -> Preprocessing -> Tensor
'''不需要test_loader
test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(size[1]),  # 256  原为transforms.Scale
        transforms.ToTensor(),
        normalize,
    ]), loader=pil_loader),
    batch_size=1, shuffle=True,
    num_workers=args.workers, pin_memory=pin_memory)
'''
############ BUILD MODEL ############

# 通过修改args.arch='{model_name[i]}'来改变所使用的模型
# all model on Pytorch (torchvison):
# model_names = ['alexnet', 'densenet121', 'densenet161', 'densenet169',
#                'densenet201', 'inception_v3', 'resnet101', 'resnet152',
#                'resnet18', 'resnet34', 'resnet50', 'squeezenet1_0',
#                'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
#                'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

print("=> creating model '{}'".format(args.arch))
model = models.__dict__[args.arch](num_classes=labels)
print(model)

criterion = nn.CrossEntropyLoss()

if cuda:
    criterion.to(device)
    print('criterion.cuda()')

# Set SGD + Momentum
parameters = model.parameters()
# optimizer = torch.optim.Adam(parameters, args.lr)
optimizer = torch.optim.SGD(parameters, args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
# optimizer = torch.optim.ASGD(parameters,lr=args.lr,weight_decay=args.weight_decay)

# Load model on GPU or CPU
if cuda:
    model.to(device)
else:
    model.cpu()

############ TRAIN/EVAL/TEST ############
# cudnn.benchmark = True 好像这一句是用于检测显卡硬件设备是否正常的

# Training
if args.train:
    print("=> training...")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        ## 保存模型的参数:
        # Evaluate on validation set
        prec1 = validate(train_loader, model, criterion)
        is_best = bool(prec1.cpu().numpy() > best_prec1.numpy())
        # Get greater Tensor
        best_prec1 = torch.FloatTensor(max(prec1.cpu().numpy(), best_prec1.cpu().numpy()))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print("训练的开始时间: ", start_time)
    print("训练的结束时间: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# Evaluate?
if args.evaluate:
    print("=> evaluating...")
    avg = validate(test_loader, model, criterion)
    print("finnal test accuracy:", avg)
# Testing?
if args.test:

    vv = torch.load('./output/vgg19.pkl')
    epoch, arch, state_dict, best_prec1, optimizer = vv['epoch'], vv['arch'], vv['state_dict'], vv['best_prec1'], vv[
        'optimizer']
    model.load_state_dict(state_dict)
    print('best_prec1: ', best_prec1)

    print("=> testing...")
    '''
    names = get_images_name('./input/test/2')
    test(test_loader, model, names, train_dataset.classes)
    '''
    # Name generator
    y_pred = []
    y_true = []
    names = []
    folders = train_dataset.classes
    for folder in folders:
        dir = './input/test/' + folder
        tmp_names = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        names += tmp_names
        y_true.extend([folder] * len(tmp_names))

    print(y_true)
    # 将list类型的names转换成一个generator：自己写了一个 list2generator() 函数
    y_pred.extend(test(test_loader, model, list2generator(names), train_dataset.classes))  # y_pred的元素时str类型
    print(y_pred)

    ## 画出混淆矩阵:
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    class_names = train_dataset.classes
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

