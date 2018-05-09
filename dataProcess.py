import os
from glob import glob
import shutil

pic_tag = dict()
with open('E:/Data/dianshi/datasets/train.txt', 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        fname, lable = line.split()
        pic_tag[fname] = lable



for fname, lable in pic_tag.items():
    path = os.path.join('E:/Data/dianshi/datasets/train', lable)
    print(path)
    file = 'E:/Data/dianshi/datasets/train/' + fname
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(os.path.join(path, fname)):
        shutil.move(file, path)
    else:
        print(os.path.join(path, fname), "已经存在...")


