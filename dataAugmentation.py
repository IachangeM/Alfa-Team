import os
import time
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

# 可以是相对路径 也可以是绝对路径
# 将input_dir下的图片. 包括所有子文件夹以内的图片进行处理
# 如果不指定output_dir 将会在input_dir的同一级目录生成_dealed的文件夹
input = 'D:\Code\ACM\pat'
output = ''
saveFmt = '.png'
generalPicFmt =['.png', '.tif', '.tiff', '.bmp', '.pcx', '.jpeg', '.jpg' , '.gif']
size = (224, 224)


def getPath(input, output):
    """
    该函数主要对路径进行处理,参数:
        input: 原始数据的路径
        output: 处理后数据的存储路径
    该函数会遍历input路径下所有子文件夹下的数据进行处理,并根据output生成相应的目录存储已处理的数据.

    该函数主要功能:
        1.根据input&output生成相应的目录 用于存储处理后的图片数据
        2.返回所有input目录下包含图片的路径以及图片名称 以有序字典形式,
            key表示路径 value表示该路径下所有文件(图片)名字, sourcePathandFilename{}
            以及生成的路径,以列表的形式,和有序字典的key保持一致, savePath[]
    """
    sourcePathandFilename = OrderedDict()
    savePath = []

    # 如果output为'' 则使用默认的保存路径, 在input_dir的同一级目录生成__dealed的文件夹
    tail = '(' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + ')'
    if output=='':
        output = input + '_dealed'
        if os.path.exists(output):
            output = output + tail
    elif os.path.exists(output):
            output = output + tail
    os.makedirs(output)
    savePath.append(output)

    for (crtDir, subfolder, files) in os.walk(input):
        # crtDir是当前正在遍历的路径(str type)...subfolder是当前路径下的子文件夹路径(list type)
        # files是当前路径下所有文件的名称(list type)

        path = crtDir.replace(input, output)
        if not os.path.exists(path):
            os.makedirs(path)
            savePath.append(path)

        sourcePathandFilename[crtDir] = []
        for file in files:
            filename, filetype = os.path.splitext(file)
            if filetype in generalPicFmt:
                sourcePathandFilename[crtDir].append(file)
            else:
                print("Find a file that is not picture: ", os.path.join(crtDir,file))

    return sourcePathandFilename, savePath


def singlePicProcess(im, do):
    # im 是由PIL.Image打开的一张图像
    # do 是对图像进行的所有操作,除了resize之外, do是一个
    pass

def main():
    sourcePathandFilename, savePath = getPath(input, output)
    i = 0
    for path, pics in sourcePathandFilename.items():
        for pic in pics:
            name, type = os.path.splitext(pic)
            imgPath = os.path.join(path, pic)

            im = Image.open(imgPath)
            for deg in [30, 60, 90, 270]:
                #   _rotate_degree30 60 90 270
                #   resize还是原来的名字
                im = im.rotate(deg)
                tname = name + '_rotate_degree'+str(deg)
                imgSavePath = os.path.join(savePath[i], tname + saveFmt)
                im.save(imgSavePath)
        i+=1

        #
        # print("当前源路径: " + key + "\n创建的所对应文件夹:" + savePath[i])
        # print("当前原路径下所有的图片文件:  ")
        # print(val)
        # print("\n\n")
        # i+=1


if __name__ == '__main__':
    main()
