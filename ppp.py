import os
from PIL import Image
from torchvision import transforms


picpath = "20051020_43832_0100_PP.tif"
im = Image.open(picpath)
transform1 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

im = transforms.ToPILImage()(transform1(im))
im.show()






