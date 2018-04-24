import os
import skimage
from PIL import Image


def get_image_paths(path):
    image_paths = []
    dirs = os.listdir(path)
    for dir in dirs:
        # print (dir)
        dir = path +'/'+dir
        dir1 = os.listdir(dir)
        for dir2 in dir1:
            # print(dir2)
            if dir2 == 'keypics':
                image_paths.append(dir + '/' + dir2)
    return image_paths


def file_name(image_path):
    images = []
    for root, dirs, files in os.walk(image_path):
        for image in files:
            images.append(image_path + '/' + image)
    return images
all_images=[]
lists=get_image_paths('/root/lichen/20180302/cctv')
for list in lists:
    list = file_name(list)
    for f in list:
       all_images.append(f)
x=1
for i in all_images:
   
    print(str(x))
    img=Image.open(i)
    img.save('/root/lichen/all_images/'+str(x)+'.jpg')
    x=x+1
