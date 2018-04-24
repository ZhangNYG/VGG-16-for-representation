#<span style="font-size:14px;color:#000000;">#! -*- coding: utf-8 -*-

## by Colie (lijixiang)

import tensorflow as tf
from scipy.misc import imsave
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import os
from skimage import io


############################ Macros Variable Define ###########################

# 训练轮数
train_epochs = 1  ## int(1e5+1)

# 图像的像素大小
INPUT_HEIGHT =180
INPUT_WIDTH = 320

# batch样本个数
batch_size = 0
# 统计训练集所含样本数
images=0

# 噪声大小
noise_factor = 0.5  ## (0~1)随机数



############################ Distance Definition ###########################

###### 1. Euclidean Distance ######

def euclidean(p,q):
# 如果两数据集数目不同，计算两者之间都对应有的数
 same = 0
 for i in p:
    if i in q:
        same +=1

# 计算欧几里德距离,并将其标准化
 e = sum([(p[i] - q[i])**2 for i in range(same)])
 return 1/(1+e**.5)

####### 2. Pearson Distance #######

def pearson(p,q):
# 只计算两者共同有的
    same = 0
    for i in p:
        if i in q:
            same +=1

    n = same
    # 分别求p，q的和
    sumx = sum([p[i] for i in range(n)])
    sumy = sum([q[i] for i in range(n)])
    # 分别求出p，q的平方和
    sumxsq = sum([p[i]**2 for i in range(n)])
    sumysq = sum([q[i]**2 for i in range(n)])
    # 求出p，q的乘积和
    sumxy = sum([p[i]*q[i] for i in range(n)])
    #print sumxy
    # 求出pearson相关系数
    up = sumxy - sumx*sumy/n
    #down = ((sumxsq - pow(sumxsq,2)/n)*(sumysq - pow(sumysq,2)/n))**.5
    down = ((sumxsq - pow(sumx,2)/n)*(sumysq - pow(sumy,2)/n))**.5
    # 若down为零则不能计算，return 0
    if down == 0 :return 0
    r = up/down
    return r

####### 2. Consin Distance #######

def consin(p,q):
# 只计算两者共同有的
    same = 0
    for i in p:
            same +=1

    n = same
    # 求出p，q的乘积和
    sumxy = sum([p[i]*q[i] for i in range(n)])
    # 分别求出p，q的平方和
    sumxsq = sum([p[i]**2 for i in range(n)])
    sumysq = sum([q[i]**2 for i in range(n)])
    # 求出consin相关系数
    up = sumxy
    down = (sumxsq*sumysq)**.5
    # 若down为零则不能计算，return 0
    if down == 0 :return 0
    r = up/down
    return r


################# 对所有的images分batch块，获取next_batch ###################

def next_batch(batch_size, each, images):
    batch_x = np.zeros([batch_size, INPUT_HEIGHT * INPUT_WIDTH * 3])

    def get_image(i, batch):
        image_num = batch * batch_size + i
        image_path = image_paths[image_num]
        captcha_image = Image.open(image_path)  # 按照路径打开图片
        #print(image_path)
        captcha_image = np.array(captcha_image)
        return captcha_image
 
    for i in range(batch_size):
        imagei = get_image(i, batch)
        batch_x[i, :] = imagei.flatten()# / 255  # (image.flatten()-128)/128  mean为0
        #print("batch_x.shape:",i,batch_x[i].shape)
    #print(batch_x.shape)
    return batch_x



########### 获取训练集图像URL（image_paths=[]）和样本个数（images）###########

path = '/root/lichen/20180302/cctv/video_1/keypics-test/'
fns = [os.path.join(fn) for root, dirs, files in os.walk(path) for fn in files]
image_paths=[]


for f in fns:
    fsize = os.path.getsize(path + str(f) )
    fsize = fsize / float(1024 * 1024)
    if fsize!=0:
        image_paths.append(path+str(f))
        images=images+1
print("images:",str(images))
#for i in image_paths:
  # print(i)



################## Convolutional Neural Network （CNN : 3 layer） ##################

###### 1. 输入数据准备 ######

## 原始输入是320×180*3
input_x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT * INPUT_WIDTH * 3], name='input_with_noise')
input_matrix=tf.reshape(input_x,shape=[-1,INPUT_HEIGHT,INPUT_WIDTH,3])
input_raw=tf.placeholder(tf.float32,shape=[None,INPUT_HEIGHT*INPUT_WIDTH * 3],name='input_without_noise')
#print(input_x.shape)
#print(input_raw.shape)

#### 2. CNN网络结构搭建 (3 layer) ####

## 1 conv layer
weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=0.1, name = 'weight_1'))
bias_1 = tf.Variable(tf.constant(0.0, shape=[64], name='bias_1'))
conv1 = tf.nn.conv2d(input=input_matrix, filter=weight_1, strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')
acti1 = tf.nn.relu(conv1, name='acti_1')
pool1 = tf.nn.max_pool(value=acti1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
#print(pool1.shape)

## 2 conv layer
weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1, name='weight_2'))
bias_2 = tf.Variable(tf.constant(0.0, shape=[64], name='bias_2'))
conv2 = tf.nn.conv2d(input=pool1, filter=weight_2, strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')
acti2 = tf.nn.relu(conv2, name='acti_2')
pool2 = tf.nn.max_pool(value=acti2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')
#print(pool2.shape)

## 3 conv layer
weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=0.1, name='weight_3'))
bias_3 = tf.Variable(tf.constant(0.0, shape=[128]))
conv3 = tf.nn.conv2d(input=pool2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, bias_3)
acti3 = tf.nn.relu(conv3, name='acti_3')
pool3 = tf.nn.max_pool(value=acti3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_3')
#print("pool3:",pool3.shape)



################## Deconvolution Neural Network （CNN : 3 layer） ##################

#### 1. Deconvolution-NN网络结构搭建 (3 layer) ####

## 1 deconv layer
deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=0.1), name='deconv_weight_1')
deconv1 = tf.nn.conv2d_transpose(value=pool3, filter=deconv_weight_1, output_shape=[1, 45, 80, 64], strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')
print("deconv1:",deconv1.shape)

## 2 deconv layer
deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name='deconv_weight_2')
deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=[1, 90, 160, 64], strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')
#print ("deconv2",deconv2.shape)

## 3 deconv layer
deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1, name='deconv_weight_3'))
deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=[1, 180, 320, 64], strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')
#print ("deconv3",deconv3.shape)

#### 2. CNN网络结构搭建 (1 layer) ####

## conv layer

weight_final = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 3], stddev=0.1, name = 'weight_final'))
bias_final = tf.Variable(tf.constant(0.0, shape=[3], name='bias_final'))
conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')
conv_final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')
print ("conv_final",conv_final.shape)

#### 3. CNN OUTPUT #### 
output = tf.reshape(conv_final, shape=[-1, INPUT_HEIGHT * INPUT_WIDTH * 3])
#print(output.shape)
#print(input_raw.shape)

saver = tf.train.Saver()
image_marks=[]
image_marks1=[]
ins=[]
#将batch_size设置为文件夹中图片的数量
#image_marks为帧特征
batch_size=images
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")

    ####################################################
    # Anthor:XianjieZhang
    # 对每张图片进行计算输出
    num_image_zhang = 0
    # 在这 batch_size_zhang = 1
    batch_x_zhang = np.zeros([1, INPUT_HEIGHT * INPUT_WIDTH * 3])
    for i in range(len(os.listdir('./all_images/'))):
        num_image_zhang = num_image_zhang + 1
        captcha_image = Image.open("./all_images/" + str(num_image_zhang) + ".jpg")
        print("./all_images/" + str(num_image_zhang) + ".jpg")
        captcha_image = np.array(captcha_image)
        batch_x_zhang[0, :] = captcha_image.flatten()
        noise_x_zhang = batch_x_zhang + noise_factor * np.random.randn(*batch_x_zhang.shape)
        weight_1_value = sess.run(weight_1, feed_dict={input_x: noise_x_zhang})
        weight_2_value = sess.run(weight_2, feed_dict={input_x: noise_x_zhang})
        weight_3_value = sess.run(weight_3, feed_dict={input_x: noise_x_zhang})
        conv1_value1 = sess.run(conv1, feed_dict={input_x: noise_x_zhang})
        conv_final_value = sess.run(conv_final, feed_dict={input_x: noise_x_zhang})
        print()


    total_batch = int(images /batch_size)
    for batch in range(total_batch):
        batch_x = next_batch(batch_size, batch, image_paths)
        noise_x = batch_x + noise_factor * np.random.randn(*batch_x.shape)
        #noise_x = np.clip(noise_x, 0., 1.)
        pool_ = sess.run(pool3, feed_dict={input_x: noise_x})
        input=sess.run(input_x,feed_dict={input_x: batch_x})
        list = []
        list1=[]
        #print(pool_.shape)
        for i in range(batch_size):
            zw = pool_[i]
            zw = zw.reshape(1, -1)
            #zw=zw*255
            list.append(zw)
            ins=input[i]
            ins=ins.reshape(1,-1)
            #ins=ins*255
            list1.append(ins)
        for i in range(batch_size):
            #print("这是第",batch*batch_size+i,"个图片")           
            image_marks.append(list[i][0])
            image_marks1.append(list1[i][0])
            #print("pred:",list[i][0])
            #print("test:",list1[i][0])
image_index=[]
for i in image_paths:
    image_index.append(i)

for i in range(len(image_index)-1):
    for j in range(len(image_index)-1):
       if image_index[j]>image_index[j+1]:
          t=image_index[j]
          image_index[j]=image_index[j+1]
          image_index[j+1]=t 
image_marks_new=[]
image_marks_new1=[]
for i in range(len(image_paths)):
      for j in range(len(image_index)):
          if image_paths[i]==image_index[j]:
             #print(j)
             image_marks_new.append(image_marks[j])
             image_marks_new1.append(image_marks1[j])
    
distances=[]
for index in range(len(image_marks)-1):
            #x=x+1
            distance1=consin(image_marks_new[index],image_marks_new[index+1])
            distance2=consin(image_marks_new1[index],image_marks_new1[index+1])
            distances.append(distance1)
            print("pred:number",index+1," image and num",index+2,"image's pearson:",distance1)
            print("test:number",index+1," image and num",index+2,"image's pearson:",distance2)
            #print(image_marks[index])

# 阈值
#threshold=0
            



