#<span style="font-size:14px;color:#000000;">#! -*- coding: utf-8 -*-

## by Colie (lijixiang)

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import os
from skimage import io

# 分布式实现
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', 'ps', '"ps"or"worker"这个是参数服务器的名字还是计算服务器的名字')
tf.app.flags.DEFINE_string('ps_hosts', '127.0.0.1:22222', '参数服务器列表，192.168.1.160:2222,192.168.1.161:1111')
tf.app.flags.DEFINE_string('worker_hosts', '127.0.0.1:66662', '计算服务器，192.168.1.161:5555,192.168.162:6666')
tf.app.flags.DEFINE_integer('task_id', 0, '当前程序的任务ID，参数服务器和计算服务器都是从0开始')

ps_hosts = FLAGS.ps_hosts.split(',')
worker_hosts = FLAGS.worker_hosts.split(',')
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

if FLAGS.job_name == 'ps':
    # 参数服务器就运行到这
    server.join()




#训练轮数
train_epochs = 35  ## int(1e5+1)

#图像的像素大小为32*32
INPUT_HEIGHT =180
INPUT_WIDTH = 320

#batch样本个数
batch_size = 16
images=0

#噪声大小
noise_factor = 0.5  ## (0~1)

path = '/root/lichen/CNN/outimage/'
fns = [os.path.join(fn) for root, dirs, files in os.walk(path) for fn in files]
image_paths=[]
image1=[1,2,3,4]

for f in fns:
    fsize = os.path.getsize(path + str(f) )
    fsize = fsize / float(1024 * 1024)
    if fsize!=0:
        image_paths.append(path+str(f))
        images=images+1
print("images:",str(images))

def next_batch(batch_size, each, images):
    batch_x = np.zeros([batch_size, INPUT_HEIGHT * INPUT_WIDTH * 3])

    def get_image(i, batch):
        image_num = batch * batch_size + i
        image_path = image_paths[image_num]
        captcha_image = Image.open(image_path)  # 按照路径打开图片
        captcha_image = np.array(captcha_image)
        return captcha_image

    for i in range(batch_size):
        imagei = get_image(i, batch)
        batch_x[i, :] = imagei.flatten() #/ 255  # (image.flatten()-128)/128  mean为0
        #print("batch_x.shape:",i,batch_x[i].shape)
    #print(batch_x.shape)
    return batch_x


with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id,
        cluster=cluster)):
## 原始输入是320×180*3
    input_x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT * INPUT_WIDTH * 3], name='input_with_noise')
    input_matrix=tf.reshape(input_x,shape=[-1,INPUT_HEIGHT,INPUT_WIDTH,3])
    input_raw=tf.placeholder(tf.float32,shape=[None,INPUT_HEIGHT*INPUT_WIDTH * 3],name='input_without_noise')
    #print(input_x.shape)
    #print(input_raw.shape)

    ## 1 conv layer
    ## 输入28*28*3要变为32*32*3
    ## 经过卷积、激活、池化，输出14*14*64
    weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=0.1, name = 'weight_1'))
    bias_1 = tf.Variable(tf.constant(0.0, shape=[64], name='bias_1'))
    conv1 = tf.nn.conv2d(input=input_matrix, filter=weight_1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')
    acti1 = tf.nn.relu(conv1, name='acti_1')
    pool1 = tf.nn.max_pool(value=acti1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
    #print(pool1.shape)

    ## 2 conv layer
    ## 输入14*14*64
    ## 经过卷积、激活、池化，输出7×7×64
    weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1, name='weight_2'))
    bias_2 = tf.Variable(tf.constant(0.0, shape=[64], name='bias_2'))
    conv2 = tf.nn.conv2d(input=pool1, filter=weight_2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')
    acti2 = tf.nn.relu(conv2, name='acti_2')
    pool2 = tf.nn.max_pool(value=acti2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')
    #print(pool2.shape)
    ## 3 conv layer
    ## 输入7*7*64
    ## 经过卷积、激活、池化，输出4×4×32
    ## 原始输入是28*28*3=2352，转化为4*4*32=512，大量噪声会在网络中过滤掉
    weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=0.1, name='weight_3'))
    bias_3 = tf.Variable(tf.constant(0.0, shape=[128]))
    conv3 = tf.nn.conv2d(input=pool2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, bias_3)
    acti3 = tf.nn.relu(conv3, name='acti_3')
    pool3 = tf.nn.max_pool(value=acti3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_3')
    #print("pool3:",pool3.shape)
    ## 1 deconv layer
    ## 输入4*4*32
    ## 经过反卷积，输出8*8*32
    deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=0.1), name='deconv_weight_1')
    deconv1 = tf.nn.conv2d_transpose(value=pool3, filter=deconv_weight_1, output_shape=[batch_size, 45, 80, 64], strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')
    print("deconv1:",deconv1.shape)
    ## 2 deconv layer
    ## 输入8*8*32
    ## 经过反卷积，输出16*16*64
    deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name='deconv_weight_2')
    deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=[batch_size, 90, 160, 64], strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')
    #print ("deconv2",deconv2.shape)
    ## 3 deconv layer
    ## 输入16*16*64
    ## 经过反卷积，输出28*28*64
    deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1, name='deconv_weight_3'))
    deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=[batch_size, 180, 320, 64], strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')
    #print ("deconv3",deconv3.shape)
    ## conv layer
    ## 输入32*32*64
    ## 经过卷积，输出为32*32*3
    weight_final = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 3], stddev=0.1, name = 'weight_final'))
    bias_final = tf.Variable(tf.constant(0.0, shape=[3], name='bias_final'))
    conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')
    conv_final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')
    print ("conv_final",conv_final.shape)
    ## output
    ## 输入32*32*3
    ## reshape为32*32*3
    output = tf.reshape(conv_final, shape=[-1, INPUT_HEIGHT * INPUT_WIDTH * 3])
    #print(output.shape)
    #print(input_raw.shape)

    ## loss and optimizer
    loss = tf.reduce_mean(tf.pow(tf.subtract(output, input_raw), 2.0))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    print(images)
    sess_config = tf.ConfigProto(
                allow_soft_placement=True,  # 软配置 ， 可以自动分配硬件
                log_device_placement=True)  # 显示那个设备执行
    saver = tf.train.Saver()
    with tf.Session(server.target, config=sess_config) as sess:

        print('batch size: %d' % batch_size)
        total_batch=int(images/batch_size)
        print('total batchs: %d' % total_batch)
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(train_epochs):
            #print("epoch:",epoch)
            avg_cost = 0
            for batch in range(total_batch):
                #print("batch:",batch)
                batch_x = next_batch(batch_size,batch, image_paths)
                noise_x =batch_x + noise_factor * np.random.randn(*batch_x.shape)
                #noise_x=np.clip(noise_x,0.,1.)
                _,train_loss = sess.run([optimizer, loss], feed_dict={input_x: noise_x, input_raw: batch_x})
                print('epoch: %04d\tbatch: %04d\ttrain loss: %.9f' % (epoch + 1, batch + 1, train_loss))
        saver.save(sess, "Saved_model/model.ckpt")
        #print(sess.run(weight_3))
        #print(sess.run(input_raw, feed_dict={input_raw: batch_x}))
