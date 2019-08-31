#Darknet with new features

added:

1.depthwise convolutional

you can use this layer like this: 

[depthwise_convolutional]

batch_normalize=1

size=3

stride=1

pad=1

activation=leaky

2.mix convolutional

[mix_convolutional]

batch_normalize=1

filters=128

sizeList= 3,5

stride=1

pad=1

activation=leaky

you can change the different convolutional kernel sizeLise as you will

[mix_convolutional]

batch_normalize=1

filters=128

sizeList= 3,5,7,3,7,11

stride=1

pad=1

activation=leaky

3.SE Block
[se]
batch_normalize=1
activation=logistic

if you want to disable the L2 in the connected layer, you can change  it by hand in the code. 

4.rectangle convolutional

[convolutional]

batch_normalize=1

filters=512

ksize_h=1

ksize_w=3

rectFlg=1

size=3

stride=1

pad=1

activation=leaky

so you can use it try different rectangle shape of convolution.

5. F1 score performance eval:

run it like:

.darknet f1 xxx.data xxx.cfg xxx.weights

6. mAP python is added:

 two steps:
 
(1).darknet valid xxx.data xxx.cfg xxx.weights

(2)python map.python

BUGS WARNING:

1.right now you can not run it on cudnn, you can trian it on GPU=1,CUDNN=0 or CPU

2.the mixConv seems doesn't works good under resizing net

IN THE FUTURES:

1. Some fine prune and channel prune technicals are added

2. The bugs will be solved

3. The tansplant of Cornernet is on the way.

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
