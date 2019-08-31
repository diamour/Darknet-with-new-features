# -*- coding:utf-8 -*-

from voc_eval import voc_eval

import os

current_path = os.getcwd()
results_path = current_path+"/results"
sub_files = os.listdir(results_path)
mAP = []
for i in range(len(sub_files)):
    class_name = sub_files[i].split(".txt")[0]
    # print(class_name)
    rec, prec, ap = voc_eval('/home/gong/darknet2/results/{}.txt','/home/gong/darknet2/data/NOK/VOCdevkit/VOC2007/Annotations/{}.xml','/home/gong/darknet2/data/NOK/VOCdevkit/VOC2007/ImageSets/Main/val.txt',class_name, '/home/gong/darknet2/result')
    print("{} :\t {} ".format(class_name, ap))
    mAP.append(ap)

mAP = tuple(mAP)

print("***************************")
print("mAP :\t {}".format( float( sum(mAP)/len(mAP)) ))
