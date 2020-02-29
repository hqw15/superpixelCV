import glob
import json
import os
import sys
from tkinter import _flatten

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import xmltodict
from scipy.misc import imread
from scipy.ndimage import distance_transform_edt as distance
from skimage import measure, morphology
from skimage.segmentation import (chan_vese, felzenszwalb, quickshift,
                                  random_walker, slic)
from sklearn.metrics import confusion_matrix

o_path = os.getcwd().split('/Exp/')[0] + '/Src' 
sys.path.append(o_path) 

from superpixelCV import cv_cut





def maxflow(img_data, segments, tmp_cut, show_type, save_name):
    '''show_type:
        0: 全连接
        1: 分割结果
        2: 前景
        3: 背景
    '''

    nodes_num = segments.max() + 1
    adjmat = tmp_cut.adjmat
    from matplotlib.pyplot import MultipleLocator
    plt.ion()
    plt.axis('off')
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(111)

    xl, yl = img_data.shape
    if show_type == 0 or show_type == 1 :
        ax1.plot([0,yl], [0,0], color = '#D9D9D9', linewidth=2)
        ax1.plot([0,xl/1.414], [0,xl/1.414], color = '#D9D9D9', linewidth=2)
        ax1.plot([xl/1.414,yl + xl/1.414], [xl/1.414,xl/1.414], color = '#D9D9D9', linewidth=2)
        ax1.plot([yl,yl+xl/1.414],[0, xl/1.414], color = '#D9D9D9', linewidth=2)


    allx, ally = [],[]
    for index in range(nodes_num):
        loc = (segments == index).nonzero()
        x = xl - loc[0].mean()
        allx.append(x)
        x /= 1.414
        y = loc[1].mean()
        ally.append(y)
        y += x

        for j in range(0,index):
            if adjmat[index,j] > 0:
                if show_type != 0:
                    if tmp_cut.lable[j] != tmp_cut.lable[index]:
                        continue
                    if show_type == 2 and tmp_cut.lable[j] != 0:
                        continue
                    if show_type == 3 and tmp_cut.lable[j] != 1:
                        continue

                loc = (segments == j).nonzero()
                ax = xl - loc[0].mean()
                ax /= 1.414
                ay = loc[1].mean()
                ay += ax
                ax1.plot([y,ay],[x, ax], color = 'black', linewidth=2)

        if show_type == 0:
            ax1.scatter(y, x, s=80, color = 'r')
        else:
            if show_type != 3 and tmp_cut.lable[index] == 0:
                ax1.scatter(y, x, s=80, color = 'b')
            if show_type != 2 and tmp_cut.lable[index] == 1:
                ax1.scatter(y, x, s=80, color = 'black')

    dx = 50
    dy = 10
    allx = np.mean(allx) / 1.414
    ally = np.mean(ally) + allx
    if show_type != 3:
        ax1.scatter(ally+dy, allx + dx, s=100, color = 'b')
    if show_type != 2:
        ax1.scatter(ally-dy, allx - dx, s=100, color = 'black')

    
    for index in range(nodes_num):
        loc = (segments == index).nonzero()
        x = xl - loc[0].mean()
        x /= 1.414
        y = loc[1].mean()
        y += x

        if show_type == 0:
            ax1.plot([y,ally+dy],[x, allx + dx], color = 'gray', linewidth=0.3)
            ax1.plot([y,ally-dy],[x, allx - dx], color = 'gray', linewidth=0.3)
        else:
            if tmp_cut.lable[index] == 0 and show_type != 3:
                ax1.plot([y,ally+dy],[x, allx + dx], color = 'gray', linewidth=0.3)
            if tmp_cut.lable[index] == 1 and show_type != 2:
                ax1.plot([y,ally-dy],[x, allx - dx], color = 'gray', linewidth=0.3)

    x_major_locator=MultipleLocator(10)
    y_major_locator=MultipleLocator(10)
    ax1=plt.gca()
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.yaxis.set_major_locator(y_major_locator)
    plt.axis('scaled')
    plt.axis('off')
    plt.savefig(os.path.join('result',save_name))
    plt.close()



def save_superpixel_center(img_data, segments, save_name):

    from matplotlib.pyplot import MultipleLocator

    plt.ion()
    plt.axis('off')
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(111)

    xl, yl = img_data.shape
    ax1.plot([0,yl], [0,0], color = '#D9D9D9', linewidth=2)
    ax1.plot([0,0], [0,xl], color = '#D9D9D9', linewidth=2)
    ax1.plot([yl,yl], [xl,0], color = '#D9D9D9', linewidth=2)
    ax1.plot([yl,0],[xl, xl], color = '#D9D9D9', linewidth=2)

    nodes_num = segments.max() + 1
    for index in range(nodes_num):
        loc = (segments == index).nonzero()
        x = xl - loc[0].mean()
        y = loc[1].mean()
        ax1.scatter(y, x, s=80, color = 'r')

    x_major_locator=MultipleLocator(10)
    y_major_locator=MultipleLocator(10)
    ax1=plt.gca()
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.yaxis.set_major_locator(y_major_locator)
    plt.axis('scaled')
    plt.axis('off')
    plt.savefig(os.path.join('result',save_name))
    plt.close()


def save_superpixel(img, seg, save_name):
    ''' 存储superpixel 分割结果'''
    plt.ion()
    plt.axis('off')
    fig2 = plt.figure(1,(15,7))
    ax1 = fig2.add_subplot(111)
    ax1.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for x in range(seg.max()+1):
        cut = (seg == x)
        contours = measure.find_contours(cut, 0.5)
        for n, contour in enumerate(contours):
            ax1.plot(contour[:, 1], contour[:, 0], color = 'r', linewidth=3)
    plt.savefig(os.path.join('result',save_name))
    plt.close()

def save_seg(img_data, cut, save_name):

    plt.ion()
    plt.axis('off')
    fig2 = plt.figure(1,(15,7))
    ax1 = fig2.add_subplot(111)
    ax1.imshow(img_data, interpolation='nearest', cmap=plt.cm.gray)


    contours = measure.find_contours(cut, 0.5)
    for n, contour in enumerate(contours):
        ax1.plot(contour[:, 1], contour[:, 0], color = 'r', linewidth=2)
    plt.savefig(os.path.join('result',save_name))
    plt.close()

if __name__ == '__main__':

    img_src = 'I00058.bmp'
    oldimg = np.array(imread(img_src, True), dtype='float64')

    xmin, xmax, ymin, ymax = 183, 286, 241, 319
    newimg = oldimg[ymin:ymax, xmin:xmax]

    img_data = newimg


    ## 归一化到0-255
    img_data = img_data/img_data.max()*255
    ## superpixel
    segments = slic(img_data, n_segments=100, compactness=100, max_iter=1, convert2lab=False)
    '''存储superpixel初始化'''
    save_superpixel(img_data, segments,'superpixel_init.png')

    segments = slic(img_data, n_segments=100, compactness=100, max_iter=100, convert2lab=False)
    '''存储superpixel分割结果'''
    save_superpixel(img_data, segments,'superpixel_final.png')
    save_superpixel_center(img_data, segments, 'superpixel_center.png')
    
    ## cv分割
    tmp_cut = cv_cut(img_data, segments, init_t = 0.05, mu=0.01, local_in = 1, local_out = 1.,
                    global_in = 1, global_out = 1., lambda_in = 30)
    cut, last, cnt = tmp_cut.main(10)


    maxflow(img_data, segments, tmp_cut, show_type = 0, save_name = 'full_join.png')
    maxflow(img_data, segments, tmp_cut, show_type = 1, save_name = 'part.png')
    maxflow(img_data, segments, tmp_cut, show_type = 2, save_name = 'fg.png')
    maxflow(img_data, segments, tmp_cut, show_type = 3, save_name = 'bg.png')
    
    '''存储分割结果'''
    save_seg(img_data, cut, save_name='seg_result.png')
    

    
    

    
    


    
    