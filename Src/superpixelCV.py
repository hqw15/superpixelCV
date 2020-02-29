import datetime
import math
import queue
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage
import json
import maxflow
import networkx as nx
import cv2
import time
from scipy.misc import imread
# import imageio
from scipy.ndimage import distance_transform_edt as distance
from skimage import measure
from skimage.segmentation import felzenszwalb, quickshift, random_walker, slic





class cv_cut:

	def __init__(self, img, seg, mu=0.1, local_in = 1, local_out = 1.,
					global_in = 1, global_out = 1., lambda_in = 1, init_t = 0.5, el = 0.3):
		'''
			paras:
			----------------------
				img	: 待分割图像
				seg	: 超像素标签
				mu	: 周长参数， 值越大，表示像素间连接越紧密
				lambda_in : 相邻像素参数 
				local_in : 局部前景参数, 值越大，表示前景越均匀
				local_out : 局部背景参数，值越大，表示背景越均匀
				global_in : 全局前景参数
				global_out : 全局背景参数
				init_t : 初始化参数
		'''
		
		self.mu = mu
		self.lambda_in = lambda_in
		self.local_in = local_in
		self.local_out = local_out
		self.global_in = global_in
		self.global_out = global_out
		self.init_t = init_t
		self.el = el/3

		## 输入图像归一化
		self.img = img
		self.img -= self.img.min()
		if self.img.max() > 0:
			self.img /= self.img.max()

		self.seg = seg
		assert self.img.shape == self.seg.shape

		self.adjmat = self.__adj_mat()
		self.gray = self.ave_gray()
		self.lable = self.init_lable()

		# self.dis_mat =  self.__min_path()
		# self.d_mat =  self.__dist_center()*(self.adjmat>0)
		# self.d_mat -= self.d_mat.min()
		# self.d_mat /= self.d_mat.max()

		self.dis_mat = self.__adjacency_matrix()

		self.checkP = []



	def __adj_mat(self):
		'''	得到每个超像素与其它超像素的邻接像素个数
			以边界元素的四邻域计算
		'''

		LENGTH = self.seg.max()+1
		LX, LY = self.img.shape[0], self.img.shape[1]

		dx = [-1,0,1,0]
		dy = [0,1,0,-1]
		per_matrix = np.zeros([LENGTH, LENGTH])
		for i in range(LX):
			for j in range(LY):
				for k in range(4):
					if i+dx[k]>=0 and i+dx[k]<LX and j+dy[k]>=0 and j+dy[k]<LY:
						if self.seg[i,j] != self.seg[i+dx[k], j+dy[k]]:
							per_matrix[self.seg[i,j], self.seg[i+dx[k], j+dy[k]]] += 1
		## 归一化
		if per_matrix.max():
			per_matrix /= per_matrix.max()
		return per_matrix


	def __adjacency_matrix(self, e = 0.1):
		'''
			基于马尔科夫链得到超像素节点之间的往返时间，以此作为
			超像素节点之间的相似度
		'''

		LENGTH = self.seg.max()+1

		### 得到图的邻接矩阵
		gray_col = np.array(self.gray)
		gray_row = gray_col.reshape(-1,1)
		d = (gray_row - gray_col)*(self.adjmat>0)
		adj_matrix = self.__distance(0,d, e = e)*(self.adjmat>0)

		### 按行归一化/得到稳态pi
		adj_sum = np.sum(adj_matrix)
		D = np.sum(adj_matrix, 1)
		norm_adj_matrix = adj_matrix / D.reshape(-1,1)
		pi = D/adj_sum
		
		### 得到马尔科夫链的基础矩阵（case1:W的量级很小）
		I = np.identity(LENGTH)
		W = np.tile(pi, (LENGTH, 1))
		Z = np.linalg.inv(I - norm_adj_matrix + W)

		### 获取节点之间的期望往返时间
		pi_row = 1/pi
		Zdig = Z.diagonal()
		exp_time = pi_row*(Zdig - Z)
		exp_time += exp_time.T
		exp_time += np.diag(1/pi)

		### 归一化
		row_min = np.min(exp_time, 1).reshape(-1,1)
		exp_time -= row_min
		row_max = np.max(exp_time, 1).reshape(-1,1)
		exp_time = exp_time/row_max

		return exp_time


	def __dist_center(self):
		'''	
			得到超像素的重心，并且计算超像素重心之间的距离，以此作为后续图结点的距离
		'''

		nodes_num = self.seg.max()+1
		dis_mat = np.zeros([nodes_num, nodes_num])
		centers = []
		for i in range(nodes_num):
			index = (self.seg == i).nonzero()
			x = index[0].mean()
			y = index[1].mean()
			centers.append([x,y])
		
		def com_dis(x, y):
			return ((x[0] - y[0])**2 + (x[1] - y[1])**2)**(1/2)

		for i in range(nodes_num):
			for j in range(i+1,nodes_num):
				dis_mat[i,j] = com_dis(centers[i], centers[j])
				dis_mat[j,i] = dis_mat[i,j]
		
		dis_mat = dis_mat - dis_mat.min()
		if dis_mat.max():
			dis_mat =  dis_mat/dis_mat.max()
		return dis_mat


	def __min_path(self):
		'''	
			没有使用
			获取两个超像素之间的最短路径，比较耗时
		'''

		G=nx.DiGraph()
		nodes_num = self.seg.max()+1
		for i in range(0,nodes_num):
			G.add_node(i)

		for i in range(nodes_num):
			for j in range(i+1, nodes_num):
				if self.adjmat[i,j] > 0 or self.adjmat[j,i] > 0:
					G.add_edge(i,j)
					G.add_edge(j,i)
		
		dis_mat = np.zeros([nodes_num, nodes_num])
		for i in range(nodes_num):
			for j in range(i+1,nodes_num):
				dis_mat[i,j] = nx.shortest_path_length(G,source=i,target=j)
				dis_mat[j,i] = dis_mat[i,j]

		dis_mat = dis_mat - dis_mat.min()
		if dis_mat.max():
			dis_mat =  dis_mat/dis_mat.max()

		return dis_mat



	def ave_gray(self):
		'''	得到每个超像素的平均像素值
		'''
		num = self.seg.max() + 1
		gray = []
		for i in range(num):
			index = (self.seg == i)
			single = (index*self.img).sum() / index.sum()
			gray.append(single)
		return gray


	def init_lable(self, t = 0.5):
		'''	初始化灰度值前 1-t 的超像素为前景像素
		'''
		t = self.init_t
		lable = []
		tmp_gray = sorted(self.gray)
		t_gray = tmp_gray[int(len(tmp_gray)*t)]
		for i in self.gray:
			if i > t_gray:
				lable.append(0)
			else:
				lable.append(1)
		return lable


	def init_lable_2(self, ph = 0.2, pw = 0.2):

		ph = self.init_t
		pw = self.init_t
		tmp = np.zeros_like(self.img)
		h = self.img.shape[0]
		w = self.img.shape[1]
		tmp[int(h*ph):int(h*(1-ph)), int(w*pw):int(w*(1-pw))] = 1
		lable = []
		for i in range(self.seg.max()+1):
			index = (self.seg == i)
			if (index*tmp).sum() == index.sum():
				lable.append(0)
			else:
				lable.append(1)
		return lable


	def create_graph(self):
		'''	根据超像素为结点建图，并且为边赋值
		'''

		g = maxflow.Graph[float]()
		nodes_num = self.seg.max()+1
		g.add_nodes(nodes_num)

		## 赋值：像素内边   
		index = np.tril(self.adjmat, -1).nonzero()
		for _id in range(len(index[0])):
			i, j = index[0][_id], index[1][_id]
			sim = self.lambda_in*self.__distance(self.gray[i], self.gray[j], e=0.5) + self.adjmat[i,j]
			g.add_edge(i,j, self.mu*sim, self.mu*sim)

		## 赋值：前背景节点t,s与像素边
		# 全局能量
		pos, neg = self.get_ave()
		gray = np.array(self.gray)
		pos = (gray - pos) ** 2
		neg = (gray - neg) ** 2

		# 局部能量
		for i in range(nodes_num):

			pos_inter, neg_inter = self.local_inter(i)
			energy_s = self.local_out*neg_inter + self.global_out*neg[i]
			energy_t = self.local_in*pos_inter + self.global_in*pos[i] 
			g.add_tedge(i, energy_s, energy_t)
		
		g.maxflow()
		return g


	def __distance(self, p, q, e=0.3):
		'''	高斯距离
		'''
		assert e != 0
		index = -1*(p-q)**2/(2*e*e)
		value = math.e**(index)
		return value

	def get_ave(self):
		'''	获取全局平均的前背景像素值
		'''
		label = np.array(self.lable)
		gray = np.array(self.gray)

		w_pos = (label == 0)
		w_neg = (label == 1)

		cnt_pos, cnt_neg = w_pos.sum(), w_neg.sum()		
		pos = 0 if (cnt_pos == 0) else (gray*w_pos).sum() / cnt_pos 
		neg = 0 if (cnt_neg == 0) else (gray*w_neg).sum() / cnt_neg

		return pos, neg


	def local_inter(self, index = 0):
		'''	对内局部能量，表示:	某点为中心，与周围像素的差异加权
		'''

		gray = np.array(self.gray) - self.gray[index]
		weight = self.__distance(0,self.dis_mat[index], e = self.el)

		label = np.array(self.lable)

		tmp = (gray**2)*weight
		w_pos = (weight*(label == 0)).sum()
		w_neg = (weight*(label == 1)).sum()
		pos, neg = 0, 0
		if w_pos:
			pos = (tmp*(label == 0)).sum() / w_pos
		if w_neg:
			neg = (tmp*(label == 1)).sum() / w_neg

		return pos, neg



	def local_exter(self, index = 0):
		'''	对外局部能量，表示:	某点为中心，与周围加权平均像素的差异
			（废弃）
		'''

		weight = self.__distance(0,self.dis_mat[index], e = self.el)
		gray = np.array(self.gray)*weight
		label = np.array(self.lable)

		pos_ave = (gray*(label == 0)).sum()
		pos_cnt = (weight*(label == 0)).sum()

		neg_ave = (gray*(label == 1)).sum()
		neg_cnt = (weight*(label == 1)).sum()

		if pos_cnt:
			pos_ave /= pos_cnt
		if neg_cnt:
			neg_ave /= neg_cnt

		return (self.gray[index] - pos_ave)**2, (self.gray[index] - neg_ave)**2		


	def main(self, max_iters):

		last = 0
		cnt = 0

		while cnt < max_iters:

			cnt += 1
			g = self.create_graph()
			flow = g.maxflow()

			lable = []
			for i in range(len(self.lable)):
				lable.append(g.get_segment(i))
			self.lable = lable

			if flow == last:
				break
			last = flow


		cut = np.zeros_like(self.img)
		for i in range(len(self.lable)):
			index = (self.seg == i)
			cut += index*self.lable[i]
		
		return cut, last, cnt





		
if __name__ == "__main__":


	'''
		use demo

		step 1 : 
			segments = slic(img, n_segments=400, compactness=25, max_iter=500, convert2lab=False)
		step 2:
			tmp_cut = cv_cut(old_img, segments, init_t = 0.4)
			cut, last, cnt = tmp_cut.main(10)
	'''

	
	img_src = 'I00058.bmp'
	oldimg = np.array(imread(img_src, True), dtype='float64')
	xmin, xmax, ymin, ymax = 183, 286, 241, 319
	img_data = oldimg[ymin:ymax, xmin:xmax]


	## 归一化到0-255
	img_data = img_data/img_data.max()*255

	## superpixel
	segments = slic(img_data, n_segments=100, compactness=100, max_iter=100, convert2lab=False)

	## cv分割
	tmp_cut = cv_cut(img_data, segments, init_t = 0.05, mu=0.01, local_in = 1, local_out = 1.,
					global_in = 1, global_out = 1., lambda_in = 30)
	cut, last, cnt = tmp_cut.main(10)

	## 展示结果
	plt.ion()
	plt.axis('off')
	fig2 = plt.figure(1,(15,7))
	ax1 = fig2.add_subplot(111)
	ax1.imshow(img_data, interpolation='nearest', cmap=plt.cm.gray)
	contours = measure.find_contours(cut, 0.5)
	for n, contour in enumerate(contours):
		ax1.plot(contour[:, 1], contour[:, 0], color = 'r', linewidth=2)
	
	plt.pause(10000)
	plt.show()