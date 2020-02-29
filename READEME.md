import os
from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
from skimage import io
import glob
import numpy as np


def read_dicom(fpath):
	img_im = pydicom.read_file(fpath)
	img = img_im.pixel_array.astype(np.float)
	return img, img_im.SeriesDescription

dirName = 'p27238'
sudDir = 's2'

if not os.path.exists(os.path.join('img', dirName)):
	os.mkdir(os.path.join('img', dirName))
if not os.path.exists(os.path.join('img', dirName, sudDir)):
	os.mkdir(os.path.join('img', dirName, sudDir))


imgs = glob.glob(dirName +'/' + sudDir +'/*.dcm')

for name in imgs:

	img, des = read_dicom(name)
	img = img/img.max()

	if not os.path.exists(os.path.join('img', dirName, sudDir, des)):
		os.mkdir(os.path.join('img', dirName, sudDir, des))

	io.imsave(os.path.join('img', dirName, sudDir, des, name.split('/')[-1][:-4]+'.png'),img)

	print (name)