#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import nrrd
from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata, threshold_li
from skimage.filters import threshold_mean, threshold_niblack, threshold_sauvola, threshold_triangle
import numpy as np 


folder_path = "/groups/scicompsoft/home/dingx/Documents/ExM/data/optic_lobe_1/"

img, head = nrrd.read(folder_path+"C1-4228_3823_4701-background_subtract.nrrd")

thres_otsu = threshold_otsu(img)
img_otsu = np.zeros(img.shape, dtype=img.dtype)
img_otsu[img>thres_otsu] = 255
nrrd.write(folder_path+"img_otsu.nrrd", img_otsu)

thres_yen = threshold_yen(img)
img_yen = np.zeros(img.shape, dtype=img.dtype)
img_yen[img>thres_yen] = 255
nrrd.write(folder_path+"img_yen.nrrd", img_yen)

thres_isodata = threshold_isodata(img)
img_isodata = np.zeros(img.shape, dtype=img.dtype)
img_isodata[img>thres_isodata] = 255
nrrd.write(folder_path+"img_isodata.nrrd", img_isodata)

thres_li = threshold_li(img)
img_li = np.zeros(img.shape, dtype=img.dtype)
img_li[img>thres_li] = 255
nrrd.write(folder_path+"img_li.nrrd", img_li)

thres_mean = threshold_mean(img)
img_mean = np.zeros(img.shape, dtype=img.dtype)
img_mean[img>thres_mean] = 255
nrrd.write(folder_path+"img_mean.nrrd", img_mean)

thres_niblack = threshold_niblack(img)
img_niblack = np.zeros(img.shape, dtype=img.dtype)
img_niblack[img>thres_niblack] = 255
nrrd.write(folder_path+"img_niblack.nrrd", img_niblack)

thres_sauvola = threshold_sauvola(img)
img_sauvola = np.zeros(img.shape, dtype=img.dtype)
img_sauvola[img>thres_sauvola] = 255
nrrd.write(folder_path+"img_sauvola.nrrd", img_sauvola)

thres_triangle = threshold_triangle(img)
img_triangle = np.zeros(img.shape, dtype=img.dtype)
img_triangle[img>thres_triangle] = 255
nrrd.write(folder_path+"img_triangle.nrrd", img_triangle)