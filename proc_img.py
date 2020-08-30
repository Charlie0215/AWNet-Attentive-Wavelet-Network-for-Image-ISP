import cv2
import numpy as np
import os

def add_boarder(img, width=3):
	img = cv2.copyMakeBorder(img, width, width, width, width, cv2.BORDER_CONSTANT, value=(0,255,0))
	return img

def crop(img, point, dist, scale=2, boarder_width=3):
	#
	cropped = img[point[0]:point[0]+dist, point[1]:point[1]+dist, :]
	rescaled = rescale(cropped, scale)
	cropped_boarder = add_boarder(cropped, boarder_width)
	cropped_rescale_boarder = add_boarder(rescaled, boarder_width)
	print('####', cropped_rescale_boarder.shape)
	return cropped_boarder, cropped_rescale_boarder

def rescale(img ,scale):
	return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

def add_back(img, 
			cropped_boarder, 
			cropped_rescale_boarder, 
			point, 
			crop_size, 
			boarder_width,
			scale
			):
	'''Added cropped area'''
	img[point[0]-boarder_width:point[0]+crop_size+boarder_width, point[1]-boarder_width:point[1]+crop_size+boarder_width, :] = cropped_boarder
	'''Added rescale area to corner'''
	img[img.shape[0]-int(crop_size*scale)-boarder_width*2:img.shape[0],img.shape[1]-int(crop_size*scale)-boarder_width*2:img.shape[1],:] = cropped_rescale_boarder
	return img

def ddn_real_2():
	boarder_width = 3
	crop_size = 200
	scale = 2
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ddn_real_png/2/2_rainy.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ddn_real_png/2/'
	save_folder = './experiment_images_on_latex/experiments/ddn_real_png/2/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (img.shape[0]//11, img.shape[1]//3) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)

def ddn_real_19():
	boarder_width = 3
	crop_size = 100
	scale = 1.5
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ddn_real_png/19/19_rainy.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ddn_real_png/19/'
	save_folder = './experiment_images_on_latex/experiments/ddn_real_png/19/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (10, img.shape[1]//4*3) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)

def our_real_2():
	boarder_width = 3
	crop_size = 200
	scale = 2
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ours_real_png/2/2_rainy.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ours_real_png/2/'
	save_folder = './experiment_images_on_latex/experiments/ours_real_png/2/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (4, 4) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)

def our_real_1218():
	boarder_width = 3
	crop_size = 200
	scale = 2
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ours_real_png/rain_01218/rain_01218_rainy.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ours_real_png/rain_01218/'
	save_folder = './experiment_images_on_latex/experiments/ours_real_png/rain_01218/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (img.shape[0]//4, img.shape[1]//4) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)

def ddn_901():
	boarder_width = 3
	crop_size = 100
	scale = 2
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ddn_test/901/901.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ddn_test/901/'
	save_folder = './experiment_images_on_latex/experiments/ddn_test/901/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (img.shape[0]//2, img.shape[1]//4) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)

def ddn_905():
	boarder_width = 3
	crop_size = 100
	scale = 2
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ddn_test/905/905.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ddn_test/905/'
	save_folder = './experiment_images_on_latex/experiments/ddn_test/905/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (img.shape[0]//3, 3) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)


def ours_563():
	boarder_width = 3
	crop_size = 100
	scale = 2
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ours_test/0563/0563.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ours_test/0563/'
	save_folder = './experiment_images_on_latex/experiments/ours_test/0563/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (200, 100) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)


def ours_2800():
	boarder_width = 3
	crop_size = 100
	scale = 2
	s_img = cv2.imread('./experiment_images_on_latex/experiments/ours_test/2800/2800.jpg')
	img_shape = s_img.shape
	print(img_shape)
	folder = './experiment_images_on_latex/experiments/ours_test/2800/'
	save_folder = './experiment_images_on_latex/experiments/ours_test/2800/refine'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	filenames = os.listdir(folder)
	for filename in filenames:
		if filename.endswith('.jpg') or filename.endswith('.png'):
			img_name = os.path.join(folder, filename)
			print(img_name)
			img = cv2.imread(img_name)
			img = cv2.resize(img, (img_shape[1], img_shape[0]))
			point = (350, 370) #h, w
			cropped_boarder, cropped_rescale_boarder = crop(img, point, crop_size, scale=scale)
			result = add_back(img, cropped_boarder, cropped_rescale_boarder, point, crop_size, boarder_width, scale)
			save_path = os.path.join(save_folder, filename)
			cv2.imwrite(save_path, result)

if __name__ == '__main__':
	#ddn_real_2()
	#ddn_real_19()
	#our_real_2()
	#our_real_1218()
	#ddn_901()
	#ddn_905()
	#ours_563()
	ours_2800()
	