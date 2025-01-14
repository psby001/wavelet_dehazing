import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import v2



def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()

def read_img(filename):
	img = cv2.imread(filename)
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	return img[:, :, ::-1]

def augment(imgs=[], size=(256,256), edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = size

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :].astype('float32') / 255.0

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=(256,256)):
	H, W, _ = imgs[0].shape
	Hc, Wc = size
	Hs = (H - Hc) // 2 if H > Hc else 0
	Ws = (W - Wc) // 2 if W > Wc else 0
	if Hs == 0 and H // 16 != 0:
		Hs = H % 16
	if Ws == 0 and W // 16 != 0:
		Ws = W % 16
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :].astype('float32') / 255.0

	return imgs


def keep_same_align(imgs=[],transforms=v2.ToTensor()):
	for i in range(len(imgs)):
		imgs[i] = transforms(imgs[i].copy())
	return imgs
class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=(256,256), edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)
		self.transforms = v2.Compose([
			v2.ToTensor(),
			v2.Resize(size)
		])

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name))
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name))
		
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)
			return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)
			return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}

		if self.mode == 'test':
			[source_img, target_img] = keep_same_align([source_img, target_img],self.transforms)
			return {'source': source_img, 'target': target_img, 'filename': img_name}
		# if (not os.path.exists("test_img_gt")):
		# 	os.makedirs("test_img_gt")
		# if (not os.path.exists("test_img_hz")):
		# 	os.makedirs("test_img_hz")
		# cv2.imwrite("test_img_gt/"+img_name,target_img*255)
		# cv2.imwrite("test_img_hz/" + img_name, source_img*255)


		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}