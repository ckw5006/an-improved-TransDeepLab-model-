




from torch.utils.data import Dataset
import glob
import os
# from skimage.io import imread
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.data import SequentialSampler
import numpy as np
import logging
from PIL import Image
import torch
logger = logging.getLogger(__name__)

# COLOR_MAP = OrderedDict(
#     Background=(255, 255, 255),
#     Building=(255, 0, 0),
#     Road=(255, 255, 0),
#     Water=(0, 0, 255),
#     Barren=(159, 129, 183),
#     Forest=(0, 255, 0),
#     Agricultural=(255, 195, 128),
# )


LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)

def rgb_to_label(rgb_img):
    rgb_img=np.array(rgb_img)
    label = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    for i, color in enumerate(COLOR_MAP.values()):
        mask = np.all(rgb_img == color, axis=-1)
        label[mask] = i
    mask = Image.fromarray(label)
    return mask#*42.5

def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls)*label, new_cls)
    return new_cls


COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(160, 130, 180),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 130),
)




class LoveDA(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []

        image_dir, mask_dir=image_dir[0], mask_dir[0]
        print(image_dir, mask_dir,transforms)
        if isinstance(image_dir, list) and isinstance(mask_dir, list):

            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        elif isinstance(image_dir, list) and not isinstance(mask_dir, list):
            for img_dir_path in image_dir:
                self.batch_generate(img_dir_path, mask_dir)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms


    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))
        
        logger.info('%s -- Dataset images: %d' % (os.path.dirname(image_dir), len(rgb_filepath_list)))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list
    def __len__(self):
        return len(self.rgb_filepath_list)
    def __getitem__(self, idx):
        # image = imread(self.rgb_filepath_list[idx])
        image = Image.open(self.rgb_filepath_list[idx]).convert('RGB')

        # rgb_img = np.array(image)/127.-1
        # image = Image.fromarray(rgb_img)
        if len(self.cls_filepath_list) > 0:
            # mask = imread(self.cls_filepath_list[idx]).astype(np.long) -1
            
            
            mask = Image.open(self.cls_filepath_list[idx]).convert('L')
            # 将图像转换为NumPy数组
            rgb_img = np.array(mask)
            # 将所有像素值为0的像素替换为1
            rgb_img[rgb_img == 0] = 1
            rgb_img=rgb_img-1
            #print(self.cls_filepath_list[idx],np.shape(mask),np.max(mask),np.min(mask),np.shape(rgb_img),np.max(rgb_img),np.min(rgb_img))
            mask = Image.fromarray(rgb_img)
            if self.transforms is not None:
                image,mask = self.transforms(image, mask)
                # image=image/127.-1
                #print(image.size(),torch.max(image),torch.min(image),mask.size(),torch.max(mask),torch.min(mask))
                # print(torch.max(image),mask.size())
                # print(image.size(),mask.size())

                # image = blob['image']
                # mask = blob['mask']
                sample = {'image': image, 'label': mask}
                sample['case_name'] = self.rgb_filepath_list[idx].strip('\n')
                return sample
            # return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

