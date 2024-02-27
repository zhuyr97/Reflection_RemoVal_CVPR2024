import torch,os,random,glob,math
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader



class my_dataset(Dataset):
    def __init__(self, rootA_in, rootA_label ,crop_size =256, fix_sample_A = 500, regular_aug =False):
        super(my_dataset,self).__init__()

        self.regular_aug = regular_aug
        #in_imgs
        self.fix_sample_A = fix_sample_A

        in_files_A = os.listdir(rootA_in)
        if self.fix_sample_A > len(in_files_A):
            self.fix_sample_A = len(in_files_A)
        in_files_A = random.sample(in_files_A, self.fix_sample_A)
        self.imgs_in_A = [os.path.join(rootA_in, k) for k in in_files_A]
        self.imgs_gt_A = [os.path.join(rootA_label, k) for k in in_files_A]#gt_imgs

        len_imgs_in_A = len(self.imgs_in_A)
        self.length = len_imgs_in_A
        self.crop_size = crop_size
    def __getitem__(self, index):
        data_IN_A, data_GT_A, img_name_A = self.read_imgs_pair(self.imgs_in_A[index], self.imgs_gt_A[index],
                                                               self.train_transform, self.crop_size)
        return data_IN_A, data_GT_A, img_name_A

    def read_imgs_pair(self,in_path, gt_path, transform, crop_size):
        in_img_path_A = in_path  #
        img_name_A = in_img_path_A.split('/')[-1]

        in_img_A = np.array(Image.open(in_img_path_A))
        gt_img_path_A = gt_path  # self.imgs_gt_A[index]

        gt_img_A = np.array(Image.open(gt_img_path_A))
        data_IN_A, data_GT_A = transform(in_img_A, gt_img_A, crop_size)

        return data_IN_A, data_GT_A, img_name_A

    def augment_img(self, img, mode=0):
        """图片随机旋转"""
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.flipud(np.rot90(img, k=3)))

    def train_transform(self, img, label, patch_size=256):
        """对图片和标签做一些数值处理"""
        ih, iw,_ = img.shape

        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))
        img = img[iy:iy + patch_size, ix: ix + patch_size]
        label = label[iy:iy + patch_size, ix: ix + patch_size]

        #print('debug',img.shape)
        if self.regular_aug:
            mode = random.randint(0, 7)
            # img = np.expand_dims(img, axis=2)
            # label = np.expand_dims(label, axis=2)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
            img = img.copy()
            label = label.copy()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        label = transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs_in_A)


def read_txt(txt_name = 'RealHaze.txt',sample_num=5000):
    path_in = []
    path_gt = []
    paths =[]
    with open(txt_name, 'r') as f:  # RealSnow
        for line in f:
            # print(line.strip('\n'))
            paths.append(line.strip('\n'))
    if sample_num > len(paths):
        sample_num = len(paths)
    paths_random = random.sample(paths, sample_num)
    for path in paths_random:
        path_in.append(path.strip('\n').split(' ')[0])
        path_gt.append(path.strip('\n').split(' ')[1])
    return path_in,path_gt

class my_dataset_wTxt(Dataset):
    def __init__(self, rootA, rootA_txt, crop_size=256, fix_sample_A=500, regular_aug=False):
        super(my_dataset_wTxt, self).__init__()

        self.regular_aug = regular_aug
        # in_imgs
        self.fix_sample_A = fix_sample_A

        in_files_A, gt_files_A = read_txt(rootA_txt, sample_num=self.fix_sample_A)  #os.listdir(rootA_in)
        self.imgs_in_A = [rootA + k for k in in_files_A]#os.path.join(rootA_in, k)
        self.imgs_gt_A = [rootA + k for k in gt_files_A]#gt_imgs  os.path.join(rootA_label

        # in_files_A = os.listdir(rootA_in)
        # if self.fix_sample_A > len(in_files_A):
        #     self.fix_sample_A = len(in_files_A)
        # in_files_A = random.sample(in_files_A, self.fix_sample_A)
        # self.imgs_in_A = [os.path.join(rootA_in, k) for k in in_files_A]
        # self.imgs_gt_A = [os.path.join(rootA_label, k) for k in in_files_A]  # gt_imgs

        len_imgs_in_A = len(self.imgs_in_A)
        self.length = len_imgs_in_A
        self.crop_size = crop_size

    def __getitem__(self, index):
        data_IN_A, data_GT_A, img_name_A = self.read_imgs_pair(self.imgs_in_A[index], self.imgs_gt_A[index],
                                                               self.train_transform, self.crop_size)
        return data_IN_A, data_GT_A, img_name_A

    def read_imgs_pair(self, in_path, gt_path, transform, crop_size):
        in_img_path_A = in_path  #
        img_name_A = in_img_path_A.split('/')[-1]

        in_img_A = np.array(Image.open(in_img_path_A))
        gt_img_path_A = gt_path  # self.imgs_gt_A[index]

        gt_img_A = np.array(Image.open(gt_img_path_A))
        data_IN_A, data_GT_A = transform(in_img_A, gt_img_A, crop_size)

        return data_IN_A, data_GT_A, img_name_A

    def augment_img(self, img, mode=0):
        """图片随机旋转"""
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.flipud(np.rot90(img, k=3)))

    def train_transform(self, img, label, patch_size=256):
        """对图片和标签做一些数值处理"""
        ih, iw, _ = img.shape

        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))
        img = img[iy:iy + patch_size, ix: ix + patch_size]
        label = label[iy:iy + patch_size, ix: ix + patch_size]

        #print('---debug', img.shape)
        if self.regular_aug:
            mode = random.randint(0, 7)
            # img = np.expand_dims(img, axis=2)
            # label = np.expand_dims(label, axis=2)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
            img = img.copy()
            label = label.copy()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        label = transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs_in_A)

class my_dataset_eval(Dataset):
    def __init__(self,root_in, root_label, transform =None,fix_sample=100):
        super(my_dataset_eval,self).__init__()
        #in_imgs

        self.fix_sample = fix_sample
        in_files = sorted(os.listdir(root_in))
        if self.fix_sample > len(in_files):
            self.fix_sample = len(in_files)
        #in_files = random.sample(in_files, self.fix_sample)
        gt_files = sorted(os.listdir(root_label))
        # print('--' * 40)
        # print(in_files)
        # print(gt_files)
        # print('--'*40)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        #gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]

        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        img_name =in_img_path.split('/')[-1]

        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)

        data_IN = self.transform(in_img)
        data_GT = self.transform(gt_img)

        _, h, w = data_GT.shape
        if (h % 16 != 0) or (w % 16 != 0):
            data_GT = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(data_GT)
            data_IN = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(data_IN)

        return data_IN,data_GT,img_name

    def __len__(self):
        return len(self.imgs_in)


class DatasetForInference(Dataset):
    def __init__(self, dir_path):
        self.image_paths =  glob.glob( os.path.join(dir_path, '*') )
        self.transform = transforms.Compose([
            transforms.Resize([128, 128]),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]) #transforms.ToTensor()
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        input_path = self.image_paths[index]
        input_image = Image.open(input_path).convert('RGB')
        input_image = self.transform(input_image)
        _, h, w = input_image.shape
        if (h%16 != 0) or (w%16 != 0):
            input_image = transforms.Resize(((h//16)*16, (w//16)*16))(input_image)
        return input_image #, os.path.basename(input_path)



import bisect
import warnings

from torch._utils import _accumulate
from torch import randperm


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class BaseDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def reset(self):
        return


class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1. / len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' % (
        self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size



if __name__ == '__main__':
    # rootA_in = 'C://Users//10219//Downloads//Restoration_Codes//data//solid200//blended//'
    # rootA_label = 'C://Users//10219//Downloads//Restoration_Codes//data//solid200//transmission_layer//'
    #
    # train_set = my_dataset(rootA_in, rootA_label, crop_size =224, fix_sample_A = 20,regular_aug = False)
    # train_loader = DataLoader(train_set, batch_size=2, num_workers=4, shuffle=True, drop_last=False,pin_memory=True)
    # for train_idx, train_data in enumerate(train_loader):
    #     inputs, label, img_name = train_data
    #     print('---------',train_idx)
    #     print(inputs.size(),label.size(), img_name)

    # print('-=-=-'*20)
    # root = 'C://Users//10219//Downloads//Restoration_Codes//data/'
    # root_txt = 'C://Users//10219//Downloads//Restoration_Codes//solid200.txt'

    #
    # train_set_wTxt = my_dataset_wTxt(root, root_txt, crop_size =224, fix_sample_A = 20,regular_aug = False)
    # train_set_wTxt1 = my_dataset_wTxt(root, root_txt1, crop_size =224, fix_sample_A = 20,regular_aug = False)
    #
    #
    # train_loader_wTxt = DataLoader(train_set_wTxt, batch_size=2, num_workers=4, shuffle=True, drop_last=False,pin_memory=True)
    # for train_idx, train_data in enumerate(train_loader_wTxt):
    #     inputs, label, img_name = train_data
    #     print('---------',train_idx)
    #     print(inputs.size(),label.size(), img_name)

    print('-=-=-' * 20)
    root = 'D://Datasets//Reflection//Check_SIRR/'
    root_txt = 'D://Datasets//Reflection//Check_SIRR//DeRef_USTC.txt'
    root_txt1 = 'D://Datasets//Reflection//Check_SIRR//real_train.txt'

    train_set_wTxt = my_dataset_wTxt(root, root_txt, crop_size=224, fix_sample_A=20, regular_aug=False)
    train_set_wTxt1 = my_dataset_wTxt(root, root_txt1, crop_size=224, fix_sample_A=200, regular_aug=False)

    train_set = FusionDataset([train_set_wTxt, train_set_wTxt1], [0.7, 0.3])
    train_loader_wTxt = DataLoader(train_set, batch_size=2, num_workers=4, shuffle=True, drop_last=False,
                                   pin_memory=True)
    for train_idx, train_data in enumerate(train_loader_wTxt):
        inputs, label, img_name = train_data
        print('---------', train_idx)
        print(inputs.size(), label.size(), img_name)

