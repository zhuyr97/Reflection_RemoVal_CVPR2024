import os.path
from os.path import join
import sys

sys.path.append('/ghome/zhuyr/Deref_RW/')
print(sys.path)
from datasets.image_folder import make_dataset
from datasets.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2, \
    ReflectionSythesis_3
from PIL import Image
import random
import torch
import math

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# import util.util as util
import datasets.torchdata as torchdata


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(224, 448) / 2.) * 2
    # target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (224, 224))
    # i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)

    return img_1, img_2


def paired_data_transforms_wcrop(img_1, img_2, unaligned_transforms=False, crop_size=256):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(crop_size, 448) / 2.) * 2
    # target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (crop_size, crop_size))
    # i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)

    return img_1, img_2


BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


class CEILDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True,
                 low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3, crop_size=256,
                 low_A=0.02, high_A=0.06, low_beta=0.02, high_beta=0.12, mode=1,
                 mask_flag=False, mask_mode=1, mask_threshold=0.1):
        super(CEILDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms

        self.mode = mode
        # mask setting
        self.mask_flag = mask_flag
        self.mask_mode = mask_mode
        self.mask_threshold = mask_threshold

        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = self.paths[:size]

        if self.mode == 1:
            self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
                                                  low_gamma=low_gamma, high_gamma=high_gamma)
        elif self.mode == 2:
            self.syn_model = ReflectionSythesis_2()
        elif self.mode == 3:
            self.syn_model = ReflectionSythesis_3(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
                                                  low_gamma=low_gamma, high_gamma=high_gamma,
                                                  low_A=low_A, high_A=high_A, low_beta=low_beta, high_beta=high_beta,
                                                  mask_mode=self.mask_mode, mask_thresh=self.mask_threshold
                                                  )
        else:
            print(' ERROR!!! Please set the accurate sythesis mode!!!  ')

        # self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma, low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)
        self.crop_size = crop_size

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.B_paths = self.paths[0:num_paths]
        self.R_paths = self.paths[num_paths:2 * num_paths]

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms_wcrop(t_img, r_img, crop_size=self.crop_size)
        syn_model = self.syn_model
        if self.mode == 3:
            t_img, r_img, m_img, mask = syn_model(t_img, r_img)
            B = to_tensor(t_img)
            R = to_tensor(r_img)
            M = to_tensor(m_img)
            mask = to_tensor(mask)  # torch.unsqueeze(to_tensor(mask), 0)

        else:
            t_img, r_img, m_img = syn_model(t_img, r_img)
            B = to_tensor(t_img)
            R = to_tensor(r_img)
            M = to_tensor(m_img)
            mask = np.zeros_like(M)

        if self.mask_flag:
            return B, R, M, mask
        else:
            return B, R, M

    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)

        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]

        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')
        fn = os.path.basename(B_path)

        if self.mode == 3 and self.mask_flag:
            B, R, M, mask = self.data_synthesis(t_img, r_img)
        else:
            B, R, M = self.data_synthesis(t_img, r_img)

        if self.mask_flag:
            return M, B, mask, fn
        else:
            return M, B, fn

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))


class CEILTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None, crop_size=256):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.crop_size = crop_size

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms_wcrop(t_img, m_img, self.unaligned_transforms,
                                                        crop_size=self.crop_size)

        B = to_tensor(t_img)
        M = to_tensor(m_img)

        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': B}  # fake reflection gt
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1

        m_img = Image.open(join(self.datadir, fn)).convert('RGB')

        M = to_tensor(m_img)
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class PairedCEILDataset(CEILDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5):
        self.size = size
        self.datadir = datadir

        self.fns = fns or os.listdir(join(datadir, 'reflection_layer'))
        if size is not None:
            self.fns = self.fns[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma)
        self.enable_transforms = enable_transforms
        self.reset()

    def reset(self):
        return

    def __getitem__(self, index):
        fn = self.fns[index]
        B_path = join(self.datadir, 'transmission_layer', fn)
        R_path = join(self.datadir, 'reflection_layer', fn)

        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        B, R, M = self.data_synthesis(t_img, r_img)

        data = {'input': M, 'target_t': B, 'target_r': R, 'fn': fn}
        # return M, B
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


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


class RepeatedDataset(BaseDataset):
    def __init__(self, dataset, repeat=1):
        self.dataset = dataset
        self.size = len(dataset) * repeat
        # self.reset()

    def reset(self):
        self.dataset.reset()

    def __getitem__(self, index):
        dataset = self.dataset
        return dataset[index % len(dataset)]

    def __len__(self):
        return self.size


from datasets.image_folder import read_fns

if __name__ == '__main__':
    import sys
    import numpy as np
    import matplotlib.image as img

    save_synD_path1 = '/gdata1/zhuyr/Deref/training_data/synD_ours/blended/'
    save_synD_path2 = '/gdata1/zhuyr/Deref/training_data/synD_ours/transmission_layer/'
    os.makedirs(save_synD_path2, exist_ok=True)
    os.makedirs(save_synD_path1, exist_ok=True)

    sys.path.append('/ghome/zhuyr/Deref_RW/')
    print(sys.path)
    datadir_syn = '/gdata1/zhuyr/Deref/training_data/JPEGImages'

    train_datasets = CEILDataset(
        datadir_syn, read_fns('/ghome/zhuyr/ADeref_two1/VOC2012_224_train_png.txt'), size=None,
        enable_transforms=True, crop_size=400,
        low_A=0.02, high_A=0.06, low_beta=0.02, high_beta=0.06, mode=3)

    train_loader = DataLoader(dataset=train_datasets, batch_size=1, num_workers=8, shuffle=True)
    for i, train_data in enumerate(train_loader, 0):
        inputs, label, img_name = train_data  # train_data['input'], train_data['target_t'], train_data['fn']

        data_in = inputs
        label = label

        in_np = np.squeeze(torch.clamp(data_in, 0., 1.).cpu().detach().numpy()).transpose((1, 2, 0))
        label_np = np.squeeze(torch.clamp(label, 0., 1.).cpu().detach().numpy()).transpose((1, 2, 0))
        # print(save_synD_path1 + img_name[0])
        img.imsave(save_synD_path1 + img_name[0], np.uint8(in_np * 255.))
        img.imsave(save_synD_path2 + img_name[0], np.uint8(label_np * 255.))
        if i % 100:
            print(i, len(train_loader))


