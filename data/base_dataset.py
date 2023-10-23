import random
from random import choices
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from skimage import util, feature
from skimage.color import rgb2gray
import torch
import numbers


from util import tps_warp


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size, input_im):
    new_w, new_h = size
    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    params = {'crop_pos': (x, y), 'crop': random.random() > 0.5, "flip": random.random() > 0.5}

    for affine_trans in opt.affine_transforms.keys():
        apply_affine_trans = (random.random() > 0.5) and affine_trans in opt.affine_aug
        if apply_affine_trans:
            params[affine_trans] = random.uniform(opt.affine_transforms[affine_trans][0],
                                                  opt.affine_transforms[affine_trans][1])

    # choose: affine / tps / identity
    apply_tps = random.random() < opt.tps_percent
    apply_affine = not apply_tps
    params["apply_affine"] = apply_affine

    np_im = np.array(input_im)

    random_num = random.randint(1, 3)
    if random_num == 1:
        src = tps_warp._get_regular_grid_new_ours(np_im, points_per_dim=random_num)
        dst = tps_warp._generate_random_vectors(np_im, src, scale=0.25 * new_w)
        params['tps'] = {'src': src, 'dst': dst, 'apply_tps': apply_tps}
    if random_num == 2:
        src = tps_warp._get_regular_grid_new_ours(np_im, points_per_dim=random_num)
        dst = tps_warp._generate_random_vectors(np_im, src, scale=0.167 * new_w)
        params['tps'] = {'src': src, 'dst': dst, 'apply_tps': apply_tps}
    elif random_num == 3:
        src = tps_warp._get_regular_grid_new_ours(np_im, points_per_dim=random_num)
        dst = tps_warp._generate_random_vectors(np_im, src, scale=0.125 * new_w)
        params['tps'] = {'src': src, 'dst': dst, 'apply_tps': apply_tps}

    if opt.cutmix_aug:
        patch_size = random.randint(opt.cutmix_min_size, opt.cutmix_max_size)
        first_cutmix_x = random.randint(0, np.maximum(0, new_w - patch_size))
        first_cutmix_y = random.randint(0, np.maximum(0, new_h - patch_size))
        second_cutmix_x = random.randint(0, np.maximum(0, new_w - patch_size))
        second_cutmix_y = random.randint(0, np.maximum(0, new_h - patch_size))
        params['cutmix'] = {'first_cutmix_x': first_cutmix_x, 'first_cutmix_y': first_cutmix_y,
                            'second_cutmix_x': second_cutmix_x, 'second_cutmix_y': second_cutmix_y,
                            'patch_size': patch_size, 'apply': random.random() > 0.5}

    if opt.canny_aug:
        canny_img = _create_canny_aug(np_im, opt.canny_color, opt.canny_sigma_l_bound, opt.canny_sigma_u_bound, opt.canny_sigma_step)
        params['canny_img'] = canny_img
    return params


def get_transform(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            transform_list.append(
                transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img,opt.canny_color)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        # transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
        #                                         (0.5, 0.5, 0.5))]
        transform_list += [transforms.Normalize((0.5),
                                                (0.5))]
    return transforms.Compose(transform_list)


def get_transform_new_ours(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        tps_change = False
        if opt.tps_aug:
            tps_change_prob = random.random()
            if tps_change_prob > 0.5:
                tps_change = True
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img,opt.canny_color)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        # transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
        #                                         (0.5, 0.5, 0.5))]
        transform_list += [transforms.Normalize((0.5),
                                                (0.5))]
    return transforms.Compose(transform_list), tps_change


def get_transform_without_tps_aug(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        # if opt.tps_aug:
        #     transform_list.append(
        #         transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img,opt.canny_color)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class ColorJitterOurs(torch.nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

        self.fn_idx = torch.randperm(4)
        # brightness
        self.brightness_factor = torch.tensor(1.0).uniform_(self.brightness[0], self.brightness[1]).item()
        # contrast
        self.contrast_factor = torch.tensor(1.0).uniform_(self.contrast[0], self.contrast[1]).item()
        # saturation
        self.saturation_factor = torch.tensor(1.0).uniform_(self.saturation[0], self.saturation[1]).item()
        # hue
        self.hue_factor = torch.tensor(1.0).uniform_(self.hue[0], self.hue[1]).item()

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness is not None:
                img = F.adjust_brightness(img, self.brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                img = F.adjust_contrast(img, self.contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                img = F.adjust_saturation(img, self.saturation_factor)

            if fn_id == 3 and self.hue is not None:
                img = F.adjust_hue(img, self.hue_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class ColorJitterOurs_1(torch.nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, fn_idx=None, brightness_factor=0, contrast_factor=0, saturation_factor=0, hue_factor=0):
        super().__init__()


        self.fn_idx = fn_idx
        # brightness
        self.brightness_factor = brightness_factor
        # contrast
        self.contrast_factor = contrast_factor
        # saturation
        self.saturation_factor = saturation_factor
        # hue
        self.hue_factor = hue_factor

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness_factor is not None:
                img = F.adjust_brightness(img, self.brightness_factor)

            if fn_id == 1 and self.contrast_factor is not None:
                img = F.adjust_contrast(img, self.contrast_factor)

            if fn_id == 2 and self.saturation_factor is not None:
                img = F.adjust_saturation(img, self.saturation_factor)

            if fn_id == 3 and self.hue_factor is not None:
                img = F.adjust_hue(img, self.hue_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


def get_transform_mode_1(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list = []
    transform_list_tgt = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params)))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            # transform_list.append(
            #     transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img,opt.canny_color)))
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))

    color_change_prob = random.random()
    if color_change_prob > 0.05:
        color_aug = ColorJitterOurs(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        transform_list.append(color_aug)
        color_aug_1 = ColorJitterOurs_1(fn_idx=color_aug.fn_idx, brightness_factor=color_aug.brightness_factor, contrast_factor=color_aug.contrast_factor,
                                        saturation_factor=color_aug.saturation_factor, hue_factor=color_aug.hue_factor)
        transform_list_tgt.append(color_aug_1)

    transform_list += [transforms.ToTensor()]
    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)


def get_transform_mode_1_palette_first(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list = []
    transform_list_tgt = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params)))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            # transform_list.append(
            #     transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img,opt.canny_color)))
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))

    color_change_prob = random.random()
    if color_change_prob > 0.05:
        color_aug = ColorJitterOurs(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        transform_list.append(color_aug)
        color_aug_1 = ColorJitterOurs_1(fn_idx=color_aug.fn_idx, brightness_factor=color_aug.brightness_factor, contrast_factor=color_aug.contrast_factor,
                                        saturation_factor=color_aug.saturation_factor, hue_factor=color_aug.hue_factor)
        transform_list_tgt.append(color_aug_1)

    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)


def get_transform_mode_1_palette_second(normalize=True):
    transform_list = []
    transform_list_tgt = []
    transform_list += [transforms.ToTensor()]
    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)


def get_transform_mode_1_contrastive(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list_tgt = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))

    color_aug = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
    transform_list_tgt.append(color_aug)

    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list_tgt)


def get_transform_mode_2(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list = []
    transform_list_tgt = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params)))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            transform_list.append(
                transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))
            # transform_list_tgt.append(
            #     transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img,opt.canny_color)))
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))

    color_change_prob = random.random()
    if color_change_prob > 0.05:
        color_aug = ColorJitterOurs(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        transform_list.append(color_aug)
        color_aug_1 = ColorJitterOurs_1(fn_idx=color_aug.fn_idx, brightness_factor=color_aug.brightness_factor, contrast_factor=color_aug.contrast_factor,
                                        saturation_factor=color_aug.saturation_factor, hue_factor=color_aug.hue_factor)
        transform_list_tgt.append(color_aug_1)

    transform_list += [transforms.ToTensor()]
    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)


def get_transform_mode_2_contrastive(opt, params, normalize=True, is_primitive=False, is_edges=False):
    transform_list_tgt = []
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        # if opt.tps_aug:
            # 变形彩色参考图
            # transform_list_tgt.append(
            #     transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))

    color_aug = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
    transform_list_tgt.append(color_aug)

    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list_tgt)


# 变形线稿图　+　变形彩色参考图1　=　变形彩色图
def get_transform_mode_3(opt, params, params_bianxing, normalize=True, is_primitive=False, is_edges=False, tps_change_edge=False):
    transform_list = []  # 变形彩色参考图1
    transform_list_tgt = []  # 变形彩色图
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params_bianxing['canny_img'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        color_change_prob = random.random()
        if color_change_prob > 0.05:
            color_aug = ColorJitterOurs(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)
            transform_list.append(color_aug)
            color_aug_1 = ColorJitterOurs_1(fn_idx=color_aug.fn_idx, brightness_factor=color_aug.brightness_factor,
                                            contrast_factor=color_aug.contrast_factor,
                                            saturation_factor=color_aug.saturation_factor,
                                            hue_factor=color_aug.hue_factor)
            transform_list_tgt.append(color_aug_1)

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params_bianxing['flip'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params_bianxing)))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            # 变形彩色参考图
            tps_change_reference_prob = random.random()
            if tps_change_reference_prob > 0.04:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_tps(img, params_bianxing['tps'])))
            if tps_change_edge:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params_bianxing['cutmix'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params_bianxing['crop'])))
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))

    transform_list += [transforms.ToTensor()]
    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)


# 变形线稿图　+　变形彩色参考图1　=　变形彩色图
def get_transform_mode_3_palette_first(opt, params, params_bianxing, normalize=True, is_primitive=False, is_edges=False, tps_change_edge=False):
    transform_list = []  # 变形彩色参考图1
    transform_list_tgt = []  # 变形彩色图


    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params_bianxing['canny_img'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params_bianxing['flip'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params_bianxing)))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))


        if opt.tps_aug:
            # 变形彩色参考图
            tps_change_reference_prob = random.random()
            if tps_change_reference_prob > 0.04:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_tps(img, params_bianxing['tps'])))
            if tps_change_edge:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params_bianxing['cutmix'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params_bianxing['crop'])))
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))


    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)

# 变形线稿图　+　变形彩色参考图1　=　变形彩色图
def get_transform_mode_3_palette_first_inference(opt, params, params_bianxing, normalize=True, is_primitive=False, is_edges=False, tps_change_edge=False):
    transform_list = []  # 变形彩色参考图1
    transform_list_tgt = []  # 变形彩色图


    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params_bianxing['canny_img'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params_bianxing['flip'])))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list.append(
                transforms.Lambda(lambda img: __affine(img, params_bianxing)))
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            transform_list.append(
                transforms.Lambda(lambda img: __apply_tps(img, params_bianxing['tps'])))

            if tps_change_edge:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params_bianxing['cutmix'])))
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params_bianxing['crop'])))
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))


    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)


# 变形线稿图　+　变形彩色参考图1　=　变形彩色图
def get_transform_mode_3_palette_second(normalize=True):
    transform_list = []
    transform_list_tgt = []

    transform_list += [transforms.ToTensor()]
    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list), transforms.Compose(transform_list_tgt)


# 需要的是变形彩色图
# 变形线稿图　+　变形彩色参考图1　=　变形彩色图
def get_transform_mode_3_contrastive(opt, params, normalize=True, is_primitive=False, is_edges=False, tps_change_edge=False):
    transform_list_tgt = []  # 变形彩色图
    if opt.isTrain:
        # transforms applied only to the primitive
        if is_primitive:
            if opt.canny_aug and is_edges:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __add_canny_img(img, params['canny_img'])))

        color_aug = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        transform_list_tgt.append(color_aug)

        # transforms applied to both the primitive and the real image
        if not opt.no_flip:
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if opt.affine_aug != "none":
            transform_list_tgt.append(
                transforms.Lambda(lambda img: __affine(img, params)))

        if opt.tps_aug:
            if tps_change_edge:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_tps(img, params['tps'])))

        if opt.cutmix_aug:
            if 'cutmix' in params:
                transform_list_tgt.append(
                    transforms.Lambda(lambda img: __apply_cutmix(img, params['cutmix'])))

        if 'crop' in opt.resize_or_crop:
            transform_list_tgt.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.fineSize, params['crop'])))

    if is_edges:
        transform_list_tgt.append(transforms.Lambda(lambda img: __binary_thresh(img, opt.canny_color)))


    transform_list_tgt += [transforms.ToTensor()]

    if normalize:
        transform_list_tgt += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list_tgt)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


# ====== primitive and real image augmentations ======
def __crop(img, pos, size, crop):
    if crop:
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            im = img.crop((x1, y1, x1 + tw, y1 + th))
            im = im.resize((ow, oh), Image.BICUBIC)
            return im
    return img


def __flip(img, flip):
    if flip:
        im = img.transpose(Image.FLIP_LEFT_RIGHT)
        return im
    return img


def __affine(img, params):
    if params["apply_affine"]:
        affine_map = {"shearx": __apply_shear_x,
                      "sheary": __apply_shear_y,
                      "translationx": __apply_translation_x,
                      "translationy": __apply_translation_y,
                      "rotation": __apply_rotation}
        for affine_trans in affine_map.keys():
            if affine_trans in params.keys():
                img = affine_map[affine_trans](img, params[affine_trans])
    return img


def __apply_tps(img, tps_params):
    new_im = img
    if tps_params['apply_tps']:
        np_im = np.array(img)
        np_im = tps_warp.tps_warp_2(np_im, tps_params['dst'], tps_params['src'])
        new_im = Image.fromarray(np_im)
    return new_im


def __apply_cutmix(img, cutmix_params):
    if cutmix_params["apply"]:
        np_im = np.array(img)
        patch_size = cutmix_params["patch_size"]
        first_patch = np_im[cutmix_params["first_cutmix_y"]:cutmix_params["first_cutmix_y"] + patch_size,
                      cutmix_params["first_cutmix_x"]:cutmix_params["first_cutmix_x"] + patch_size, :].copy()
        second_patch = np_im[cutmix_params["second_cutmix_y"]:cutmix_params["second_cutmix_y"] + patch_size,
                       cutmix_params["second_cutmix_x"]:cutmix_params["second_cutmix_x"] + patch_size, :].copy()
        np_im[cutmix_params["first_cutmix_y"]:cutmix_params["first_cutmix_y"] + patch_size,
        cutmix_params["first_cutmix_x"]:cutmix_params["first_cutmix_x"] + patch_size, :] = second_patch
        np_im[cutmix_params["second_cutmix_y"]:cutmix_params["second_cutmix_y"] + patch_size,
        cutmix_params["second_cutmix_x"]:cutmix_params["second_cutmix_x"] + patch_size, :] = first_patch
        new_im = Image.fromarray(np_im)
        return new_im
    return img


def __apply_shear_x(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def __apply_shear_y(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def __apply_translation_x(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.2 <= v <= 0.2
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def __apply_translation_y(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.2 <= v <= 0.2
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def __apply_rotation(img, v):  # [-10, 10]
    v = v * 10
    assert -10 <= v <= 10
    return img.rotate(v)


# ====== primitive augmentations ======
def _create_canny_aug(np_im,canny_color, l_bound, u_bound, step):
    population = np.arange(l_bound, u_bound, step)
    canny_sigma = choices(population)
    img_gray = rgb2gray(np_im)
    img_canny = feature.canny(img_gray, sigma=canny_sigma[0])
    if canny_color ==0:
        img_canny = util.invert(img_canny)
    return img_canny


def __add_canny_img(input_im, canny_img):
    canny_lst = [canny_img.astype(np.int) for i in range(np.array(input_im).ndim)]
    canny_stack = (np.stack(canny_lst, axis=2) * 255).astype(np.uint8)
    return Image.fromarray(canny_stack)


def __binary_thresh(edges,canny_color):
    np_edges = np.array(edges)

    if canny_color == 0:
        np_edges[np_edges != np_edges.max()] = np_edges.min()
    else:
        np_edges[np_edges != np_edges.min()] = np_edges.max()
    return Image.fromarray(np_edges)



