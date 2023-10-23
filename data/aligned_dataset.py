import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.base_dataset import get_transform_new_ours
from data.base_dataset import get_transform_mode_3_palette_first, get_transform_mode_3_palette_second, get_transform_mode_3_palette_first_inference
from data.image_folder import make_dataset
from PIL import Image
import imageio
from skimage import util, feature
from skimage.color import rgb2gray
import numpy as np
from data.extract_color_theme import extract_color_theme_MMCQ, extract_color_theme_MMCQ_1
import json


from .transforms import (
    RandomHue
)

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.color_theme_dict = None

        # icon
        with open(self.opt.train_palette_json, encoding="utf-8") as fp:
            self.color_theme_dict = json.load(fp)

        ### input A (label maps)
        if opt.primitive != "seg_edges":
            dir_A = "_" + opt.primitive
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            # self.A = Image.open(self.A_paths[0])
            self.A_paths = self.A_paths[:opt.max_dataset_size]

        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + "_seg")
            self.A_paths = sorted(make_dataset(self.dir_A))
            # the seg input will be saved as "A"
            self.A = Image.open(self.A_paths[0])
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.mkdir(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_edges = Image.open(self.A_paths_edges[0]) if self.A_paths_edges else None

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))
            # self.B = Image.open(self.B_paths[0]).convert('RGB')
            self.B_paths = self.B_paths[:opt.max_dataset_size]
            if opt.primitive == "seg_edges" and not self.A_edges:
                self.A_edges = Image.fromarray(util.invert(feature.canny(rgb2gray(np.array(self.B)), sigma=0.5)))

        # self.adjust_input_size(opt)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        self.dataset_size = len(self.A_paths)

        self.batch_shift_hue = RandomHue()

    def adjust_input_size(self, opt, A, B):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        A = A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            B = B.resize((new_w, new_h), Image.BICUBIC)
        return A, B

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index])
        B = Image.open(self.B_paths[index]).convert('RGB')
        A, B = self.adjust_input_size(self.opt, A, B)
        params = get_params(self.opt, B.size, B)

        reference_tensor = None
        prob = random.random()

        if prob < self.opt.split_prob:
            pass
        elif prob >= self.opt.split_prob and prob < 2 * self.opt.split_prob:
            pass
        else:

            if self.opt.label_nc == 0:
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=False)
                A_img = A
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()

            else:
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:

                params_bianxing = get_params(self.opt, B.size, B)
                transform_reference_first, transform_B_first = get_transform_mode_3_palette_first(self.opt, params, params_bianxing, is_primitive=False, tps_change_edge=tps_change_edge)

                temp_B = B
                temp_B = self.batch_shift_hue(temp_B)

                B_tensor = transform_B_first(temp_B)
                # tps_reference_tensor = transform_reference_first(temp_B)
                tps_reference_tensor = None

                # reference_tensor = transform_reference_first(B)

                # Online extract colors  Slow
                # temp_B_numpy_array = np.asarray(B_tensor)
                # color_theme = extract_color_theme_MMCQ_1(temp_B_numpy_array)

                # Read from json
                img_last_name = self.B_paths[index].split("/")[-1]
                color_theme = np.array(self.color_theme_dict[img_last_name])

                transform_reference_second, transform_B_second = get_transform_mode_3_palette_second()
                B_tensor = transform_B_second(B_tensor)
                # tps_reference_tensor = transform_reference_second(tps_reference_tensor)
                tps_reference_tensor = B_tensor

                # reference_tensor = transform_reference_second(reference_tensor)

                color_theme_PIL = Image.fromarray(color_theme.astype(np.uint8))
                color_theme_tensor = transform_B_second(color_theme_PIL)

                contrastive_tensor_list = []


            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'color_theme_reference': color_theme_tensor, 'tps_reference': tps_reference_tensor, 'path': self.A_paths[0]}

            return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'


# TODO : fix test loader as well with scale adjustment
class AlignedDataset_test(BaseDataset):
    def initialize(self, opt):
        print("in initialize")
        self.opt = opt
        self.root = opt.dataroot

        if opt.vid_input:
            print(os.path.join(opt.dataroot + opt.phase))
            reader = imageio.get_reader(os.path.join(opt.dataroot + opt.phase), 'ffmpeg')
            opt.phase = 'vid_frames'
            dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            if not os.path.exists(self.dir_A):
                os.mkdir(self.dir_A)
            i = 0
            for im in reader:
                print(i)
                if i == 240:
                    break
                imageio.imwrite("%s/%d.png" % (self.dir_A, i), im)
                i += 1

        ### input A (label maps)
        dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
        # Top-10
        # dir_A = "_A"
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (reference images)
        dir_B = '_B'
        # Top-10
        # dir_B = '_references'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))


        if opt.primitive == "seg_edges":
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)
        print("dataset_size", self.dataset_size)

    def adjust_input_size(self, opt, A, B):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        A = A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            B = B.resize((new_w, new_h), Image.BICUBIC)
        return A, B

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index])
        B = Image.open(self.B_paths[index]).convert('RGB')
        A, B = self.adjust_input_size(self.opt, A, B)
        params = get_params(self.opt, B.size, B)

        if self.opt.label_nc == 0:
            # transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=self.opt.primitive == "edges")
            transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=False)
            # A_img = A.convert('RGB')
            A_img = A
            A_tensor = transform_A(A_img)
            if self.opt.primitive == "seg_edges":
                # apply transforms only on the edges and then fuse it to the seg
                transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                A_edges = self.A_edges.convert('RGB')
                A_edges_tensor = transform_A_edges(A_edges)
                if self.opt.canny_color == 0:
                    A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                else:
                    A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()

        else:
            transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)

        transform_B = get_transform(self.opt, params, is_primitive=False)



        # ------------------------------------ Online Extract Color ------------------------------------------
        transform_reference_first, transform_B_first = get_transform_mode_3_palette_first_inference(self.opt, params,
                                                                                          params,
                                                                                          is_primitive=False,
                                                                                          tps_change_edge=False)

        B_tensor = transform_B_first(B)
        # reference_tensor = transform_reference_first(B)

        temp_B_numpy_array = np.asarray(B_tensor)
        color_theme = extract_color_theme_MMCQ_1(temp_B_numpy_array)
        # print(color_theme.shape)

        transform_reference_second, transform_B_second = get_transform_mode_3_palette_second()
        B_tensor = transform_B_second(B_tensor)
        # reference_tensor = transform_reference_second(reference_tensor)

        color_theme_PIL = Image.fromarray(color_theme.astype(np.uint8))
        color_theme_tensor = transform_B_second(color_theme_PIL)
        # ------------------------------------ Online Extract Color ------------------------------------------


        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))


        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'reference': B_tensor, 'path': self.A_paths[index]}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_test'


class AlignedDataset_palette(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.color_theme_dict = None

        with open(self.opt.train_palette_json, encoding="utf-8") as fp:
            self.color_theme_dict = json.load(fp)

        ### input A (label maps)
        if opt.primitive != "seg_edges":
            dir_A = "_" + opt.primitive
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            # self.A = Image.open(self.A_paths[0])
            self.A_paths = self.A_paths[:opt.max_dataset_size]

        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + "_seg")
            self.A_paths = sorted(make_dataset(self.dir_A))
            # the seg input will be saved as "A"
            self.A = Image.open(self.A_paths[0])
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.mkdir(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_edges = Image.open(self.A_paths_edges[0]) if self.A_paths_edges else None

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))
            # self.B = Image.open(self.B_paths[0]).convert('RGB')
            self.B_paths = self.B_paths[:opt.max_dataset_size]
            if opt.primitive == "seg_edges" and not self.A_edges:
                self.A_edges = Image.fromarray(util.invert(feature.canny(rgb2gray(np.array(self.B)), sigma=0.5)))

        # self.adjust_input_size(opt)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        self.dataset_size = len(self.A_paths)

        self.batch_shift_hue = RandomHue()

    def adjust_input_size(self, opt, A, B):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        A = A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            B = B.resize((new_w, new_h), Image.BICUBIC)
        return A, B

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index])
        B = Image.open(self.B_paths[index]).convert('RGB')
        A, B = self.adjust_input_size(self.opt, A, B)
        params = get_params(self.opt, B.size, B)

        reference_tensor = None
        prob = random.random()

        if prob < self.opt.split_prob:
            pass
        elif prob >= self.opt.split_prob and prob < 2 * self.opt.split_prob:
            pass
        else:
            if self.opt.label_nc == 0:
                # transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=self.opt.primitive == "edges")
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=False)
                # A_img = A.convert('RGB')
                A_img = A
                A_tensor = transform_A(A_img)
                if self.opt.primitive == "seg_edges":
                    # apply transforms only on the edges and then fuse it to the seg
                    transform_A_edges, tps_change_edge = get_transform_new_ours(self.opt, params, is_primitive=True, is_edges=True)
                    A_edges = self.A_edges.convert('RGB')
                    A_edges_tensor = transform_A_edges(A_edges)
                    if self.opt.canny_color == 0:
                        A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                    else:
                        A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()

            else:
                transform_A, tps_change_edge = get_transform_new_ours(self.opt, params, normalize=False, is_primitive=True)
                A_tensor = transform_A(A) * 255.0

            B_tensor = inst_tensor = feat_tensor = 0
            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:

                params_bianxing = get_params(self.opt, B.size, B)
                transform_reference_first, transform_B_first = get_transform_mode_3_palette_first(self.opt, params, params_bianxing, is_primitive=False, tps_change_edge=tps_change_edge)

                temp_B = B

                B_tensor = transform_B_first(temp_B)
                # tps_reference_tensor = transform_reference_first(temp_B)
                tps_reference_tensor = None

                # reference_tensor = transform_reference_first(B)

                # temp_B_numpy_array = np.asarray(B_tensor)
                # color_theme = extract_color_theme_MMCQ_1(temp_B_numpy_array)

                # Read from json
                img_last_name = self.B_paths[index].split("/")[-1]
                color_theme = np.array(self.color_theme_dict[img_last_name])


                transform_reference_second, transform_B_second = get_transform_mode_3_palette_second()
                B_tensor = transform_B_second(B_tensor)
                # tps_reference_tensor = transform_reference_second(tps_reference_tensor)
                tps_reference_tensor = B_tensor

                # reference_tensor = transform_reference_second(reference_tensor)

                color_theme_PIL = Image.fromarray(color_theme.astype(np.uint8))
                color_theme_tensor = transform_B_second(color_theme_PIL)

                contrastive_tensor_list = []

            ### if using instance maps
            if not self.opt.no_instance:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)

                if self.opt.load_features:
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))

            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'image_contrastive': contrastive_tensor_list,
                          'feat': feat_tensor, 'color_theme_reference': color_theme_tensor, 'tps_reference': tps_reference_tensor, 'path': self.A_paths[0]}
            return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_palette'


# TODO : fix test loader as well with scale adjustment
class AlignedDataset_test_palette(BaseDataset):
    def initialize(self, opt):
        print("in initialize")
        self.opt = opt
        self.root = opt.dataroot

        if opt.vid_input:
            print(os.path.join(opt.dataroot + opt.phase))
            reader = imageio.get_reader(os.path.join(opt.dataroot + opt.phase), 'ffmpeg')
            opt.phase = 'vid_frames'
            dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            if not os.path.exists(self.dir_A):
                os.mkdir(self.dir_A)
            i = 0
            for im in reader:
                print(i)
                if i == 240:
                    break
                imageio.imwrite("%s/%d.png" % (self.dir_A, i), im)
                i += 1

        self.color_theme_dict = None

        # icon
        with open(self.opt.test_palette_json, encoding="utf-8") as fp:
            self.color_theme_dict = json.load(fp)


        ### input A (label maps)
        dir_A = "_" + opt.primitive if self.opt.primitive != "seg_edges" else '_seg'
        # Top-10
        # dir_A = "_A"
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (reference images)
        dir_B = '_B'
        # Top-10
        # dir_B = '_references'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))


        if opt.primitive == "seg_edges":
            self.dir_A_edges = os.path.join(opt.dataroot, opt.phase + "_edges")
            if not os.path.exists(self.dir_A_edges):
                os.makedirs(self.dir_A_edges)
            self.A_paths_edges = sorted(make_dataset(self.dir_A_edges))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)
        print("dataset_size", self.dataset_size)

    def adjust_input_size(self, opt, A, B):
        """
        change image size once when loading the image.
        :return:
        """
        # TODO: if we use instance map, support resize method = NEAREST
        ow, oh = A.size
        # for cuda memory capacity
        if max(ow, oh) > 1000:
            ow, oh = ow // 2, oh // 2

        if opt.resize_or_crop == 'none' or opt.resize_or_crop == "crop":
            # input size should fit architecture params
            base = float(2 ** opt.n_downsample_global)
            if opt.netG == 'local':
                base *= (2 ** opt.n_local_enhancers)
            new_h = int(round(oh / base) * base)
            new_w = int(round(ow / base) * base)

        elif 'resize' in opt.resize_or_crop:
            new_h, new_w = opt.loadSize, opt.loadSize

        elif 'scale_width' in opt.resize_or_crop:
            if ow != opt.loadSize:
                new_w = opt.loadSize
                new_h = int(opt.loadSize * oh / ow)

        A = A.resize((new_w, new_h), Image.NEAREST)
        if opt.primitive == "seg_edges":
            self.A_edges = self.A_edges.resize((new_w, new_h), Image.NEAREST)
        if self.opt.isTrain:
            B = B.resize((new_w, new_h), Image.BICUBIC)
        return A, B

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index])
        B = Image.open(self.B_paths[index]).convert('RGB')
        A, B = self.adjust_input_size(self.opt, A, B)
        params = get_params(self.opt, B.size, B)

        if self.opt.label_nc == 0:
            # transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=self.opt.primitive == "edges")
            transform_A = get_transform(self.opt, params, is_primitive=True, is_edges=False)
            # A_img = A.convert('RGB')
            A_img = A
            A_tensor = transform_A(A_img)
            if self.opt.primitive == "seg_edges":
                # apply transforms only on the edges and then fuse it to the seg
                transform_A_edges = get_transform(self.opt, params, is_primitive=True, is_edges=True)
                A_edges = self.A_edges.convert('RGB')
                A_edges_tensor = transform_A_edges(A_edges)
                if self.opt.canny_color == 0:
                    A_tensor[A_edges_tensor == A_edges_tensor.min()] = A_edges_tensor.min()
                else:
                    A_tensor[A_edges_tensor == A_edges_tensor.max()] = A_edges_tensor.max()

        else:
            transform_A = get_transform(self.opt, params, normalize=False, is_primitive=True)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)

        transform_B = get_transform(self.opt, params, is_primitive=False)
        # B_tensor = transform_B(B)
        # reference_tensor = transform_B(B)

        # transform_reference = get_transform_without_tps_aug(self.opt, params, is_primitive=False)
        # reference_tensor = transform_reference(B)

        transform_reference_first, transform_B_first = get_transform_mode_3_palette_first_inference(self.opt, params,
                                                                                          params,
                                                                                          is_primitive=False,
                                                                                          tps_change_edge=False)

        B_tensor = transform_B_first(B)

        # get theme colors from json file
        img_last_name = self.B_paths[index].split("/")[-1]
        color_theme = self.color_theme_dict[img_last_name]

        ## Diverse Generation with Random Palette
        # temp_value = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)]
        # color_theme[0][0] = temp_value

        ## Diverse Generation with a spefic Palette
        # temp_value = [[232, 169, 122], [180,  29,  19], [219, 112, 3]]  # user-defined palette
        # # temp_value = color_theme[0][:]  # original palette
        # random_value = np.random.randint(-25, 25, (3, 3))
        # temp_value += random_value
        # color_theme[0][:] = temp_value

        color_theme = np.array(color_theme)
        color_theme = color_theme.clip(0, 255)


        transform_reference_second, transform_B_second = get_transform_mode_3_palette_second()
        B_tensor = transform_B_second(B_tensor)

        color_theme_PIL = Image.fromarray(color_theme.astype(np.uint8))
        color_theme_tensor = transform_B_second(color_theme_PIL)

        # ------------------------------------ Online extract color ------------------------------------------
        # transform_reference_first, transform_B_first = get_transform_mode_3_palette_first_inference(self.opt, params,
        #                                                                                   params,
        #                                                                                   is_primitive=False,
        #                                                                                   tps_change_edge=False)
        #
        # B_tensor = transform_B_first(B)
        # # reference_tensor = transform_reference_first(B)
        #
        #
        # temp_B_numpy_array = np.asarray(B_tensor)
        # color_theme = extract_color_theme_MMCQ_1(temp_B_numpy_array)
        # # print(color_theme.shape)
        #
        # transform_reference_second, transform_B_second = get_transform_mode_3_palette_second()
        # B_tensor = transform_B_second(B_tensor)
        # # reference_tensor = transform_reference_second(reference_tensor)
        #
        # color_theme_PIL = Image.fromarray(color_theme.astype(np.uint8))
        # color_theme_tensor = transform_B_second(color_theme_PIL)
        # ------------------------------------ Online extract color ------------------------------------------

        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'feat': feat_tensor, 'reference': color_theme_tensor, 'path': self.A_paths[index], 'color_theme': color_theme}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset_test_palette'
