import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from . import losses


class ReferenceModel(BaseModel):
    def name(self):
        return 'ReferenceModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_vgg_loss_refine, use_ctx_loss, use_style_loss, use_style_contrastive_loss, use_perc_loss, use_contrastive_loss, use_domain_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, use_vgg_loss_refine, use_ctx_loss, use_style_loss, use_style_contrastive_loss, use_perc_loss, use_contrastive_loss, True, use_domain_loss, use_domain_loss, use_domain_loss, use_domain_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_vgg_refine, g_ctx, g_style, g_style_contrastive, g_perc, g_contrastive, g_rec_ske, g_gan_domain_1, g_gan_domain_2, d_gan_domain_real, d_gan_domain_fake, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, g_vgg_refine, g_ctx, g_style, g_style_contrastive, g_perc, g_contrastive, g_rec_ske, g_gan_domain_1, g_gan_domain_2, d_gan_domain_real, d_gan_domain_fake, d_real, d_fake), flags) if f]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            # self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
            #                               opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids) # 0321去掉

        # Domain Discriminator Network added by wsk
        if self.isTrain:
            if not opt.no_domain_loss:
                use_sigmoid = opt.no_lsgan
                netD_Domain_input_nc = 448
                netD_Domain_ndf = 512
                netD_Domain_num_D = 1
                self.netD_Domain = networks.define_D(netD_Domain_input_nc, netD_Domain_ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                              netD_Domain_num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                if not self.opt.no_domain_loss:
                    self.load_network(self.netD_Domain, 'D_Domain', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:

            self.vggnet_fix = networks.VGG19_feature_color_torchversion(vgg_normal_correct=True)
            self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False

            self.vggnet_fix.to(self.opt.gpu_ids[0])
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)

            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_vgg_loss, not opt.no_ctx_loss, not opt.no_style_loss, not opt.no_style_contrastive_loss, not opt.no_perc_loss, not opt.no_contrastive_loss, not opt.no_domain_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                # self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                self.criterionVGG = None

            if not opt.no_domain_loss:
                self.domain_criterionGAN = networks.DomainGANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if not opt.no_contrastive_loss:
                self.patchnceloss = losses.PatchNCELoss(self.opt)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_VGG_Refine', 'G_Ctx', 'G_Content_Consistency', 'G_Style_Contrastive', 'G_Perc', 'G_Contrastive', 'G_Rec_ske', 'G_GAN_Domain_ref', 'G_GAN_Domain_ske', 'D_GAN_Domain_ref', 'D_GAN_Domain_ske', 'D_real', 'D_fake')

            # --------------- codes about optimizer ---------------

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    # from sets import Set
                    # finetune_list = Set()
                    pass

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr*0.5, betas=(opt.beta1, 0.999))


            # added by wsk
            if opt.continue_train and opt.which_epoch == 'latest':
                checkpoint = torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'optimizer.pth'))
                self.optimizer_G.load_state_dict(checkpoint['G'])

            # optimizer D_Domain
            if not opt.no_domain_loss:
                params = list(self.netD_Domain.parameters())
                self.optimizer_D_Domain = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # --------------- codes about optimizer ---------------


    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, reference_image=None, tps_reference_image=None, image_contrastive_list=None, infer=False):
        if self.opt.label_nc == 0:
            if self.gpu_ids:
                input_label = label_map.data.cuda()
            else:
                input_label = label_map.data
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
            if self.gpu_ids:
                input_label = input_label.cuda()
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            if self.gpu_ids:
                inst_map = inst_map.data.cuda()
            else:
                inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            if self.gpu_ids:
                real_image = Variable(real_image.data.cuda())
            else:
                real_image = Variable(real_image.data)

        # reference images for training
        if reference_image is not None:
            if self.gpu_ids:
                reference_image = Variable(reference_image.data.cuda())
            else:
                reference_image = Variable(reference_image.data)

        # tps reference images for training
        if tps_reference_image is not None:
            if self.gpu_ids:
                tps_reference_image = Variable(tps_reference_image.data.cuda())
            else:
                tps_reference_image = Variable(tps_reference_image.data)

        image_contrastive = []
        if image_contrastive_list is not None:
            if self.gpu_ids:
                for i in range(len(image_contrastive_list)):
                    image_contrastive.append(Variable(image_contrastive_list[i].data.cuda()))
            else:
                for i in range(len(image_contrastive_list)):
                    image_contrastive.append(Variable(image_contrastive_list[i].data.cuda()))

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                if self.gpu_ids:
                    feat_map = Variable(feat_map.data.cuda())
                else:
                    feat_map = Variable(feat_map.data)
            if self.opt.label_feat:
                if self.gpu_ids:
                    inst_map = label_map.cuda()
                else:
                    inst_map = label_map

        return input_label, inst_map, real_image, feat_map, reference_image, tps_reference_image, image_contrastive

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-4], 4), F.avg_pool2d(target[-4].detach(), 4))) * 1
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return self.norm_2(G)

    def norm_2(self, x):
        b, h, w = x.size()
        x_view = x.view(b, -1)
        x_norm = torch.norm(x_view, dim=1, keepdim=True)  # b, 1
        x_norm = x_norm.view(b, 1, 1)
        x = x / x_norm
        return x

    def mse_loss(self, input, target=0):
        return torch.mean((input - target) ** 2)

    def forward(self, label, inst, image, feat, reference, tps_reference, image_contrastive_list, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map, reference_image, tps_reference_image, image_contrastive = self.encode_input(label, inst, image, feat, reference, tps_reference, image_contrastive_list)

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        fake_image_1, fake_image_2, contour_1, unaligned_style_image = self.netG.forward(input_concat, real_image, reference_image, tps_reference_image)

        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = F.mse_loss(fake_image_1, real_image)  # Reconstruction Error

        loss_G_Rec_ske = 0  # Encode Sketch
        if not self.opt.no_vgg_loss:
            loss_G_Rec_ske = F.mse_loss(contour_1, input_label)  # Extraction Error

        # Contextual loss
        loss_G_Ctx = 0
        if not self.opt.no_ctx_loss:
            fake_features = self.vggnet_fix(fake_image_1, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
            ref_features = self.vggnet_fix(tps_reference_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
            loss_G_Ctx = self.get_ctx_loss(fake_features, ref_features) * 0.001


        loss_G_Content_Consistency = 0
        contour_2 = None
        if not self.opt.no_style_loss:
            # Content Consistency
            self.netG.content_extractor.requires_grad_(False)
            contour_2 = self.netG.extract_content(fake_image_2)
            loss_G_Content_Consistency = F.mse_loss(contour_2, input_label)
            self.netG.content_extractor.requires_grad_(True)

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(0, 0, loss_G_VGG, 0, loss_G_Ctx, loss_G_Content_Consistency, 0, 0, 0, loss_G_Rec_ske, 0, 0, 0, 0, 0, 0),
                None if not infer else [fake_image_1, fake_image_2, contour_1, contour_2, unaligned_style_image]]

    def inference(self, label, inst, image=None):
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, feat_map, reference_image, _ = self.encode_input(Variable(label), Variable(inst), None, None, image, None, infer=True)
        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image, _, _ = self.netG.forward(reference_image, input_concat, reference_image)
        else:
            fake_image = self.netG.forward(reference_image, input_concat, reference_image)
            # _, _, _, fake_image = self.netG.forward(reference_image, input_concat)
            # _, _, _, _, _, fake_image = self.netG.forward(reference_image, input_concat)
        return fake_image

    def inference_1(self, label, inst, image=None):
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, feat_map, reference_image, _, _ = self.encode_input(Variable(label), Variable(inst), None, None, image, None, None, infer=True)
        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image, _, _ = self.netG.forward(reference_image, input_concat, reference_image)
        else:

            fake_image = self.netG.forward_inference(input_concat, reference_image)
        return fake_image


    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        if self.gpu_ids:
            image = Variable(image.cuda(), volatile=True)
        else:
            image = Variable(image, volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        if self.gpu_ids:
            feat_map = self.netE.forward(image, inst.cuda())
        else:
            feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

        if not self.opt.no_domain_loss:
            self.save_network(self.netD_Domain, "D_Domain", which_epoch, self.gpu_ids)

        # added by wsk
        if which_epoch == 'latest':
            torch.save({'G': self.optimizer_G.state_dict(),
                        'lr':  self.old_lr,
                        }, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pth'))

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if not self.opt.no_domain_loss:
            for param_group in self.optimizer_D_Domain.param_groups:
                param_group['lr'] = lr


        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(ReferenceModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)


