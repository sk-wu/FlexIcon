from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=300, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=300, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--weight_gan', type=float, default=1.0, help='weight for gan loss')
        # self.parser.add_argument('--weight_rec', type=float, default=10.0, help='weight for image-level reconstruction loss')
        self.parser.add_argument('--no_style_loss', action='store_true', help='if specified, do *not* use style loss (gram)')
        self.parser.add_argument('--no_style_contrastive_loss', type=bool, default=True, help='do *not* use style contrastive loss (gram)')
        self.parser.add_argument('--weight_style', type=float, default=10.0, help='weight for style loss (gram)')
        self.parser.add_argument('--weight_style_contrastive', type=float, default=10.0, help='weight for style loss (gram)')
        self.parser.add_argument('--no_perc_loss', type=bool, default=True, help='do *not* use perceptual loss')
        self.parser.add_argument('--weight_perc', type=float, default=0.001, help='weight for perc loss')
        self.parser.add_argument('--which_perceptual', type=str, default=-2, help='relu5_2(-1) or relu4_2(-2)')
        self.parser.add_argument('--no_ctx_loss', action='store_true', help='if specified, do *not* use contextual loss')
        self.parser.add_argument('--weight_ctx', type=float, default=10.0, help='weight for contextual loss')
        self.parser.add_argument('--use_22ctx', action='store_true', help='if true, also use 2-2 in ctx loss')
        self.parser.add_argument('--no_domain_loss', type=bool, default=True, help='do *not* use domain alignment loss')
        self.parser.add_argument('--weight_domain', type=float, default=10.0, help='weight for domain adversarial loss')
        self.parser.add_argument('--feature_mapping_ratio', type=float, default=1, help='feature mapping ratio')
        self.parser.add_argument('--weight_rec_sketch', type=float, default=10.0, help='weight for sketch reconstruction')
        self.parser.add_argument('--split_prob', type=float, default=0, help='probability for split data aug')
        self.parser.add_argument('--split_prob_1', type=float, default=0.0, help='probability for split data aug 1')
        self.parser.add_argument('--split_prob_2', type=float, default=1.0, help='probability for split data aug 2')
        self.parser.add_argument('--no_contrastive_loss',  type=bool, default=True, help='do *not* use contrastive loss')
        self.parser.add_argument('--weight_contrastive', type=float, default=10.0, help='weight for contrastive loss')
        self.parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')

        self.parser.add_argument('--train_palette_json', type=str, default=None, help='train palette json file')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.isTrain = True
