import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CVCModel(BaseModel):
    """ This class implements CVC model, described in the paper
    CVC: Contrastive Learning for Voice Conversion
    Tingle Li, Yichen Liu, Chenxu Hu, Hang Zhao
    Interspeech 2021

    The code borrows heavily from the PyTorch implementation of CycleGAN and CUT
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    https://github.com/taesungp/contrastive-unpaired-translation
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CVC model
        """
        parser.add_argument('--CVC_mode', type=str, default="CVC", choices='(CVC, cvc)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')

        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--netF', type=str, default='mlp_sample', help='downsample the feature map: sample | reshape | mlp_sample')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by CUT, but not CVC")

        parser.set_defaults(pool_size=0)  # no pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CVC
        if opt.CVC_mode.lower() == "cvc":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        else:
            raise ValueError(opt.CVC_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'D', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y', 'NCE_List']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = self.real_A.size(0) // len(self.opt.gpu_ids)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()                  # calculate gradients for D
            self.backward_G()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_current_model(self):
        return self.netG

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real_A = torch.flip(self.real_A, [3])
                self.real_B = torch.flip(self.real_B, [3])
        self.fake_B = self.netG(self.real_A)
        if self.opt.nce_idt:
            self.idt_B = self.netG(self.real_B)

    def backward_D(self):
        if self.opt.lambda_GAN > 0.0:
            """Calculate GAN loss for the discriminator"""
            fake = self.fake_B.detach()
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            pred_real = self.netD(self.real_B)
            loss_D_real_unweighted = self.criterionGAN(pred_real, True)
            self.loss_D_real = loss_D_real_unweighted.mean()

            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        else:
            self.loss_D_real, self.loss_D_fake, self.loss_D = 0.0, 0.0, 0.0

    def backward_G(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE, self.loss_NCE_List = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y, _ = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both

        self.loss_G.backward()

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        nce_loss_list = []
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
            nce_loss_list.append(loss.mean())

        return total_nce_loss / n_layers, nce_loss_list
