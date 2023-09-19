# Adapted from https://github.com/seawee1/ForkGAN-pytorch/blob/master/models/fork_gan_model.py
import torch
import itertools
from models.ForkGAN.util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import torch.cuda.amp as tamp

class ForkGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction losses')
        parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--lambda_perc', type=float, default=20.0, help='perceptual loss weight')
        parser.add_argument('--lambda_dc', type=float, default=1.0, help='domain classifier loss weight')

        parser.add_argument('--instance_level', action='store_true', help='use instance-level losses.')
        parser.add_argument('--lambda_inst', type=float, default=1.0, help='global-to-instance loss weighting ratio. Default is 1-to-1 weighting.')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'G_A_rec', 'G_A_fake_rec', 'cycle_A', 'idt_A', 'A_rec', \
                           'D_B', 'G_B', 'G_B_rec', 'G_B_fake_rec', 'cycle_B', 'idt_B', 'B_rec', \
                           'G_DC', 'Perc', 'DC']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'rec_fake_B', 'fake_A_']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'rec_fake_A', 'fake_B_']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_rec_A', 'D_rec_B', 'D_rec_fake_A', 'D_rec_fake_B', 'DC']
        else:
            self.model_names = ['G_A', 'G_B']

        if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            assert (opt.input_nc == opt.output_nc)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
            self.netD_rec_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
            self.netD_rec_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
            self.netD_rec_fake_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
            self.netD_rec_fake_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
            self.netDC = networks.DomainClassifier(opt.ngf)
            self.netDC = networks.init_net(self.netDC, init_type=opt.init_type, init_gain=opt.init_gain, gpu=opt.gpu)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.rec_A_pool = ImagePool(opt.pool_size)
            self.rec_B_pool = ImagePool(opt.pool_size)
            self.rec_A_fake_pool = ImagePool(opt.pool_size)
            self.rec_B_fake_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.gpu)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionRec = torch.nn.L1Loss()
            self.criterionCls = networks.ClsLoss().to(self.gpu)
            self.criterionPerceptual = networks.PerceptualLoss().to(self.gpu)

            self.optimizer_names = ['optimizer_G', 'optimizer_D', 'optimizer_D_rec', 'optimizer_D_rec_fake', 'optimizer_DC']

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), eps = 1e-4)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), eps = 1e-4)
            self.optimizer_D_rec = torch.optim.Adam(itertools.chain(self.netD_rec_A.parameters(), self.netD_rec_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), eps = 1e-4)
            self.optimizer_D_rec_fake = torch.optim.Adam(itertools.chain(self.netD_rec_fake_A.parameters(), self.netD_rec_fake_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), eps = 1e-4)
            self.optimizer_DC = torch.optim.Adam(self.netDC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), eps = 1e-4)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_rec)
            self.optimizers.append(self.optimizer_D_rec_fake)
            self.optimizers.append(self.optimizer_DC)

            if opt.instance_level:
                # Model names
                self.model_names += ['D_A_inst', 'D_B_inst', 'D_rec_A_inst', 'D_rec_B_inst', 'D_rec_fake_A_inst', 'D_rec_fake_B_inst']

                # Visuals
                visual_names_inst_A = ['real_A_inst', 'fake_B_inst', 'rec_A_inst', 'rec_fake_B_inst', 'fake_A__inst']
                visual_names_inst_B = ['real_B_inst', 'fake_A_inst', 'rec_B_inst', 'rec_fake_A_inst', 'fake_B__inst']
                self.visual_names_inst = visual_names_inst_A + visual_names_inst_B
                self.visual_names += self.visual_names_inst

                # Discriminators
                self.netD_A_inst = networks.define_D(opt.output_nc, opt.ndf, 'basic',
                                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
                self.netD_B_inst = networks.define_D(opt.output_nc, opt.ndf, 'basic',
                                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
                self.netD_rec_A_inst = networks.define_D(opt.output_nc, opt.ndf, 'basic',
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
                self.netD_rec_B_inst = networks.define_D(opt.input_nc, opt.ndf, 'basic',
                                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
                self.netD_rec_fake_A_inst = networks.define_D(opt.output_nc, opt.ndf, 'basic',
                                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)
                self.netD_rec_fake_B_inst = networks.define_D(opt.input_nc, opt.ndf, 'basic',
                                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu)

                # Image Pools
                self.fake_A_pool_inst = ImagePool(opt.pool_size)
                self.fake_B_pool_inst = ImagePool(opt.pool_size)
                self.rec_A_pool_inst = ImagePool(opt.pool_size)
                self.rec_B_pool_inst = ImagePool(opt.pool_size)
                self.rec_A_fake_pool_inst = ImagePool(opt.pool_size)
                self.rec_B_fake_pool_inst = ImagePool(opt.pool_size)

                # Optimizers
                self.optimizer_D_inst = torch.optim.Adam(itertools.chain(self.netD_A_inst.parameters(), self.netD_B_inst.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), eps=1e-4)
                self.optimizer_D_rec_inst = torch.optim.Adam(itertools.chain(self.netD_rec_A_inst.parameters(), self.netD_rec_B_inst.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), eps=1e-4)
                self.optimizer_D_rec_fake_inst = torch.optim.Adam(itertools.chain(self.netD_rec_fake_A_inst.parameters(), self.netD_rec_fake_B_inst.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), eps=1e-4)
                self.optimizers += [self.optimizer_D_inst, self.optimizer_D_rec_inst, self.optimizer_D_rec_fake_inst]

                # Loss names
                self.loss_names += ['D_A_inst', 'G_A_inst', 'G_A_rec_inst', 'G_A_fake_rec_inst', 'cycle_A_inst', 'A_rec_inst', \
                                    'D_B_inst', 'G_B_inst', 'G_B_rec_inst', 'G_B_fake_rec_inst', 'cycle_B_inst', 'B_rec_inst']

        if opt.distributed:
            self.netG_A = torch.nn.parallel.DistributedDataParallel(self.netG_A, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
            self.netG_B = torch.nn.parallel.DistributedDataParallel(self.netG_B, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
            if opt.isTrain:
                self.netD_A = torch.nn.parallel.DistributedDataParallel(self.netD_A, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                self.netD_B = torch.nn.parallel.DistributedDataParallel(self.netD_B, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                self.netD_rec_A = torch.nn.parallel.DistributedDataParallel(self.netD_rec_A, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                self.netD_rec_B = torch.nn.parallel.DistributedDataParallel(self.netD_rec_B, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                self.netD_rec_fake_A = torch.nn.parallel.DistributedDataParallel(self.netD_rec_fake_A, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                self.netD_rec_fake_B = torch.nn.parallel.DistributedDataParallel(self.netD_rec_fake_B, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                if opt.instance_level:
                    self.netD_A_inst = torch.nn.parallel.DistributedDataParallel(self.netD_A_inst, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                    self.netD_B_inst = torch.nn.parallel.DistributedDataParallel(self.netD_B_inst, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                    self.netD_rec_A_inst = torch.nn.parallel.DistributedDataParallel(self.netD_rec_A_inst, device_ids=[opt.gpu],output_device=opt.gpu, find_unused_parameters=True)
                    self.netD_rec_B_inst = torch.nn.parallel.DistributedDataParallel(self.netD_rec_B_inst, device_ids=[opt.gpu],output_device=opt.gpu, find_unused_parameters=True)
                    self.netD_rec_fake_A_inst = torch.nn.parallel.DistributedDataParallel(self.netD_rec_fake_A_inst, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
                    self.netD_rec_fake_B_inst = torch.nn.parallel.DistributedDataParallel(self.netD_rec_fake_B_inst, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)

        self.scaler = tamp.GradScaler()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.gpu)
        self.real_B = input['B' if AtoB else 'A'].to(self.gpu)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.instance_level and self.opt.isTrain:
            self.bboxes_A = input['A_bboxes' if AtoB else 'B_bboxes']
            self.bboxes_B = input['B_bboxes' if AtoB else 'A_bboxes']
            self.crop_size_A = self.calc_crop_size(self.bboxes_A)
            self.crop_size_B = self.calc_crop_size(self.bboxes_B)

    def forward_test(self, img):
        with tamp.autocast():
            _, _, trans_img = self.netG_B(img)  # G_B(B)
            return trans_img

    def backward_test(self, img):
        with tamp.autocast():
            _, _, trans_img = self.netG_A(img)  # G_A(A)
            return trans_img

    def forward(self):
        with tamp.autocast():
            self.enc_A, self.rec_A, self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.enc_fake_B, self.rec_fake_B, self.fake_A_ = self.netG_B(self.fake_B) # G_B(G_A(A))
            self.enc_B, self.rec_B, self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.enc_fake_A, self.rec_fake_A, self.fake_B_ = self.netG_A(self.fake_A) # G_A(G_B(B))

            if self.opt.isTrain and self.opt.instance_level and self.BBoxAvailable(self.bboxes_A) and self.BBoxAvailable(self.bboxes_B):
                # A -> B instances
                self.real_A_inst = self.crop_instances(self.real_A, self.bboxes_A, self.crop_size_A) # A
                self.rec_A_inst = self.crop_instances(self.rec_A, self.bboxes_A, self.crop_size_A) # A -> A
                self.fake_B_inst = self.crop_instances(self.fake_B, self.bboxes_A, self.crop_size_A) # A -> B
                self.rec_fake_B_inst = self.crop_instances(self.rec_fake_B, self.bboxes_A, self.crop_size_A) # A -> B -> B
                self.fake_A__inst = self.crop_instances(self.fake_A_, self.bboxes_A, self.crop_size_A) # A -> B -> A

                # B -> A instances
                self.real_B_inst = self.crop_instances(self.real_B, self.bboxes_B, self.crop_size_B)
                self.rec_B_inst = self.crop_instances(self.rec_B, self.bboxes_B, self.crop_size_B)
                self.fake_A_inst = self.crop_instances(self.fake_A, self.bboxes_B, self.crop_size_B)
                self.rec_fake_A_inst = self.crop_instances(self.rec_fake_A, self.bboxes_B, self.crop_size_B)
                self.fake_B__inst = self.crop_instances(self.fake_B_, self.bboxes_B, self.crop_size_B)

    def backward_D_basic(self, netD, real, fake):
        with tamp.autocast():
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.scaler.scale(loss_D).backward()
        return loss_D

    ##############################################################################
    # Global D backward
    ##############################################################################

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_A_rec(self):
        rec_A = self.rec_A_pool.query(self.rec_A)
        self.loss_D_rec_A = self.backward_D_basic(self.netD_rec_A, self.real_A, rec_A)

    def backward_D_B_rec(self):
        rec_B = self.rec_B_pool.query(self.rec_B)
        self.loss_D_rec_B = self.backward_D_basic(self.netD_rec_B, self.real_B, rec_B)

    def backward_D_A_fake_rec(self):
        rec_fake_A = self.rec_A_fake_pool.query(self.rec_fake_A)
        self.loss_D_A_fake_rec = self.backward_D_basic(self.netD_rec_fake_A, self.real_A, rec_fake_A)

    def backward_D_B_fake_rec(self):
        rec_fake_B = self.rec_B_fake_pool.query(self.rec_fake_B)
        self.loss_D_B_fake_rec = self.backward_D_basic(self.netD_rec_fake_B, self.real_B, rec_fake_B)

    ##############################################################################
    # Instance D backward
    ##############################################################################

    def backward_D_A_inst(self):
        fake_B_inst = self.fake_B_pool_inst.query(self.fake_B_inst)
        self.loss_D_A_inst = self.backward_D_basic(self.netD_A_inst, self.real_B_inst, fake_B_inst)

    def backward_D_B_inst(self):
        fake_A_inst = self.fake_A_pool_inst.query(self.fake_A_inst)
        self.loss_D_B_inst = self.backward_D_basic(self.netD_B_inst, self.real_A_inst, fake_A_inst)

    def backward_D_A_rec_inst(self):
        rec_A_inst = self.rec_A_pool_inst.query(self.rec_A_inst)
        self.loss_D_rec_A_inst = self.backward_D_basic(self.netD_rec_A_inst, self.real_A_inst, rec_A_inst)

    def backward_D_B_rec_inst(self):
        rec_B_inst = self.rec_B_pool_inst.query(self.rec_B_inst)
        self.loss_D_rec_B_inst = self.backward_D_basic(self.netD_rec_B_inst, self.real_B_inst, rec_B_inst)

    def backward_D_A_fake_rec_inst(self):
        rec_fake_A_inst = self.rec_A_fake_pool_inst.query(self.rec_fake_A_inst)
        self.loss_D_A_fake_rec_inst = self.backward_D_basic(self.netD_rec_fake_A_inst, self.real_A_inst, rec_fake_A_inst)

    def backward_D_B_fake_rec_inst(self):
        rec_fake_B_inst = self.rec_B_fake_pool_inst.query(self.rec_fake_B_inst)
        self.loss_D_B_fake_rec_inst = self.backward_D_basic(self.netD_rec_fake_B_inst, self.real_B_inst, rec_fake_B_inst)

    ##############################################################################
    # DC backward
    ##############################################################################

    def backward_DC(self):
        with tamp.autocast():
            # Some things are switched up here compared to the original implementation
            self.loss_DC = 0.25 * self.criterionCls(self.netDC(self.enc_A.detach()), 'A') + 0.25 * self.criterionCls(self.netDC(self.enc_B.detach()), 'B') + \
                           0.25 * self.criterionCls(self.netDC(self.enc_fake_A.detach()), 'A') + 0.25 * self.criterionCls(self.netDC(self.enc_fake_B.detach()), 'B')
        self.scaler.scale(self.loss_DC).backward()

    ##############################################################################
    # Global and Instance G backward
    ##############################################################################

    def backward_G(self):
        with tamp.autocast():
            """Calculate the loss for generators G_A and G_B"""
            lambda_idt = self.opt.lambda_identity
            lambda_rec = self.opt.lambda_rec
            lambda_A = self.opt.lambda_A
            lambda_B = self.opt.lambda_B
            lambda_perc = self.opt.lambda_perc
            lambda_dc = self.opt.lambda_dc

            # Identity loss
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                _, _, self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                _, _, self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0

            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.fake_A_, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.fake_B_, self.real_B) * lambda_B

            # Reconstruction and fake reconstruction GAN losses
            self.loss_G_A_rec = self.criterionGAN(self.netD_rec_A(self.rec_A), True)
            self.loss_G_B_rec = self.criterionGAN(self.netD_rec_B(self.rec_B), True)
            self.loss_G_A_fake_rec = self.criterionGAN(self.netD_rec_fake_A(self.rec_fake_A), True)
            self.loss_G_B_fake_rec = self.criterionGAN(self.netD_rec_fake_B(self.rec_fake_B), True)

            # Reconstruction L1 losses
            self.loss_A_rec = self.criterionRec(self.rec_A, self.real_A) * lambda_rec
            self.loss_B_rec = self.criterionRec(self.rec_B, self.real_B) * lambda_rec

            # Perceputal and Classification Losses
            self.loss_G_DC = lambda_dc * (0.25 * self.criterionCls(self.netDC(self.enc_A), 'B') + 0.25 * self.criterionCls(self.netDC(self.enc_fake_A), 'B') + \
                                          0.25 * self.criterionCls(self.netDC(self.enc_B), 'A') + 0.25 * self.criterionCls(self.netDC(self.enc_fake_B), 'A'))
            self.loss_Perc = (self.criterionPerceptual(self.enc_A, self.enc_fake_B) + self.criterionPerceptual(self.enc_B, self.enc_fake_A)) * lambda_perc

            # combined loss and calculate gradients
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                          self.loss_idt_A + self.loss_idt_B + \
                          self.loss_G_A_rec + self.loss_G_B_rec + self.loss_G_A_fake_rec + self.loss_G_B_fake_rec + \
                          self.loss_A_rec + self.loss_B_rec + \
                          self.loss_G_DC + self.loss_Perc

            if self.opt.instance_level and self.BBoxAvailable(self.bboxes_A) and self.BBoxAvailable(self.bboxes_B):
                # We drop identity loss, domain classifier and perceptual loss for instance level
                lambda_inst = self.opt.lambda_inst

                # Discriminator Losses
                self.loss_G_A_inst = self.criterionGAN(self.netD_A_inst(self.fake_B_inst), True)
                self.loss_G_B_inst = self.criterionGAN(self.netD_B_inst(self.fake_A_inst), True)
                self.loss_G_A_rec_inst = self.criterionGAN(self.netD_A_inst(self.rec_A_inst), True)
                self.loss_G_B_rec_inst = self.criterionGAN(self.netD_B_inst(self.rec_B_inst), True)
                self.loss_G_A_fake_rec_inst = self.criterionGAN(self.netD_A_inst(self.rec_fake_A_inst), True)
                self.loss_G_B_fake_rec_inst = self.criterionGAN(self.netD_B_inst(self.rec_fake_B_inst), True)
                # Cycle Loss
                self.loss_cycle_A_inst = self.criterionCycle(self.fake_A__inst, self.real_A_inst) * lambda_A
                self.loss_cycle_B_inst = self.criterionCycle(self.fake_B__inst, self.real_B_inst) * lambda_B
                # L1 reconstruction loss
                self.loss_A_rec_inst = self.criterionRec(self.rec_A_inst, self.real_A_inst) * lambda_rec
                self.loss_B_rec_inst = self.criterionRec(self.rec_B_inst, self.real_B_inst) * lambda_rec

                self.loss_G_inst = self.loss_G_A_inst + self.loss_G_B_inst + self.loss_cycle_A_inst + self.loss_cycle_B_inst + \
                                   self.loss_G_A_rec_inst + self.loss_G_B_rec_inst + self.loss_G_A_fake_rec_inst + self.loss_G_B_fake_rec_inst + \
                                   self.loss_A_rec_inst + self.loss_B_rec_inst

                self.loss_G += self.loss_G_inst * lambda_inst

        self.scaler.scale(self.loss_G).backward()

    ##############################################################################
    # Optimize function
    ##############################################################################

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_rec_A, self.netD_rec_B,
                                self.netD_rec_fake_A, self.netD_rec_fake_B, self.netDC], False)
        if self.opt.instance_level:
            self.set_requires_grad([self.netD_A_inst, self.netD_B_inst, self.netD_rec_A_inst, self.netD_rec_B_inst,
                                    self.netD_rec_fake_A_inst, self.netD_rec_fake_B_inst], False)

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.scaler.step(self.optimizer_D)
        self.scaler.update()

        # D_rec_A, D_rec_B
        self.set_requires_grad([self.netD_rec_A, self.netD_rec_B], True)
        self.optimizer_D_rec.zero_grad()
        self.backward_D_A_rec()
        self.backward_D_B_rec()
        self.scaler.step(self.optimizer_D_rec)
        self.scaler.update()

        # D_rec_fake_A, D_rec_fake_B
        self.set_requires_grad([self.netD_rec_fake_A, self.netD_rec_fake_B], True)
        self.optimizer_D_rec_fake.zero_grad()
        self.backward_D_A_fake_rec()
        self.backward_D_B_fake_rec()
        self.scaler.step(self.optimizer_D_rec_fake)
        self.scaler.update()

        # DC
        self.set_requires_grad([self.netDC], True)
        self.optimizer_DC.zero_grad()
        self.backward_DC()
        self.scaler.step(self.optimizer_DC)
        self.scaler.update()

        # Instance-level
        if self.opt.instance_level and self.BBoxAvailable(self.bboxes_A) and self.BBoxAvailable(self.bboxes_B):
            # D_A_inst and D_B_inst
            self.set_requires_grad([self.netD_A_inst, self.netD_B_inst], True)
            self.optimizer_D_inst.zero_grad()  # set D_A and D_B's gradients to zero
            self.backward_D_A_inst()  # calculate gradients for D_A
            self.backward_D_B_inst()  # calculate graidents for D_B
            self.scaler.step(self.optimizer_D_inst)
            self.scaler.update()

            # D_rec_A_inst, D_rec_B_inst
            self.set_requires_grad([self.netD_rec_A_inst, self.netD_rec_B_inst], True)
            self.optimizer_D_rec_inst.zero_grad()
            self.backward_D_A_rec_inst()
            self.backward_D_B_rec_inst()
            self.scaler.step(self.optimizer_D_rec_inst)
            self.scaler.update()

            # D_rec_fake_A_inst, D_rec_fake_B_inst
            self.set_requires_grad([self.netD_rec_fake_A_inst, self.netD_rec_fake_B_inst], True)
            self.optimizer_D_rec_fake_inst.zero_grad()
            self.backward_D_A_fake_rec_inst()
            self.backward_D_B_fake_rec_inst()
            self.scaler.step(self.optimizer_D_rec_fake_inst)
            self.scaler.update()

    ##############################################################################
    # Instance-level helper functions
    ##############################################################################

    # Calculates crop size based on image batch
    def calc_crop_size(self, bboxes_batch):
        widths = bboxes_batch[:, 2] - bboxes_batch[:, 0]
        heights = bboxes_batch[:, 3] - bboxes_batch[:, 1]
        return max(32, int(torch.max(torch.cat([widths, heights])).item()))

    # Crop first instance from each image
    def crop_instances(self, img_batch, bboxes_batch, crop_size):
        instance_crops = []
        for i in range(img_batch.shape[0]):
            # Extract coordinates, calculate w0, h0 and center
            x1 = int(bboxes_batch[i, 0].data.cpu().numpy())
            y1 = int(bboxes_batch[i, 1].data.cpu().numpy())
            x2 = int(bboxes_batch[i, 2].data.cpu().numpy())
            y2 = int(bboxes_batch[i, 3].data.cpu().numpy())
            w0 = x2 - x1
            h0 = y2 - y1
            cx = int(round(x1 + w0/2.0))
            cy = int(round(y1 + h0/2.0))

            w, h = crop_size, crop_size

            # Calculate tensor coordinates. Shift if partially outside of image
            i1 = max(0, int(round(cx - w / 2.0)))
            j1 = max(0, int(round(cy - h / 2.0)))
            i2 = i1 + w
            j2 = j1 + h
            if i2 > img_batch.shape[3]:
                shift = i2 - img_batch.shape[3]
                i1 -= shift
                i2 -= shift
            if j2 > img_batch.shape[2]:
                shift = j2 - img_batch.shape[2]
                j1 -= shift
                j2 -= shift
            instance_crops.append(img_batch[i, :, j1:j2, i1:i2].unsqueeze(0))

        return torch.cat(instance_crops, 0)

    def BBoxAvailable(self, bboxes_batch):
        # COCO dataset return [-1, -1, -1, -1] as bbox, if none is available
        if -1 in bboxes_batch:
            return False
        return True