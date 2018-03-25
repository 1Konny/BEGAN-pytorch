"""solver.py"""

import time
from pathlib import Path

import visdom
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision.utils import make_grid, save_image

from utils import cuda
from models.model import Discriminator, Generator
from datasets import return_data


class BEGAN(object):
    def __init__(self, args):
        # Misc
        self.args = args
        self.cuda = args.cuda and torch.cuda.is_available()
        self.sample_num = 100

        # Optimization
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.D_lr = args.D_lr
        self.G_lr = args.G_lr
        self.gamma = args.gamma
        self.lambda_k = args.lambda_k
        self.Kt = 0.0
        self.global_epoch = 0
        self.global_iter = 0

        # Visualization
        self.visualization_init(args)

        # Network
        self.model_type = args.model_type
        self.n_filter = args.n_filter
        self.n_repeat = args.n_repeat
        self.image_size = args.image_size
        self.hidden_dim = args.hidden_dim
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.load_ckpt = args.load_ckpt
        self.model_init()

        # Dataset
        self.dataset = args.dataset
        self.data_loader = return_data(args)

        self.fixed_z = Variable(cuda(self.sample_z(self.sample_num), self.cuda))
        self.lr_step_size = len(self.data_loader['train'].dataset)//self.batch_size*self.epoch//8

    def model_init(self):
        self.D = Discriminator(self.model_type, self.image_size,
                               self.hidden_dim, self.n_filter, self.n_repeat)
        self.G = Generator(self.model_type, self.image_size,
                           self.hidden_dim, self.n_filter, self.n_repeat)

        self.D = cuda(self.D, self.cuda)
        self.G = cuda(self.G, self.cuda)

        self.D.weight_init(mean=0.0, std=0.02)
        self.G.weight_init(mean=0.0, std=0.02)

        self.D_optim = optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.G_optim = optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(0.5, 0.999))

        #self.D_optim_scheduler = lr_scheduler.ExponentialLR(self.D_optim, gamma=0.97)
        #self.G_optim_scheduler = lr_scheduler.ExponentialLR(self.G_optim, gamma=0.97)
        self.D_optim_scheduler = lr_scheduler.StepLR(self.D_optim, step_size=1, gamma=0.5)
        self.G_optim_scheduler = lr_scheduler.StepLR(self.G_optim, step_size=1, gamma=0.5)

        if self.load_ckpt:
            self.load_checkpoint()

    def visualization_init(self, args):
        self.env_name = args.env_name
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if args.visdom:
            self.visdom = args.visdom
            self.port = args.port
            self.timestep = args.timestep

            self.viz_train_curves = visdom.Visdom(env=self.env_name+'/train_curves', port=self.port)
            self.viz_train_samples = visdom.Visdom(env=self.env_name+'/train_samples', port=self.port)
            self.viz_test_samples = visdom.Visdom(env=self.env_name+'/test_samples', port=self.port)
            self.viz_interpolations = visdom.Visdom(env=self.env_name+'/interpolations', port=self.port)
            self.win_moc = None

    def sample_z(self, batch_size=0, dim=0, dist='uniform'):
        if batch_size == 0:
            batch_size = self.batch_size
        if dim == 0:
            dim = self.hidden_dim

        if dist == 'normal':
            return torch.randn(batch_size, dim)
        elif dist == 'uniform':
            return torch.rand(batch_size, dim).mul(2).add(-1)
        else:
            return None

    def sample_img(self, _type='fixed', nrow=10):
        self.set_mode('eval')

        if _type == 'fixed':
            z = self.fixed_z
        elif _type == 'random':
            z = self.sample_z(self.sample_num)
            z = Variable(cuda(z, self.cuda))
        else:
            self.set_mode('train')
            return

        samples = self.unscale(self.G(z))
        samples = samples.data.cpu()

        filename = self.output_dir.joinpath(_type+':'+str(self.global_iter)+'.jpg')
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        save_image(grid, filename=filename)
        if self.visdom:
            self.viz_test_samples.image(grid, opts=dict(title=str(filename), nrow=nrow, factor=2))

        self.set_mode('train')
        return grid

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.G.train()
            self.D.train()
        elif mode == 'eval':
            self.G.eval()
            self.D.eval()
        else:
            raise('mode error. It should be either train or eval')

    def scheduler_step(self):
        self.D_optim_scheduler.step()
        self.G_optim_scheduler.step()

    def unscale(self, tensor):
        return tensor.mul(0.5).add(0.5)

    def save_checkpoint(self, filename='ckpt.tar'):
        model_states = {'G':self.G.state_dict(),
                        'D':self.D.state_dict()}
        optim_states = {'G_optim':self.G_optim.state_dict(),
                        'D_optim':self.D_optim.state_dict()}
        states = {'iter':self.global_iter,
                  'epoch':self.global_epoch,
                  'args':self.args,
                  'win_moc':self.win_moc,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.global_iter = checkpoint['iter']
            self.global_epoch = checkpoint['epoch']
            self.win_moc = checkpoint['win_moc']
            self.G.load_state_dict(checkpoint['model_states']['G'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.G_optim.load_state_dict(checkpoint['optim_states']['G_optim'])
            self.D_optim.load_state_dict(checkpoint['optim_states']['D_optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def train(self):
        self.set_mode('train')

        for e in range(self.epoch):
            self.global_epoch += 1
            e_elapsed = time.time()

            for idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                # Discriminator Training
                x_real = Variable(cuda(images, self.cuda))
                D_real = self.D(x_real)
                D_loss_real = F.l1_loss(D_real, x_real)

                z = self.sample_z()
                z = Variable(cuda(z, self.cuda))
                x_fake = self.G(z)
                D_fake = self.D(x_fake.detach())
                D_loss_fake = F.l1_loss(D_fake, x_fake)

                D_loss = D_loss_real - self.Kt*D_loss_fake

                self.D_optim.zero_grad()
                D_loss.backward()
                self.D_optim.step()

                # Generator Training
                z = self.sample_z()
                z = Variable(cuda(z, self.cuda))
                x_fake = self.G(z)
                D_fake = self.D(x_fake)

                G_loss = F.l1_loss(x_fake, D_fake)

                self.G_optim.zero_grad()
                G_loss.backward()
                self.G_optim.step()

                # Kt update
                balance = (self.gamma*D_loss_real - D_loss_fake).data[0]
                self.Kt = max(min(self.Kt + self.lambda_k*balance, 1.0), 0.0)

                # Measure of Convergence
                M_global = (D_loss_real.data + abs(balance)).cpu()

                # Visualize process
                if self.visdom and self.global_iter%1000 == 0:
                    self.viz_train_samples.images(
                        self.unscale(x_fake).data.cpu(),
                        opts=dict(title='x_fake:{:d}'.format(self.global_iter)))
                    self.viz_train_samples.images(
                        self.unscale(D_fake).data.cpu(),
                        opts=dict(title='D_fake:{:d}'.format(self.global_iter)))
                    self.viz_train_samples.images(
                        self.unscale(x_real).data.cpu(),
                        opts=dict(title='x_real:{:d}'.format(self.global_iter)))
                    self.viz_train_samples.images(
                        self.unscale(D_real).data.cpu(),
                        opts=dict(title='D_real:{:d}'.format(self.global_iter)))

                if self.visdom and self.global_iter%self.timestep == 0:
                    X = torch.Tensor([self.global_iter])
                    if self.win_moc is None:
                        self.win_moc = self.viz_train_curves.line(
                                                    X=X,
                                                    Y=M_global,
                                                    opts=dict(
                                                        title='MOC',
                                                        fillarea=True,
                                                        xlabel='iteration',
                                                        ylabel='Measure of Convergence'))
                    else:
                        self.win_moc = self.viz_train_curves.line(
                                                    X=X,
                                                    Y=M_global,
                                                    win=self.win_moc,
                                                    update='append')

                if self.global_iter%1000 == 0:
                    print()
                    print('iter:{:d}, M:{:.3f}'.format(self.global_iter, M_global[0]))
                    print('D_loss_real:{:.3f}, D_loss_fake:{:.3f}, G_loss:{:.3f}'.format(
                        D_loss_real.data[0],
                        D_loss_fake.data[0],
                        G_loss.data[0]))

                if self.global_iter%500 == 0:
                    z1 = z[0:1]
                    z2 = z[1:2]
                    self.interpolation(z1,z2)

                if self.global_iter%self.lr_step_size == 0:
                    self.scheduler_step()


            self.save_checkpoint()
            self.sample_img('fixed')
            self.sample_img('random')
            e_elapsed = (time.time()-e_elapsed)
            print()
            print('epoch {:d}, [{:.2f}s]'.format(self.global_epoch, e_elapsed))

        print("[*] Training Finished!")


    def interpolation(self, z1, z2, n_step=10):
        filename = self.output_dir.joinpath('interpolation'+':'+str(self.global_iter)+'.jpg')

        step_size = (z2-z1)/(n_step+1)
        buff = z1
        for i in range(1, n_step+1):
            _next = z1 + step_size*(i)
            buff = torch.cat([buff, _next], dim=0)
        buff = torch.cat([buff, z2], dim=0)

        samples = self.unscale(self.G(buff))
        grid = make_grid(samples.data.cpu(), nrow=n_step+2, padding=1, pad_value=0, normalize=False)
        save_image(grid, filename=filename)
        if self.visdom:
            self.viz_interpolations.image(grid, opts=dict(title=str(filename), factor=2))




