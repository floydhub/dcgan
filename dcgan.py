""" Serving DCGAN
"""
# TODO: Error check
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Number of colours
NC = 3
# Latent Vector Size
NZ = 100
# Number Gen filter
NGF = 64

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
  """Generator model"""
  def __init__(self, ngpu):
    super(_netG, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(NGF * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(NGF * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(NGF * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
        nn.BatchNorm2d(NGF),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 64 x 64
    )


  def forward(self, input):
    if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
        output = self.main(input)
    return output


class DCGAN(object):
    """DCGAN - Generative Model class"""
    def __init__(self,
                 netG,
                 zvector=None,
                 batchSize=1,
                 imageSize=64,
                 nz=100,
                 ngf=64,
                 cuda=None,
                 ngpu=None,
                 outf="/output"):
      """
      DCGAN - netG Builder

      Args:
          netG: path to netG (to continue training)
          zvector: a Tensor of shape (batchsize, nz, 1, 1)
          batchSize: int, input batch size, default 64
          imageSize: int, the height / width of the input image to network,
            default 64
          nz: int, size of the latent z vector, default 100
          ngf: int, default 64
          cuda: bool, enables cuda, default False
          ngpu: int, number of GPUs to use
          outf: string, folder to output images, default output
      """
      # Path to Gen weight
      self._netG = netG
      # Latent Z Vector
      self._zvector = zvector
      # Number of sample to process
      self._batchSize = batchSize
      # Latent Z vector dim
      self._nz = int(nz)
      NZ = int(nz)
      # Number Gen Filter
      self._ngf = int(ngf)
      NGF = int(ngf)
      # Load netG
      try:
          torch.load(netG)
          self._netG = netG
          pass
      except IOError as e:
          # Does not exist OR no read permissions
          print ("Unable to open netG file")
      # Use Cuda
      self._cuda = cuda
      # How many GPU
      self._ngpu = int(ngpu)
      # Create outf if not exists
      try:
          os.makedirs(outf)
      except OSError:
          pass
      self._outf = outf


    # Build the model loading the weights
    def build_model(self):
      cudnn.benchmark = True
      # Build and load the model
      self._model = _netG(self._ngpu)
      self._model.apply(weights_init)
      self._model.load_state_dict(torch.load(self._netG))
      # If provided use Zvector else create a random input normalized
      if self._zvector is not None:
        self._input = self._zvector
      else:
        self._input = torch.FloatTensor(self._batchSize, self._nz, 1, 1).normal_(0, 1)
      # cuda?
      if self._cuda:
          self._model.cuda()
          self._input = self._input.cuda()
      self._input = Variable(self._input)


    # Generate the image and store in the output folder
    def generate(self):
      #print (self._input)
      fake = self._model(self._input)
      vutils.save_image(fake.data,
                  '%s/generated.png' % (self._outf),
                  normalize=True)
