""" Serving DCGAN
"""
from __future__ import print_function
import argparse, torch
from dcgan import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--netG', required=True, default='', help="path to netG (for generating images)")
parser.add_argument('--outf', default='/output', help='folder to output images')
parser.add_argument('--Zvector', help="path to Serialized Z vector")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
opt = parser.parse_args()
print(opt)

zvector = None
batchSize = 1
# Load a Z vector and Retrieve the N of samples to generate
if opt.Zvector:
    zvector = torch.load(opt.Zvector)
    batchSize = zvector.size()[0]

outf = "/output"
if opt.outf:
	outf = opt.outf

# GPU and CUDA
cuda = None
if opt.cuda:
	cuda = opt.cuda
ngpu = int(opt.ngpu)

# Generate An Image from input json or default parameters
Generator = DCGAN(netG=opt.netG, zvector=zvector, batchSize=batchSize, outf=outf, cuda=cuda, ngpu=ngpu)
Generator.build_model()
Generator.generate()
