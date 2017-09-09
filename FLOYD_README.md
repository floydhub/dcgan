# Replicate on FloydHub

##  CIFAR

nvidia-docker run --rm -it --ipc=host -v $(pwd):/dcgan -v /home/pirate/Downloads/cifar-10:/dcgan/cifar-10 -w /dcgan pytorch/pytorch:latest

Training:

- python main.py --dataset cifar10 --dataroot cifar-10 --outf cifar-10/result --cuda --ngpu 1 --niter 2


## LFW

nvidia-docker run --rm -it --ipc=host -v $(pwd):/dcgan -v /home/pirate/Downloads/lfw:/dcgan/lfw -w /dcgan pytorch/pytorch:latest

Training:

- python main.py --dataset lfw --dataroot lfw --outf lfw/result --cuda --ngpu 1 --niter 2

Generating

Random
- python generate.py --netG lfw/result/netG_epoch_0.pth --outf lfw/result

Provide a Vector
- python generate.py --netG lfw/result/netG_epoch_0.pth --Zvector lfw/result/fixed_noise.pth --outf lfw/result
- python generate.py --netG lfw/result/netG_epoch_25.pth --Zvector lfw/result/fixed_noise.pth --outf lfw/result
- python generate.py --netG lfw/result/netG_epoch_50.pth --Zvector lfw/result/fixed_noise.pth --outf lfw/result
- python generate.py --netG lfw/result/netG_epoch_69.pth --Zvector lfw/result/fixed_noise.pth --outf lfw/result

## FLOYDHUB Training

floyd run --gpu --env pytorch --data samit/datasets/lfw/1:lfw "python main.py --dataset lfw --dataroot /lfw --outf /output --cuda --ngpu 1 --niter 1"

## FLOYDHUB Generating

floyd run --gpu --env pytorch --data redeipirati/projects/dcgan/12/output:model "python generate.py --netG /model/netG_epoch_69.pth --outf"

## FLOYDHUB Serving

floyd run --gpu --mode serve --env pytorch --data redeipirati/projects/dcgan/12/output:model

- GET req (random zvector, parameter checkpoint)
curl -X GET -o <NAME_&_PATH_DOWNLOADED_IMG> -F "ckp=<MODEL_CHECKPOINT>" <SERVICE_ENDPOINT>
curl -X GET -o prova.png -F "ckp=netG_epoch_69.pth" https://www.floydhub.com/expose/wQURz6s7Q56HbLeSrRGNCL

- POST req (upload zvector, parameter checkpoint)
curl -X POST -o <NAME_&_PATH_DOWNLOADED_IMG> -F "file=@<ZVECTOR_SERIALIZED_PATH>" <SERVICE_ENDPOINT>
curl -X POST -o prova.png -F "file=@./lfw/result/fixed_noise.pth" https://www.floydhub.com/expose/wQURz6s7Q56HbLeSrRGNCL