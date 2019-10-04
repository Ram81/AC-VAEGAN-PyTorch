import argparse
import numpy as np
import torch
import torch.nn.functional as F

from network import VAEGAN
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


np.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VAEGAN")
    parser.add_argument("--z_size", default=128, action="store", type=int, dest="z_size")
    parser.add_argument("--recon_level", default=3, action="store", type=int, dest="recon_level")
    parser.add_argument("--batchsize", default=64, action="store", type=int, dest="batchsize")
    parser.add_argument("--num_classes", default=10, action="store", type=int, dest="num_classes")
    parser.add_argument("--model_path", default='model.pth', action="store", type=str, dest="model_path")

    args = parser.parse_args()

    z_size = args.z_size
    recon_level = args.recon_level
    batchsize = args.batchsize
    num_classes = args.num_classes
    model_path = args.model_path
    step_index = 0

    # TODO: add to argument parser
    dataset_name = 'cifar10'

    writer = SummaryWriter(comment="_CIFAR10_GAN")
    net = VAEGAN(z_size=z_size, recon_level=recon_level).cuda()

    # Load existing model
    model = torch.load(model_path)
    net.load_state_dict(model.state_dict())

    # switch to inference model
    net.eval()

    label = np.random.randint(0, num_classes, batchsize)
    one_hot_label = F.one_hot(torch.from_numpy(label), num_classes).float().cuda()

    out = net(None, one_hot_label, 50)
    out = out.data.cpu()
    out = (out + 1) / 2
    out = make_grid(out, nrow=8)
    writer.add_image("generated", out, step_index)
