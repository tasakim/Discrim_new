import argparse
from utils import *
import time
import models
from torch.cuda.amp import GradScaler

parser = argparse.ArgumentParser(description='Finetune')
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet56', help='Model Architecture')
parser.add_argument('--pretrained', type=str, default='./pretrained/r56_c10.pth',
                    help='Pretrained Weights of CV Models')
parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/Cifar10')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01, help='The Learning Rate.')
parser.add_argument('--lr_type', type=str, default='cos')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)

# Checkpoints
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
args.nprocs = torch.cuda.device_count()


# args.master_port = random.randint(30000, 40000)

def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += self.decouple(residual)
    out = self.relu(out)

    return out


def main():
    setup_seed(42)
    recorder = RecorderMeter(args.epochs)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    torch.backends.cudnn.benchmark = True
    model = models.__dict__[args.arch](num_classes=10)
    model.load_state_dict(torch.load(args.pretrained))
    '--------------Insert Decouple Layer--------------'
    for name, module in model.named_modules():
        if isinstance(module, models.cifar.resnet.ResNetBasicblock):
            layer = nn.Conv2d(module.conv2.out_channels, module.conv2.out_channels, 1, 1, 0, bias=False)
            layer.weight = nn.Parameter(torch.eye(module.conv2.out_channels).reshape(layer.weight.shape))
            set_module(model, name + '.decouple', layer)
            type(module).forward = forward
    '-------------------------------------------------'
    print_rank0(model)
    torch.save(model.cuda(0), 'decouple_{}.pt'.format(args.arch))
    exit()
    model.cuda(args.local_rank)
    print_rank0('--------------Test Original Model--------------')
    top1, _ = test(test_loader, model, criterion, args)
    print_rank0('--------------Original Model Acc {}%-----------'.format(top1))

    interval = 0
    test_top1_2 = 0
    for name, param in model.named_parameters():
        if 'decouple' not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    optimizer, scheduler = prepare_other(model, args)
    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        begin = time.time()
        print_rank0('\n==>>[Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [Time={:.2f}s]'.format(epoch + 1, args.epochs,
                                                                                                optimizer.state_dict()[
                                                                                                    'param_groups'][0][
                                                                                                    'lr'],
                                                                                                interval) \
                    + ' [Accuracy={:.2f}, Best : Accuracy={:.2f}]'.format(test_top1_2, recorder.max_accuracy(False)))

        train_acc1, train_los1 = train(epoch, train_loader, model, criterion, optimizer, args)

        test_top1_2, test_los_2 = test(test_loader, model, criterion, args)
        is_best = recorder.update(epoch, train_los1, train_acc1, test_los_2, test_top1_2)
        interval = time.time() - begin
        if is_best:
            if dist.get_rank() == 0:
                torch.save(model.cuda(0), 'decouple_{}.pt'.format(args.arch))


if __name__ == '__main__':
    main()
