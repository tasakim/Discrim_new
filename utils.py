import random
import torch
import os
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math


class WData(Dataset):
    def __init__(self, model, mask_index, normalize=False):
        super(WData, self).__init__()
        self.model = model
        self.dataset = []
        self.mask_index = mask_index
        self.normalize = normalize
        self.max_length = self.get_max_length()
        count = 1
        for name, module in self.model.named_modules():
            # if isinstance(module, nn.Conv2d):
            #     if count in self.mask_index:
            #         item = module.weight
            #         length = item.shape[1] * item.shape[2] * item.shape[3]
            #         param = item.data.flatten(start_dim=1)
            #         param = self._normalize(param) if self.normalize else param
            #         if length < self.max_length:
            #             pad = torch.nn.ConstantPad1d((0, int(self.max_length - length)), 0)
            #             param = pad(param)
            #             # param = param
            #         else:
            #             param = param
            #         self.dataset.append([param, length])
            #     else:
            #         pass
            #     count += 1
            if isinstance(module, nn.Linear):
                if count in self.mask_index:
                    item = module.weight
                    length = item.shape[1]
                    param = item.data.flatten(start_dim=1)
                    param = self._normalize(param) if self.normalize else param
                    if length < self.max_length:
                        pad = torch.nn.ConstantPad1d((0, int(self.max_length - length)), 0)
                        param = pad(param)
                        # param = param
                    else:
                        param = param
                    self.dataset.append([param, length])
                else:
                    pass
                count += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # return self.transform(self.dataset[idx], self.mask_percent)
        return self.dataset[idx]

    @staticmethod
    def _normalize(sample):
        # return F.normalize(sample, p=2, dim=-1) #2范数归一化,对每个filter进行
        mean = torch.mean(sample)
        var = torch.var(sample).sqrt()
        return (sample - mean) / var

    def get_max_length(self):
        max_length = 0
        for name, param in self.model.named_parameters():
            if len(param.shape) == 4:
                max_length = max(max_length, param.shape[1] * param.shape[2] * param.shape[3])
        # print('max token length is {}'.format(max_length))
        return max_length


class Discriminator(nn.Module):
    def __init__(self, max_dim, enc_dim, dec_dim, enc_depth=2, dec_depth=2, n_head=2, batch_first=True):
        super(Discriminator, self).__init__()
        self.max_dim = max_dim
        self.encoder = nn.Sequential(
            nn.Linear(max_dim, enc_dim),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=enc_dim, dim_feedforward=enc_dim, nhead=n_head,
                                                             batch_first=batch_first), num_layers=enc_depth)
        )
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        self.decoder = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dec_dim, dim_feedforward=dec_dim, nhead=n_head,
                                                             batch_first=batch_first), num_layers=dec_depth),
            nn.Linear(dec_dim, max_dim)
        )
        # self.pos_emb = nn.Embedding(16, d_model)

    def forward(self, x):
        x = self.encoder(x)
        x = self.enc_to_dec(x)
        x = self.decoder(x)
        return x


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # print(rt)
    rt /= nprocs
    return rt


def prepare_dataset(args):
    if args.dataset == 'cifar10':
        # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        # std = [x / 255 for x in [63.0, 62.1, 66.7]]
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_set = datasets.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_set = datasets.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        root = args.data_path
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'val')
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        train_set = datasets.ImageFolder(traindir, train_transform)
        test_set = datasets.ImageFolder(testdir, test_transform)
        num_classes = 1000

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    args.batch_size /= args.nprocs
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(args.batch_size), shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    return train_loader, test_loader, num_classes, train_sampler, test_sampler


def prepare_other(model, args):
    if 'cifar' in args.dataset:
        # optimizer = torch.optim.SGD(param, args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
                                                         gamma=0.1, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.99, weight_decay=1e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1, last_epoch=-1)

    return optimizer, scheduler


def train(epoch, train_loader, model, criterion, optimizer, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model.train()
    num_iter = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        target = target.cuda(args.local_rank)
        input = input.cuda(args.local_rank)

        adjust_learning_rate(optimizer, epoch, i, num_iter, args)
        output = model(input)
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_prec1 = reduce_mean(prec1, args.nprocs)
        reduced_prec5 = reduce_mean(prec5, args.nprocs)

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        # grad += model.module.layer1[1].conv1.shift.grad.mean()
        # print_rank0(model.module.conv1.shift.weight.grad)
        optimizer.step()
        # if i % (len(train_loader) // 3) == 0 and i != 0:
        #     print_rank0(
        #         'Batch: [{0}/{1}] Loss: {loss.avg:.4f} Prec@1: {top1.avg:.3f} Prec@5: {top5.avg:.3f}'.format(
        #             i, len(train_loader), loss=losses, top1=top1, top5=top5))
    # print_rank0(
    #     '**Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
    #                                                                                         error1=100 - top1.avg))
    return top1.avg, losses.avg


def test(test_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    # model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.cuda(args.local_rank), target.cuda(args.local_rank)
            output = model(input)
            # output = output.logits if hasattr(output, 'logits') else output
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_prec1 = reduce_mean(prec1, args.nprocs)
            reduced_prec5 = reduce_mean(prec5, args.nprocs)

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(reduced_prec1.item(), input.size(0))
            top5.update(reduced_prec5.item(), input.size(0))

    # print_rank0('**Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
    #                                                                                          error1=100 - top1.avg))
    return top1.avg, top5.avg


def setup_seed(seed=None):
    if seed is not None:
        manualSeed = seed
    else:
        manualSeed = random.randint(1, 100)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)


class RecorderMeter():
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        #    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = round(val_acc, 2)
        self.current_epoch = idx + 1
        # x = float(int(self.max_accuracy(False) * 1000) / 1000)
        x = self.max_accuracy(False)
        y = val_acc
        # y = float(val_acc * 1000) / 1000)
        return abs(y - x) * 100 <= 1

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain:
            return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:
            return self.epoch_accuracy[:self.current_epoch, 1].max()


def print_rank0(*info):
    if dist.get_rank() == 0:
        print(*info)


def adjust_learning_rate(optimizer, epoch, step, len_iter, args):
    if args.lr_type == 'step':
        factor = epoch // 125
        # if epoch >= 80:
        #     factor = factor + 1
        lr = args.lr * (0.1 ** factor)

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.5 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))
        # lr = 0.3 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError

    # Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # if step == 0:
    #     print_rank0('learning_rate: ' + str(lr))


def parse_structure(config):
    enc_dim = config['enc_dim']
    dec_dim = config['dec_dim']
    enc_depth = config['enc_depth']
    dec_depth = config['dec_depth']
    n_head = config['n_head']
    return enc_dim, dec_dim, enc_depth, dec_depth, n_head


def get_config(args):
    if args.arch == 'resnet56':
        l1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 42, 44, 46, 48, 50, 52, 54, 56]
        # l1 = [18]
        l2 = []
        # l3 = [19]
        l3 = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30, 32, 34, 36, 38, 40, 43, 45, 47, 49, 51, 53, 55, 57]
        skip = [22, 41]
        max_seq_len = 64

    elif args.arch == 'resnet18':
        l1 = [2, 4, 6, 9, 11, 14, 16, 19]
        l2 = []
        l3 = [3, 5, 7, 10, 12, 15, 17, 20]
        skip = [8, 13, 18]
        max_seq_len = 512


    elif args.arch == 'resnet50':
        l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
        l2 = [3, 7, 10, 13, 17, 20, 23, 26, 30, 33, 36, 39, 42, 45, 49, 52]
        l3 = [4, 8, 11, 14, 18, 21, 24, 27, 31, 34, 37, 40, 43, 46, 50, 53]
        skip = [5, 15, 28, 47]
        max_seq_len = 2048

    elif args.arch in ['vgg16', 'vgg16bn']:
        l1 = [1, ]
        l2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        l3 = [13]
        skip = []
        max_seq_len = 512

    elif args.arch == 'resnet110':
        l1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 43, 45, 47, 49, 51, 53, 55,
              57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106,
              108, 110]
        l2 = []
        l3 = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 42, 44, 46, 48, 50, 52, 54, 56,
              58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107,
              109, 111]
        skip = [40, 77]
        max_seq_len = 64

    elif args.arch == 'densenet':
        l1 = [2, 15, 28]
        l2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36,
              37, 38]
        l3 = [13, 26, 39]
        skip = []
        max_seq_len = 12

    elif 'deit' in args.arch:
        l1 = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71]
        l2 = []
        l3 = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
        skip = []
        if 'base' in args.arch:
            max_seq_len = 3072
        elif 'tiny' in args.arch:
            max_seq_len = 768

    return l1, l2, l3, skip, max_seq_len


def ema(prev_discrim, curr_discrim, beta=0.99):
    prev_param = {}
    for item in prev_discrim.module.named_parameters():
        prev_param[item[0]] = item[1].data

    for item in curr_discrim.module.named_parameters():
        item[1].data = beta * prev_param[item[0]] + (1 - beta) * item[1].data

    return curr_discrim


import random
from utils import print_rank0, reduce_mean
import torch
import torch.distributed as dist


def train_one_layer(dataloader, mask_percent, model, epoch, criterion, optimizer, scheduler, args, tb_logger=None):
    model = model.train()
    total_loss = 0
    for iter, (weight, length) in enumerate(dataloader):
        # print_rank0(str(iter))
        weight = weight.repeat(args.repeat, 1, 1)

        # if random.random() > 0.5:
        #     weight = weight[torch.arange(args.repeat).unsqueeze(1), torch.rand(args.repeat, weight.shape[1]).argsort(dim=1), :]
        weight = weight.cuda() if torch.cuda.is_available() else weight
        # mask_percent = ratio * (epoch / args.epochs)
        num_masked = max(int(mask_percent * weight.shape[1]), 1)
        rand_indices = torch.rand(args.repeat, weight.shape[1], device=weight.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(args.repeat)[:, None]
        data = weight[batch_range, unmasked_indices]
        data = data.cuda() if torch.cuda.is_available() else data
        out = model.encoder[0](data)
        if model.pos_emd0 != None:
            out += model.pos_emd0(unmasked_indices)
        out = model.encoder[1](out)
        out = model.enc_to_dec(out)
        mid = torch.zeros([args.repeat, weight.shape[1], out.shape[2]])
        mid = mid.cuda() if torch.cuda.is_available() else mid
        mid[batch_range, unmasked_indices] = out
        if model.pos_emd is not None:
            mid[batch_range, masked_indices] += model.pos_emd(masked_indices)
            # mid[batch_range, unmasked_indices] += model.pos_emd(unmasked_indices)

        res = model.decoder(mid)
        # loss = criterion(res[batch_range, unmasked_indices, :length], weight[batch_range, unmasked_indices, :length])
        factor = torch.as_tensor(length / weight.shape[2], device=res.device).pow(2)
        loss = factor * criterion(res[batch_range, masked_indices, :length],
                                  weight[batch_range, masked_indices, :length])
        # loss = criterion(res[batch_range, :, :length], weight[batch_range, :, :length]) * amp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += (loss / factor).item()
        # loss = 0
        # print(epoch, iter, 'loss={:.4f}'.format(loss.item()))
        l = total_loss
        # print('epoch={}, layer={}, loss={:.4f}'.format(epoch, iter, loss.item()))
    print('Train: epoch={}, loss={:.4f}'.format(epoch, total_loss / len(dataloader)))

    if tb_logger is not None:
        tb_logger.add_scalar(tag='Loss/train', scalar_value=total_loss / len(dataloader), global_step=epoch)
        tb_logger.add_scalar(tag='Others/Mask Ratio', scalar_value=mask_percent, global_step=epoch)
        tb_logger.add_scalar(tag='Others/LR', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)
    if scheduler is not None:
        scheduler.step()
    else:
        pass
    # print(epoch, 'loss={:.4f}'.format(l / len(trainloader)))


def test_one_layer(dataloader, model, criterion, epoch, args, tb_logger=None):
    total_loss = 0
    model = model.eval()
    batch_size = args.repeat
    with torch.no_grad():
        for iter, (weight, length) in enumerate(dataloader):
            # print(weight.shape)
            weight = weight.cuda() if torch.cuda.is_available() else weight
            iter_loss = 0
            for n in range(weight.shape[1] // batch_size):
                new_weight = weight.repeat(batch_size, 1, 1)
                # if model.pos_emd0 != None:
                #     new_weight += model.pos_emd0(torch.LongTensor([i for i in range(new_weight.shape[1])]))
                # masked_indices = torch.arange(weight.shape[0], device=weight.device).unsqueeze(1)
                masked_indices = torch.tensor([i for i in range(n * batch_size, (n + 1) * batch_size)],
                                              device=new_weight.device).unsqueeze(1)
                unmasked_indices = torch.tensor(
                    [[i for i in range(n * batch_size, (n + 1) * batch_size) if i != j] for j in
                     range(n * batch_size, (n + 1) * batch_size)], device=new_weight.device)
                # unmasked_indices = torch.tensor([i for i in range(weight.shape[1]) if i != n], device=weight.device)
                masked_indices = masked_indices.long()
                unmasked_indices = unmasked_indices.long()
                batch_range = torch.arange(batch_size)[:, None]
                data = new_weight[batch_range, unmasked_indices]
                data = data.cuda() if torch.cuda.is_available() else data
                out = model.encoder[0](data)
                if model.pos_emd0 != None:
                    out += model.pos_emd0(unmasked_indices)
                out = model.encoder[1](out)
                out = model.enc_to_dec(out)
                mid = torch.zeros([batch_size, new_weight.shape[1], out.shape[2]])
                mid = mid.cuda() if torch.cuda.is_available() else mid
                # for i in range(mid.shape[0]):
                #     mid[i, index_list[i], :] = out[i]
                mid[:, unmasked_indices] = out
                if model.pos_emd is not None:
                    mid[batch_range, masked_indices] += model.pos_emd(masked_indices)
                    # mid[batch_range, unmasked_indices] += model.pos_emd(unmasked_indices)

                res = model.decoder(mid)
                # loss = criterion(res[batch_range, unmasked_indices, :length], weight[batch_range, unmasked_indices, :length])
                loss = criterion(res[batch_range, masked_indices, :length],
                                 new_weight[batch_range, masked_indices, :length])
                iter_loss += loss.item()
            total_loss += (iter_loss * batch_size)
    print('Test: epoch={}, loss={:.4f}'.format(epoch, total_loss))
    if tb_logger is not None:
        tb_logger.add_scalar(tag='Loss/test', scalar_value=total_loss, global_step=epoch)

    return total_loss


def train_one_layer_ddp(dataloader, mask_percent, model, epoch, criterion, optimizer, scheduler, args, tb_logger=None):
    model = model.train()
    # total_loss = 0
    repeat = max(1, args.repeat // args.nprocs)
    # for iteration in range(60):
    round = 320 // args.nprocs
    for iteration in range(round):
        total_loss = 0
        for iter, (weight, length) in enumerate(dataloader):
            weight = weight.repeat(repeat, 1, 1)

            # if random.random() > 0.5:
            #     weight = weight[torch.arange(args.repeat).unsqueeze(1), torch.rand(args.repeat, weight.shape[1]).argsort(dim=1), :]
            weight = weight.cuda(args.local_rank)
            # mask_percent = ratio * (epoch / args.epochs)
            num_masked = max(int(mask_percent * weight.shape[1]), 1)
            rand_indices = torch.rand(repeat, weight.shape[1], device=weight.device).argsort(dim=-1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
            batch_range = torch.arange(repeat)[:, None]
            data = weight[batch_range, unmasked_indices]
            data = data.cuda(args.local_rank, non_blocking=True)
            # print(data.device, model.device)
            out = model.module.encoder[0](data)
            if model.module.pos_emd0 != None:
                out += model.module.pos_emd0(unmasked_indices)
            out = model.module.encoder[1](out)
            out = model.module.enc_to_dec(out)
            mid = torch.zeros([repeat, weight.shape[1], out.shape[2]])
            mid = mid.cuda(args.local_rank, non_blocking=True)
            mid[batch_range, unmasked_indices] = out
            if model.module.pos_emd is not None:
                mid[batch_range, masked_indices] += model.module.pos_emd(masked_indices)
                # mid[batch_range, unmasked_indices] += model.pos_emd(unmasked_indices)

            res = model.module.decoder(mid)
            # loss = criterion(res[batch_range, unmasked_indices, :length], weight[batch_range, unmasked_indices, :length])
            factor = torch.as_tensor(length / weight.shape[2], device=res.device).pow(1)
            # factor = torch.as_tensor(weight.shape[2] / weight.shape[2], device=res.device).pow(2)

            loss = factor * criterion(res[batch_range, masked_indices, :length],
                                      weight[batch_range, masked_indices, :length])
            # loss = criterion(res[batch_range, :, :length], weight[batch_range, :, :length]) * amp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.distributed.barrier()

            total_loss += (loss / factor)
            # loss = 0
            # print(epoch, iter, 'loss={:.4f}'.format(loss.item()))
            l = total_loss
            # print('epoch={}, layer={}, loss={:.4f}'.format(epoch, iter, loss.item()))
        # print(dist.get_rank(), epoch, total_loss)

        total_loss = reduce_mean(total_loss, args.nprocs)
        # print(total_loss)
        print_rank0('Train: epoch={}, loss={:.4f}'.format(epoch, total_loss.item() / len(dataloader)))
    if tb_logger is not None and dist.get_rank() == 0:
        tb_logger.add_scalar(tag='Loss/train', scalar_value=total_loss / len(dataloader), global_step=epoch)
        tb_logger.add_scalar(tag='Others/Mask Ratio', scalar_value=mask_percent, global_step=epoch)
        tb_logger.add_scalar(tag='Others/LR', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)
    if scheduler is not None:
        scheduler.step()
    else:
        pass
    # print(epoch, 'loss={:.4f}'.format(l / len(trainloader)))


def test_one_layer_ddp(dataloader, model, criterion, epoch, args, tb_logger=None):
    total_loss = 0
    model = model.eval()
    # batch_size = args.repeat // args.nprocs
    batch_size = args.repeat
    with torch.no_grad():
        for iter, (weight, length) in enumerate(dataloader):
            # print(weight.shape)
            weight = weight.cuda() if torch.cuda.is_available() else weight
            iter_loss = 0
            for n in range(weight.shape[1] // batch_size):
                new_weight = weight.repeat(batch_size, 1, 1)
                # if model.pos_emd0 != None:
                #     new_weight += model.pos_emd0(torch.LongTensor([i for i in range(new_weight.shape[1])]))
                # masked_indices = torch.arange(weight.shape[0], device=weight.device).unsqueeze(1)
                masked_indices = torch.tensor([i for i in range(n * batch_size, (n + 1) * batch_size)],
                                              device=new_weight.device).unsqueeze(1)
                unmasked_indices = torch.tensor(
                    [[i for i in range(n * batch_size, (n + 1) * batch_size) if i != j] for j in
                     range(n * batch_size, (n + 1) * batch_size)], device=new_weight.device)
                # unmasked_indices = torch.tensor([i for i in range(weight.shape[1]) if i != n], device=weight.device)
                masked_indices = masked_indices.long()
                unmasked_indices = unmasked_indices.long()
                batch_range = torch.arange(batch_size)[:, None]
                data = new_weight[batch_range, unmasked_indices]
                data = data.cuda(args.local_rank, non_blocking=True)
                out = model.module.encoder[0](data)
                if model.module.pos_emd0 != None:
                    out += model.module.pos_emd0(unmasked_indices)
                out = model.module.encoder[1](out)
                out = model.module.enc_to_dec(out)
                mid = torch.zeros([batch_size, new_weight.shape[1], out.shape[2]])
                mid = mid.cuda(args.local_rank, non_blocking=True)
                # for i in range(mid.shape[0]):
                #     mid[i, index_list[i], :] = out[i]
                mid[:, unmasked_indices] = out
                if model.module.pos_emd is not None:
                    mid[batch_range, masked_indices] += model.module.pos_emd(masked_indices)
                    # mid[batch_range, unmasked_indices] += model.pos_emd(unmasked_indices)

                res = model.module.decoder(mid)
                # loss = criterion(res[batch_range, unmasked_indices, :length], weight[batch_range, unmasked_indices, :length])
                loss = criterion(res[batch_range, masked_indices, :length],
                                 new_weight[batch_range, masked_indices, :length])
                # torch.distributed.barrier()
                iter_loss += loss
            total_loss += (iter_loss * batch_size)
    torch.distributed.barrier()
    total_loss = reduce_mean(total_loss, args.nprocs).item()
    print_rank0('Test: epoch={}, loss={:.4f}'.format(epoch, total_loss))
    if tb_logger is not None and dist.get_rank() == 0:
        tb_logger.add_scalar(tag='Loss/test', scalar_value=total_loss, global_step=epoch)

    return total_loss


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location='cpu').state_dict()
    for item in model.state_dict().items():
        if 'layers' not in item[0]:
            checkpoint[item[0]] = item[1]

    model.load_state_dict(checkpoint)
    print_rank0('loading checkpoint complete!!')
    for name, param in model.named_parameters():
        if 'layers' in name:
            param.requires_grad = False
        else:
            pass

    return model


if __name__ == '__main__':
    model = Discriminator(max_dim=4608, enc_dim=512, dec_dim=512, enc_depth=8, dec_depth=8, n_head=2)
    print(model)
