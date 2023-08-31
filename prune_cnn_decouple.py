import torch
from prune_utils import *
from utils import *
import argparse
import models
import copy
from thop import profile, clever_format

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/Cifar10')
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose between Cifar10/100 and ImageNet.')
# parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/ImageNet')
# parser.add_argument('--dataset', type=str, default='imagenet', help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', type=str, default='resnet56', help='Model Architecture')
parser.add_argument('--pretrained', type=str, default='./decouple_resnet56.pt',
                    help='Pretrained Weights of CV Models')
parser.add_argument('--ckpt', type=str, help='Checkpoint of Discriminator')
# --------------------------------------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=256, help='Data Batch size.')
parser.add_argument('--p_ratio', type=float, default=0.5, help='Percent of Total Pruned Filters')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--manner', type=str, default='replace')
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--num_iter', type=int, default=100)

args = parser.parse_args()
args.nprocs = torch.cuda.device_count()

print(args)

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

seed = 42
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    target_model = models.__dict__[args.arch](num_classes=num_classes)
    target_model = torch.load(args.pretrained).cpu()

    pruned_model = copy.deepcopy(target_model)
    manner_list = {'iter': get_mask_iterative,
                   'replace': get_mask_wreplacement,
                   'woreplace': get_mask_woreplacement}
    criterion_cls = nn.CrossEntropyLoss().cuda(args.local_rank)
    layer_score_id = 0
    conv_count = 1
    mask_index = []
    # l1, l2, l3, skip, max_length = get_config(args)
    l1 = [2]
    conv1 = [5, 8, 11, 14, 17, 20, 23, 26, 29, 33, 36, 39, 42, 45, 48, 51, 54, 57, 61, 64, 67, 70, 73, 76, 79, 82]
    conv2 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 34, 37, 40, 43, 46, 49, 52, 55, 58, 62, 65, 68, 71, 74, 77, 80]
    decouple = [4, 7, 10, 13, 16, 19, 22, 25, 28, 32, 35, 38, 41, 44, 47, 50, 53, 56, 60, 63, 66, 69, 72, 75, 78, 81]
    l2 = conv1 + conv2
    l3 = [83, 84]
    skip = [31, 59]
    max_length = 64
    ratio_list = [0, 0] + [0.1] * 200
    discrim = torch.load(args.ckpt, map_location='cpu')
    discrim = discrim.cuda()
    max_dim, max_length = discrim.max_dim, max_length
    criterion_rec = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    discrim = discrim.cuda() if torch.cuda.is_available() else discrim
    print_rank0(''.join(['-' * 20, 'Start Pruning', '-' * 20]))
    decouple_in_indices = None
    decouple_out_indices = None
    for (name, module) in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(name)
            if conv_count in l1:
                print_rank0('---------------Pruning Layer {}---------------'.format(conv_count))
                # mask_index = [conv_count]  # r56
                ratio = 1 - ratio_list[conv_count]
                unmask, mask = manner_list[args.manner](module.weight.data, discrim, criterion_rec, max_dim, ratio,
                                                        args)
                mask_index.append(mask.cpu())
                # mask_index.append(mask.cpu())
                # print(mask_index[-1].device, module.weight.data.device)
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                bias = torch.index_select(module.bias.data, dim=0,
                                          index=torch.LongTensor(mask_index[-1])) if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                layer_score_id += 1
                conv_count += 1

            elif conv_count in l2:
                print_rank0('---------------Pruning Layer {}---------------'.format(conv_count))
                ratio = 1 - ratio_list[conv_count]
                unmask, mask = manner_list[args.manner](module.weight.data, discrim, criterion_rec, max_dim, ratio,
                                                        args)
                mask_index.append(mask.cpu())
                # import pdb;pdb.set_trace()
                decouple_in_indices = decouple_out_indices if 'conv2' in name and conv_count+1 not in skip else decouple_in_indices
                decouple_out_indices = mask.cpu() if 'conv2' in name else decouple_out_indices
                print(decouple_in_indices, decouple_out_indices)
                # mask_index.append(torch.Tensor(layer_score[layer_score_id] > np.array(sorted(layer_score[layer_score_id])[:int(len(layer_score[layer_score_id]) * ratio)][-1])).nonzero().squeeze())
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                weight = torch.index_select(weight, dim=1, index=torch.LongTensor(mask_index[-1 - 1]))
                bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(
                    mask_index[-1])) if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                layer_score_id += 1
                conv_count += 1
                # print(new_layer)

            elif conv_count in l3:
                weight = torch.index_select(module.weight.data, dim=1, index=torch.LongTensor(mask_index[-1]))
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                conv_count += 1
                layer_score_id += 1
                if args.test:
                    pruned_model.cuda(args.local_rank)
                    top1, _ = test(test_loader, pruned_model, criterion_cls, args)
                    print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
                    pruned_model.cpu()

            elif conv_count in skip:
                # import pdb;pdb.set_trace()
                weight = torch.index_select(module.weight.data, dim=1, index=torch.LongTensor(decouple_in_indices))
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                conv_count += 1
                layer_score_id += 1

            elif conv_count in decouple:
                weight = module.weight
                # import pdb;pdb.set_trace()
                print(decouple_in_indices, decouple_out_indices)
                weight = torch.index_select(weight.data, dim=1, index=torch.LongTensor(decouple_in_indices)) if decouple_in_indices is not None and conv_count-1 not in skip else weight
                weight = torch.index_select(weight.data, dim=0, index=torch.LongTensor(decouple_out_indices)) if decouple_out_indices is not None else weight
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                conv_count += 1
                layer_score_id += 1
            else:
                conv_count += 1
        elif isinstance(module, nn.BatchNorm2d):
            if conv_count - 1 in l1 + l2:
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                running_mean = torch.index_select(module.running_mean.data, dim=0,
                                                  index=torch.LongTensor(mask_index[-1]))
                running_var = torch.index_select(module.running_var.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                new_layer = nn.BatchNorm2d(num_features=weight.shape[0])
                new_layer.weight.data = nn.Parameter(weight)
                new_layer.bias.data = nn.Parameter(bias) if module.bias is not None else None
                new_layer.running_mean = running_mean
                new_layer.running_var = running_var
                set_module(pruned_model, name, new_layer)

    '--------------Insert Decouple Layer--------------'
    for name, module in pruned_model.named_modules():
        if isinstance(module, models.cifar.resnet.ResNetBasicblock):
            type(module).forward = forward
    '-------------------------------------------------'

    pruned_model.cuda(args.local_rank)
    top1, _ = test(test_loader, pruned_model, criterion_cls, args)
    print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
    pruned_model.cpu()

    if dist.get_rank() == 0:
        print(pruned_model)
        size = 32 if num_classes in [10, 100] else 224
        ori_flops, ori_params = clever_format(profile(target_model, inputs=(torch.randn(1, 3, size, size),)), '%.2f')
        pruned_flops, pruned_params = clever_format(profile(pruned_model, inputs=(torch.randn(1, 3, size, size),)),
                                                    '%.2f')
        print(ori_flops, ori_params)
        print(pruned_flops, pruned_params)
        torch.save(target_model, 'target_model.pt')
        torch.save(pruned_model, 'pruned_model_{}.pt'.format(ratio_list[2]))

# 唐良智 3516芯片
