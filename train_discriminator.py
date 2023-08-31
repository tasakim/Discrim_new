from copy import deepcopy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import *
import argparse
import models
import copy
from torch.utils.tensorboard import SummaryWriter
import time
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=800, help='Training Epochs')  # !!!!!!!!!!!!!
parser.add_argument('--arch', type=str, default='deit', help='Model Architecture')
parser.add_argument('--repeat', type=int, default=32,
                    help='Number of Samples within a forward when training Discrim')  # !!!!!!!!!!!!!
parser.add_argument('--pretrained', type=str, default='./pretrained/deit_base',
                    help='Pretrained Weights of CV Models')
# parser.add_argument('--pretrained', type=str, default='./pretrained/densenet_c10.pth',
#                     help='Pretrained Weights of CV Models')
parser.add_argument('--p_ratio', type=float, default=0.75, help='Percent of Total Pruned Filters')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()
args.nprocs = torch.cuda.device_count()
# print(args.nprocs)
if args.arch == 'resnet56':
    # r56
    test_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 42, 44, 46, 48, 50, 52, 54, 56]
    train_list = [2, 4, 6, 8, 10, 12, 14, 16, 18,   20, 23, 25, 27, 29, 31, 33, 35, 37,   39, 42, 44, 46, 48, 50, 52, 54, 56]
    max_dim = 576
    max_seq_len = 64
    num_classes = 10

elif args.arch == 'resnet110':
    # r56
    test_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110]
    train_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110]
    max_dim = 576
    max_seq_len = 64
    num_classes = 10

elif args.arch == 'resnet50':
    # r50
    test_list = [2, 3, 6, 7, 9, 10, 12, 13, 16, 17, 19, 20, 22, 23, 25, 26, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44,
                 45, 48, 49, 51, 52]
    train_list = [2, 3, 6, 7, 9, 10, 12, 13, 16, 17, 19, 20, 22, 23, 25, 26, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44,
                  45, 48, 49, 51, 52]
    max_dim = 4608
    max_seq_len = 2048
    num_classes = 1000

elif args.arch == 'resnet18':
    # r50
    test_list = [2, 4, 6, 9, 11, 14, 16, 19]
    train_list = [2, 4, 6, 9, 11, 14, 16, 19]
    max_dim = 4608
    max_seq_len = 512
    num_classes = 1000

elif args.arch in ['vgg16', 'vgg16bn']:
    train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    max_dim = 4608
    max_seq_len = 512
    num_classes = 10

elif args.arch == 'densenet':
    test_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    train_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    max_dim = 3888
    max_seq_len = 12
    num_classes = 1000

elif args.arch == 'deit_base':
    test_list = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71]
    train_list = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71]
    max_dim = 768
    max_seq_len = 3072
    num_classes = 1000


elif args.arch == 'deit_tiny':
    test_list = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71]
    train_list = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71]
    max_dim = 768
    max_seq_len = 3072
    num_classes = 1000

elif args.arch == 'efficientnet_b0':
    test_list = [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76]

    train_list = [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76]
    max_dim = 192
    max_seq_len = 1152
    num_classes = 1000

elif args.arch == 'efficientnet_b1':
    test_list = [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191]

    train_list = [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191]
    max_dim = 320
    max_seq_len = 1920
    num_classes = 1000

seed = 1042
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# config = {
#     'enc_dim': 768,
#     'dec_dim': 512,
#     'enc_depth': 12,
#     'dec_depth': 8,
#     'n_head': 8
# }

config = {
    'enc_dim': 128,
    'dec_dim': 128,
    'enc_depth': 4,
    'dec_depth': 4,
    'n_head': 4
}

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    print_rank0(args)
    print_rank0('Available GPU Number {}'.format(args.nprocs))
    if dist.get_rank() == 0:
        # tb_logger = SummaryWriter(log_dir="runs/{}/{}AdamW_pos_{}".format(args.arch, str(train_list),
        #                                                                   time.strftime("%Y-%m-%d %H-%M-%S",
        #                                                                                 time.localtime())))
        # tb_logger.add_text('structure_config', str(config))
        tb_logger = None
    else:
        tb_logger = None
    # target_model = models.__dict__[args.arch](num_classes=num_classes)
    # target_model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    target_model = transformers.AutoModelForImageClassification.from_pretrained(args.pretrained)
    target_model = [x for x in target_model.children()][0]
    print_rank0('-' * 20, 'Train Discriminator', '-' * 20)

    enc_dim, dec_dim, enc_depth, dec_depth, n_head = parse_structure(config)
    discrim = Discriminator(max_dim=max_dim, enc_dim=enc_dim, dec_dim=dec_dim, enc_depth=enc_depth, dec_depth=dec_depth,
                            n_head=n_head)
    discrim.pos_emd = nn.Embedding(max_seq_len, dec_dim)
    discrim.pos_emd0 = nn.Embedding(max_seq_len, enc_dim)
    print_rank0(discrim)
    # discrim = torch.load('./mae/best_imagenet_deit-b.pt')
    discrim = load_checkpoint(discrim, args.ckpt) if args.ckpt is not None else discrim #!!!!!!!!!
    discrim = discrim.cuda(args.local_rank)
    discrim = torch.nn.parallel.DistributedDataParallel(discrim, device_ids=[args.local_rank])
    trainset = WData(target_model, train_list)
    testset = WData(target_model, test_list)
    print_rank0(str(len(trainset)))
    max_length = trainset.max_length
    # trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    # testloader = DataLoader(testset, batch_size=1, shuffle=False)
    #
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)
    trainloader = DataLoader(trainset, batch_size=1, pin_memory=True, sampler=train_sampler)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, sampler=test_sampler)

    criterion_rec = nn.MSELoss().cuda(args.local_rank)
    lr = 1.5e-4 * args.repeat * torch.cuda.device_count() / 256
    optimizer_rec = optim.AdamW(filter(lambda x: x.requires_grad, discrim.parameters()), lr=lr, weight_decay=5e-2, betas=(0.9, 0.95))  # 5e-4 1e-4
    scheduler_rec = optim.lr_scheduler.CosineAnnealingLR(optimizer_rec, T_max=args.epochs, eta_min=1e-5)
    discrim = discrim.cuda() if torch.cuda.is_available() else discrim

    # target_model_new = models.__dict__['resnet18'](num_classes=1000)
    # target_model_new.load_state_dict(torch.load('./pretrained/r18_imagenet.pth'))
    # test_list_new = [2, 4, 6, 9, 11, 14, 16, 19]
    # testset_new = WData(target_model_new, test_list_new, normalize=args.normalize)
    # testloader_new = DataLoader(testset_new, batch_size=1, shuffle=False)
    tmp = 300
    set_prefix = 'c10' if args.arch in ['resnet56', 'vgg16', 'vgg16bn', 'densenet'] else 'imagenet'
    save_prefix = 'best_' + set_prefix
    epoch_flag = -10
    # test_loss = test_one_layer_ddp(testloader, discrim, criterion_rec, 0, args, tb_logger)
    # exit()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        prev_discrim = deepcopy(discrim)
        # train_one_layer(trainloader, args.p_ratio, discrim, epoch, criterion_rec, optimizer_rec, scheduler_rec, args, tb_logger)
        # test_loss = test_one_layer(testloader, discrim, criterion_rec, epoch, args, tb_logger)

        train_one_layer_ddp(trainloader, args.p_ratio, discrim, epoch, criterion_rec, optimizer_rec, scheduler_rec,
                            args, tb_logger)

        # _ = test_one_layer(testloader_new, discrim, criterion_rec, epoch, args, tb_logger=None)

        discrim = ema(prev_discrim, discrim)
        test_loss = test_one_layer_ddp(testloader, discrim, criterion_rec, epoch, args, tb_logger)

        if test_loss < tmp:
            if dist.get_rank() == 0:
                torch.save(deepcopy(discrim).module.cpu(), save_prefix + '_{}_.pt'.format(args.arch))
                # torch.save(deepcopy(discrim).module.cpu(), save_prefix + '_{}_finetune_on_resnet18.pt'.format('resnet56'))
            tmp = test_loss

        # if test_loss < tmp and dist.get_rank() == 0 and (epoch-epoch_flag)>=4 and epoch<600:
        #     print_rank0('saving epoch{}'.format(epoch))
        #     torch.save({'weight': deepcopy(discrim).module.cpu(),
        #                 'loss': test_loss},
        #                './ckpt/ckpt_{}_{}.pth'.format(args.p_ratio, epoch))
        #     tmp = test_loss
        #     epoch_flag = epoch
    # torch.save(discrim.state_dict(), './ckpt/ckpt_L_{}.pth'.format(args.p_ratio))
