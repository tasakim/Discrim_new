import numpy as np
import torch
from prune_utils import *
from utils import *
import argparse
import models
import copy
from thop import profile, clever_format
import transformers
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/Cifar10')
# parser.add_argument('--dataset', type=str, default='cifar10', help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/ImageNet')
parser.add_argument('--dataset', type=str, default='imagenet', help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', type=str, default='deit-tiny', help='Model Architecture')
parser.add_argument('--pretrained', type=str, default='./pretrained/deit_tiny',
                    help='Pretrained Weights of CV Models')
parser.add_argument('--ckpt', type=str, help='Checkpoint of Discriminator')
# --------------------------------------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=256, help='Data Batch size.')
parser.add_argument('--p_ratio', type=float, default=0.5, help='Percent of Total Pruned Filters')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--manner', type=str, default='woreplace')
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--num_iter', type=int, default=100)

args = parser.parse_args()
args.nprocs = torch.cuda.device_count()

print(args)

seed = 42
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# rate = {'head': 0.3, 'atten': 0.75, 'inter': 0.6}
rate = {'head': 0.0, 'atten': 0.0, 'inter': 0.5}

def get_score(weight, model, max_length):
    score_list = []
    batch_list = []
    batch_size = min(weight.shape[0], 256)
    mask_list = []
    unmask_list = []
    length = reduce(lambda x, y: x * y, weight.shape[1:])
    weight = weight.flatten(1)
    weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
    weight = weight.cuda() if torch.cuda.is_available() else weight
    with torch.no_grad():
        for round in range(weight.shape[0] // batch_size):
            for i in range(round * batch_size, (round + 1) * batch_size):
                masked_indices = torch.LongTensor([i])
                unmasked_indices = torch.LongTensor(list(set(range(weight.shape[0])) - set(masked_indices.tolist())))
                w = weight[unmasked_indices, :]
                w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
                batch_list.append(w)
                mask_list.append(masked_indices)
                unmask_list.append(unmasked_indices)

            w = torch.cat(batch_list, dim=0)
            masked_indices = torch.cat(mask_list, dim=0)
            w = model.encoder[0](w)
            w += model.pos_emd0(unmasked_indices.cuda())
            w = model.encoder[1](w)
            w = model.enc_to_dec(w)
            f = torch.zeros([batch_size, weight.shape[0], w.shape[2]], device=w.device)
            f[:, unmasked_indices, :] = w
            f[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
            # f[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
            res = model.decoder(f)
            # imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
            imp = F.mse_loss(res[:, masked_indices, :length],
                             weight.unsqueeze(0).repeat(batch_size, 1, 1)[:, masked_indices, :length],
                             reduction='none').mean(dim=(1, 2))
            # imp = criterion(res[:, :, :length], weight.unsqueeze(0)[:, :, :length])
            # print(imp.item())
            score_list.extend(imp.tolist())
            batch_list.clear()
            mask_list.clear()
    return score_list

def get_mask(module, model, max_dim, ratio, name_atten):
    head_mask = None
    atten_mask = None
    inter_mask = None

    model = model.eval()
    attention = getattr(module.attention, name_atten)
    query, key, value = attention.query, attention.key, attention.value
    num_head = attention.num_attention_heads
    wk = key.weight.reshape(num_head, -1, key.weight.shape[1])
    remain_head = wk.shape[0]
    '''
    prune attention head
    '''
    if rate['head'] != 0:
        num_keep = int((1 - rate['head']) * num_head)
        score_head = []
        for nh in range(num_head):
            score_head.append(np.mean(get_score(wk[nh], model, max_dim)))
        head_mask = np.argsort(score_head)[num_head - num_keep:]
        remain_head = len(head_mask)

    '''
    prune attention dim
    '''
    if rate['atten'] != 0:
        num_keep = int((1-rate['atten']) * wk.shape[1])
        score_atten = 0
        for nh in range(remain_head):
            score_atten += np.array(get_score(wk[head_mask][nh], model, max_dim))
        atten_mask = score_atten.argsort()[wk.shape[1]-num_keep:]
        atten_mask = [(atten_mask + wk.shape[1] * i).tolist() for i in range(remain_head)]
        atten_mask = sum(atten_mask, [])

    '''
    prune hidden dim
    '''
    if rate['inter'] != 0:
        score_hidden = []
        intermediate, output = module.intermediate, module.output
        inter, out = intermediate.dense, output.dense
        num_keep = int((1 - rate['inter']) * inter.weight.shape[0])
        score_list_1 = get_score(inter.weight, model, max_dim)
        score_list_2 = []
        for i in range(len(score_list_1)):
            indices = torch.ones(len(score_list_1), dtype=torch.bool)
            indices[i] = False
            score_list_2.append(np.mean(get_score(out.weight[:, indices], model, max_dim)))
        score = np.array(score_list_1) + np.array(score_list_2)
        inter_mask = score.argsort()[inter.weight.shape[0]-num_keep:]
    return head_mask, atten_mask, inter_mask

def prune_block(module, model, max_dim, ratio):
    device = module.output.dense.weight.device
    name_atten = 'attention' if type(module).__name__ == 'ViTLayer' else 'self'
    atten, intermediate, output = module.attention, module.intermediate, module.output
    head_mask, atten_mask, inter_mask = get_mask(module, model, max_dim, ratio, name_atten)

    if head_mask is not None:
        head_mask = torch.LongTensor(head_mask).to(device)
        num_heads = getattr(atten, name_atten).num_attention_heads
        getattr(atten, name_atten).num_attention_heads = len(head_mask)
        query_weight = torch.index_select(getattr(atten, name_atten).query.weight.reshape(num_heads, -1, getattr(atten, name_atten).query.weight.shape[1]), dim=0, index=head_mask).flatten(0, 1)
        getattr(atten, name_atten).query.weight = nn.Parameter(query_weight)
        query_bias = torch.index_select(getattr(atten, name_atten).query.bias.reshape(num_heads, -1), dim=0, index=head_mask).flatten()
        getattr(atten, name_atten).query.bias = nn.Parameter(query_bias)
        getattr(atten, name_atten).query.out_features = query_weight.shape[0]
        key_weight = torch.index_select(
            getattr(atten, name_atten).key.weight.reshape(num_heads, -1, getattr(atten, name_atten).key.weight.shape[1]), dim=0,
            index=head_mask).flatten(0, 1)
        getattr(atten, name_atten).key.weight = nn.Parameter(key_weight)
        key_bias = torch.index_select(getattr(atten, name_atten).key.bias.reshape(num_heads, -1), dim=0,
                                        index=head_mask).flatten()
        getattr(atten, name_atten).key.bias = nn.Parameter(key_bias)
        getattr(atten, name_atten).key.out_features = key_weight.shape[0]
        value_weight = torch.index_select(
            getattr(atten, name_atten).value.weight.reshape(num_heads, -1, getattr(atten, name_atten).value.weight.shape[1]), dim=0,
            index=head_mask).flatten(0, 1)
        getattr(atten, name_atten).value.weight = nn.Parameter(value_weight)
        value_bias = torch.index_select(getattr(atten, name_atten).value.bias.reshape(num_heads, -1), dim=0,
                                        index=head_mask).flatten()
        getattr(atten, name_atten).value.bias = nn.Parameter(value_bias)
        getattr(atten, name_atten).value.out_features = value_weight.shape[0]
        output_weight = torch.index_select(module.attention.output.dense.weight.reshape(output.dense.weight.shape[0], num_heads, -1), dim=1, index=head_mask).flatten(1)
        module.attention.output.dense.weight = nn.Parameter(output_weight)
        module.attention.output.dense.in_features = output_weight.shape[0]
        getattr(atten, name_atten).all_head_size = output_weight.shape[1]


    if atten_mask is not None:
        atten_mask = torch.LongTensor(atten_mask).to(device)
        query_weight = torch.index_select(getattr(atten, name_atten).query.weight, dim=0, index=atten_mask)
        getattr(atten, name_atten).query.weight = nn.Parameter(query_weight)
        query_bias = torch.index_select(getattr(atten, name_atten).query.bias, dim=0, index=atten_mask)
        getattr(atten, name_atten).query.bias = nn.Parameter(query_bias)
        getattr(atten, name_atten).query.out_features = query_weight.shape[0]
        key_weight = torch.index_select(getattr(atten, name_atten).key.weight, dim=0, index=atten_mask)
        getattr(atten, name_atten).key.weight = nn.Parameter(key_weight)
        key_bias = torch.index_select(getattr(atten, name_atten).key.bias, dim=0, index=atten_mask)
        getattr(atten, name_atten).key.bias = nn.Parameter(key_bias)
        getattr(atten, name_atten).key.out_features = key_weight.shape[0]
        value_weight = torch.index_select(getattr(atten, name_atten).value.weight, dim=0, index=atten_mask)
        getattr(atten, name_atten).value.weight = nn.Parameter(value_weight)
        value_bias = torch.index_select(getattr(atten, name_atten).value.bias, dim=0, index=atten_mask)
        getattr(atten, name_atten).value.bias = nn.Parameter(value_bias)
        getattr(atten, name_atten).value.out_features = value_weight.shape[0]
        output_weight = torch.index_select(module.attention.output.dense.weight, dim=1, index=atten_mask)
        module.attention.output.dense.weight = nn.Parameter(output_weight)
        module.attention.output.dense.in_features = output_weight.shape[1]
        getattr(atten, name_atten).attention_head_size = key_weight.shape[0] // getattr(atten, name_atten).num_attention_heads
        getattr(atten, name_atten).all_head_size = output_weight.shape[1]


    if inter_mask is not None:
        inter_mask = torch.LongTensor(inter_mask).to(device)
        intermediate_weight = torch.index_select(intermediate.dense.weight, dim=0, index=inter_mask)
        module.intermediate.dense.weight = nn.Parameter(intermediate_weight)
        intermediate_bias = torch.index_select(intermediate.dense.bias, dim=0, index=inter_mask)
        module.intermediate.dense.bias = nn.Parameter(intermediate_bias)
        module.intermediate.dense.out_features = intermediate_weight.shape[0]

        output_weight = torch.index_select(output.dense.weight, dim=1, index=inter_mask)
        module.output.dense.weight = nn.Parameter(output_weight)
        module.output.dense.in_features = output_weight.shape[1]

    # layernorm_before = module.layernorm_before
    # layernorm_before_weight = torch.index_select(torch.index_select(layernorm_before.weight.reshape(num_heads, -1), dim=0, index=head_mask).flatten(), dim=0, index=inter_mask)
    # module.layernorm_before.weight = nn.Parameter(layernorm_before_weight)
    # layernorm_before_bias = torch.index_select(torch.index_select(layernorm_before.bias.reshape(num_heads, -1), dim=0, index=head_mask).flatten(), dim=0,index=inter_mask)
    # module.layernorm_before.bias = nn.Parameter(layernorm_before_bias)
    # module.layernorm_before.normalized_shape = tuple(layernorm_before_bias.shape, )

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    target_model = transformers.AutoModelForImageClassification.from_pretrained(args.pretrained)
    target_model = target_model.cuda(args.local_rank)
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    # top1, _ = test(test_loader, target_model, criterion, args)
    # print_rank0('---------------Original Model Acc {}%---------------'.format(top1))
    pruned_model = copy.deepcopy(target_model)
    manner_list = {'iter': get_mask_iterative,
                   'replace': get_mask_wreplacement,
                   'woreplace': get_mask_woreplacement}
    criterion_cls = nn.CrossEntropyLoss().cuda(args.local_rank)
    layer_score_id = 0
    conv_count = 1
    mask_index = []
    l1, l2, l3, skip, max_length = get_config(args)

    ratio_list = [0, 0] + [0.0] * 70
    discrim = torch.load(args.ckpt, map_location='cpu')
    discrim = discrim.cuda()
    max_dim, max_length = discrim.max_dim, max_length
    criterion_rec = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    discrim = discrim.cuda() if torch.cuda.is_available() else discrim
    print_rank0(''.join(['-' * 20, 'Start Pruning', '-' * 20]))
    import models
    for (name, module) in pruned_model.named_modules():
        if type(module).__name__ in ['ViTLayer', 'SwinLayer']:
            print_rank0(name)
            # import pdb
            # pdb.set_trace()
            prune_block(module, discrim, max_dim, ratio=0.5)

    pruned_model.cuda(args.local_rank)
    top1, _ = test(test_loader, pruned_model, criterion_cls, args)
    print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
    pruned_model.cpu()

    if dist.get_rank() == 0:
        print(pruned_model)
        size = 32 if num_classes in [10, 100] else 224
        ori_flops, ori_macs, ori_params = get_model_profile(target_model, input_shape=(1, 3, size, size), print_profile=False,detailed=False)
        pruned_flops, pruned_macs, pruned_params = get_model_profile(pruned_model, input_shape=(1, 3, size, size), print_profile=False,detailed=False)
        print(ori_macs, ori_params)
        print(pruned_macs, pruned_params)
        torch.save(target_model, 'target_model.pt')
        torch.save(pruned_model, 'pruned_model_{}.pt'.format(ratio_list[2]))

# 唐良智 3516芯片
