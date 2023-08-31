import torch
import torch.nn as nn
from utils import WData, print_rank0
import numpy as np
from functools import reduce
import torch.nn.functional as F


def get_mask_iterative(weight, model, criterion, max_length, p_ratio, args, return_score=False):
    score_list = []
    unmask_list = []
    mask_list = []
    model = model.eval()
    num_iter = args.num_iter  # !!!!!!!!!!!!!
    with torch.no_grad():
        length = weight.shape[1] * weight.shape[2] * weight.shape[3]
        weight = weight.flatten(1)
        weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
        weight = weight.cuda() if torch.cuda.is_available() else weight
        num_masked = int(p_ratio * weight.shape[0])
        for iter in range(num_iter):
            rand_indices = torch.rand(weight.shape[0], device=weight.device).argsort(dim=-1)
            masked_indices, unmasked_indices = rand_indices[:num_masked], rand_indices[num_masked:]
            w = weight[unmasked_indices, :]
            w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
            w = model.encoder(w)
            w = model.enc_to_dec(w)

            f = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
            f[:, unmasked_indices, :] = w
            f[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
            f[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
            res = model.decoder(f)
            # imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
            imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
            # print(imp.item())
            score_list.append(imp.item())
            unmask_list.append(unmasked_indices)
            mask_list.append(masked_indices)
        # print(max(score_list), min(score_list))
        unmask = unmask_list[score_list.index(min(score_list))]
        mask = mask_list[score_list.index(max(score_list))]
    return torch.LongTensor(unmask.cpu()), torch.LongTensor(mask.cpu())
    # return torch.LongTensor(mask)


# def get_mask_wreplacement(weight, model, criterion, max_length, p_ratio, args):
#     score_list = []
#     model = model.eval()
#     length = reduce(lambda x,y:x*y, weight.shape[1:])
#     weight = weight.flatten(1)
#     weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
#     weight = weight.cuda() if torch.cuda.is_available() else weight
#     num_masked = int(p_ratio * weight.shape[0])
#     with torch.no_grad():
#         for i in range(weight.shape[0]):
#             masked_indices = torch.LongTensor([i])
#             unmasked_indices = torch.LongTensor(list(set(range(weight.shape[0])) - set(masked_indices.tolist())))
#             w = weight[unmasked_indices, :]
#             w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
#             w = model.encoder[0](w)
#             w += model.pos_emd0(unmasked_indices.cuda())
#             w = model.encoder[1](w)
#             w = model.enc_to_dec(w)
#             f = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
#             f[:, unmasked_indices, :] = w
#             f[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
#             # f[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
#             res = model.decoder(f)
#             imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
#             # imp = criterion(res[:, :, :length], weight.unsqueeze(0)[:, :, :length])
#             # print(imp.item())
#             score_list.append(imp.item())
#         # print_rank0(', '.join([str(max(score_list)), str(min(score_list))]))
#     mask = np.argsort(score_list)[weight.shape[0] - num_masked:]
#     # threshold = np.mean(score_list)
#     # mask = torch.LongTensor(np.nonzero(np.array(score_list) > threshold)[0])
#     unmask = np.argsort(score_list)[:num_masked]
#     return torch.LongTensor(unmask), torch.LongTensor(mask)

def get_mask_wreplacement(weight, model, criterion, max_length, p_ratio, args, return_score=False, threshold=None):
    score_list = []
    model = model.eval()
    length = reduce(lambda x,y:x*y, weight.shape[1:])
    weight = weight.flatten(1)
    weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
    weight = weight.cuda() if torch.cuda.is_available() else weight
    num_masked = int(p_ratio * weight.shape[0])
    batch_list = []
    batch_size = 16
    mask_list = []
    unmask_list = []
    with torch.no_grad():
        for round in range(weight.shape[0] // batch_size):
            for i in range(round*batch_size, (round + 1)*batch_size):
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
            imp = F.mse_loss(res[:, masked_indices, :length], weight.unsqueeze(0).repeat(batch_size, 1, 1)[:, masked_indices, :length], reduction='none').mean(dim=(1,2))
            # imp = criterion(res[:, :, :length], weight.unsqueeze(0)[:, :, :length])
            # print(imp.item())
            score_list.extend(imp.tolist())
            batch_list.clear()
            mask_list.clear()
        # print_rank0(', '.join([str(max(score_list)), str(min(score_list))]))
    # import pdb
    # pdb.set_trace()
    score = np.array(score_list) / np.max(score_list)
    if threshold is not None:
        num_masked = (score>=threshold).sum()
        print(num_masked)
    mask = np.argsort(score_list)[weight.shape[0] - num_masked:]
    # threshold = np.mean(score_list)
    # mask = torch.LongTensor(np.nonzero(np.array(score_list) > threshold)[0])
    unmask = np.argsort(score_list)[:num_masked]
    if return_score:
        return torch.LongTensor(unmask), torch.LongTensor(mask), torch.Tensor(score_list), score
    else:
        return torch.LongTensor(unmask), torch.LongTensor(mask), score

# def get_mask_woreplacement(weight, model, criterion, max_length, p_ratio):
#     N_out, N_in, h, w = weight.shape
#     num_keep = int((1 - p_ratio) * N_out)
#     num_prune = N_out - num_keep
#     length = reduce(lambda x, y: x * y, weight.shape[1:])
#     weight = weight.flatten(1)
#     weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
#     weight = weight.cuda() if torch.cuda.is_available() else weight
#     remain_index = [i for i in range(N_out)]
#     for round in num_prune:
#         remain_weight = torch.index_select(weight, dim=0, index=torch.LongTensor(remain_index).to(weight.device))
#         score_list = []
#         for i in range(len(remain_index)):
#             masked_indices = torch.LongTensor([i])
#             unmasked_indices = torch.LongTensor([j for j in range(len(remain_index))][:i] + [j for j in range(len(remain_index))][i+1:])
#             w = remain_weight[unmasked_indices, :]
#             w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
#             w = model.encoder[0](w)
#             w += model.pos_emd0(unmasked_indices.cuda())
#             w = model.encoder[1](w)
#             w = model.enc_to_dec(w)
#             f = torch.zeros([1, remain_weight.shape[0], w.shape[2]], device=w.device)
#             f[:, unmasked_indices, :] = w
#             f[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
#             # f[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
#             res = model.decoder(f)
#             imp = criterion(res[:, masked_indices, :length], remain_weight.unsqueeze(0)[:, masked_indices, :length])
#             score_list.append(imp)
#         min_index = remain_index[torch.Tensor(score_list).argmin().item]
#         remain_index.remove(min_index)
#     mask_list = remain_index
#     unmask = list(set([i for i in range(N_out)]) - set(mask_list))
#     return torch.LongTensor(unmask), torch.LongTensor(mask_list)


def get_mask_woreplacement(weight, model, criterion, max_length, p_ratio, args, return_score=False, threshold=None):
    N_out, N_in, h, w = weight.shape
    num_keep = int((1 - p_ratio) * N_out)
    num_prune = N_out - num_keep
    length = reduce(lambda x, y: x * y, weight.shape[1:])
    weight = weight.flatten(1)
    weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
    weight = weight.cuda() if torch.cuda.is_available() else weight
    remain_index = [i for i in range(N_out)]
    score = np.zeros(N_out)
    indices = []
    for round in range(num_prune):
        remain_weight = torch.index_select(weight, dim=0, index=torch.LongTensor(remain_index).to(weight.device))
        score_list = []
        for i in range(len(remain_index)):
            masked_indices = torch.LongTensor([i])
            unmasked_indices = torch.LongTensor([j for j in range(len(remain_index))][:i] + [j for j in range(len(remain_index))][i+1:])
            w = remain_weight[unmasked_indices, :]
            w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
            w = model.encoder[0](w)
            w += model.pos_emd0(unmasked_indices.cuda())
            w = model.encoder[1](w)
            w = model.enc_to_dec(w)
            f = torch.zeros([1, remain_weight.shape[0], w.shape[2]], device=w.device)
            f[:, unmasked_indices, :] = w
            f[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
            # f[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
            res = model.decoder(f)
            imp = criterion(res[:, masked_indices, :length], remain_weight.unsqueeze(0)[:, masked_indices, :length])
            score_list.append(imp)
        # print(remain_weight.shape[0], max(score_list) - min(score_list))
        # print(torch.Tensor(score_list).std())
        min_index = remain_index[torch.Tensor(score_list).argmin().item()]
        # import pdb;pdb.set_trace()
        score[min_index] = min(score_list) / max(score_list)
        remain_index.remove(min_index)
        indices.append(min_index)

    if threshold is not None:
        num_masked = (score >= threshold).sum()
        print(num_masked)
        mask_list = indices[weight.shape[0] - num_masked:]
    else:
        mask_list = remain_index #no threshold
    unmask = list(set([i for i in range(N_out)]) - set(mask_list))
    return torch.LongTensor(unmask), torch.LongTensor(mask_list), score
