import torch
import torch.nn.functional as F

from tools import get_distance_matrix


def get_valid_mask(labels):
    bs = len(labels)
    t1, t2, t3 = torch.meshgrid(labels, labels, labels)
    mask = torch.stack((t1.reshape(-1), t2.reshape(-1), t3.reshape(-1))).transpose(0, 1)
    mask = ((mask[:, 0] == mask[:, 1]) & (mask[:, 2] != mask[:, 1])).float()
    mask = mask.view((bs, bs, bs))
    for i in range(bs):
        mask[i:, i, :] = 0.0
    # mask = torch.tril(mask, -1)
    return mask


def get_easy_mask(loss_tensor):
    return ((loss_tensor < 0)).to(torch.float)


def get_semihard_mask(loss_tensor, alpha):
    return ((loss_tensor < 2*alpha) & (loss_tensor != 0)).to(torch.float)


def get_hard_mask(loss_tensor, alpha):
    return ((loss_tensor > 0) & (loss_tensor != 0)).to(torch.float)


def get_hardest_mask(embeddings, labels):
    pass


def margin_loss(embeddings, labels, alpha, mask_type=None):
    all_triplets = len(labels)**3
    distance_matrix = get_distance_matrix(embeddings, embeddings)
    anch_pos_dist = torch.unsqueeze(distance_matrix, 2)
    anch_neg_dist = torch.unsqueeze(distance_matrix, 1)
    triplet_loss = anch_pos_dist - anch_neg_dist + alpha
    valid_mask = get_valid_mask(labels)
    triplet_loss = triplet_loss*valid_mask
    easy_mask = get_easy_mask(triplet_loss)
    num_easy = torch.sum(easy_mask)
    triplet_loss = F.relu(triplet_loss)
    semihard_mask = get_semihard_mask(triplet_loss, alpha)
    num_semi = torch.sum(semihard_mask)
    hard_mask = get_hard_mask(triplet_loss, alpha)
    num_hard = torch.sum(hard_mask)
    if mask_type == 'semi':
        triplet_loss = (triplet_loss*semihard_mask).sum()/(num_semi + 1e-16)
    elif mask_type == 'hard':
        triplet_loss = (triplet_loss*hard_mask).sum()/(num_hard + 1e-16)
    else:
        triplet_loss = triplet_loss.sum()/((triplet_loss != 0).sum().float() + 1e-16)
    return (triplet_loss, 100*num_easy/all_triplets, 100*num_semi/all_triplets,
            100*num_hard/all_triplets)


def exponent_loss(embeddings, labels):
    distance_matrix = get_distance_matrix(embeddings, embeddings)
    anch_pos_dist = torch.exp(torch.unsqueeze(distance_matrix, 2).clamp_max(50.0))
    anch_neg_dist = torch.exp(torch.unsqueeze(distance_matrix, 1).clamp_max(50.0))
    triplet_loss = anch_pos_dist/(anch_pos_dist + anch_neg_dist)
    mask = get_valid_mask(labels)
    triplet_loss = triplet_loss*mask
    num_positive_samples = torch.sum(mask)
    loss = torch.sum(triplet_loss)/num_positive_samples
    return loss, num_positive_samples/len(labels)**3


if __name__ == '__main__':
    pass
