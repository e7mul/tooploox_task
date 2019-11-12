import torch
import torch.nn.functional as F

from tools import get_distance_matrix


def get_mask(labels):
    bs = len(labels)
    t1, t2, t3 = torch.meshgrid(labels, labels, labels)
    mask = torch.stack((t1.reshape(-1), t2.reshape(-1), t3.reshape(-1))).transpose(0, 1)
    mask = ((mask[:, 0] == mask[:, 1]) & (mask[:, 2] != mask[:, 1])).float()
    mask = mask.view((bs, bs, bs))
    for i in range(bs):
        mask[i:, i, :] = 0.0
    # mask = torch.tril(mask, -1)
    return mask

def get_semihard_mask(loss_tensor, alpha):
    return ((loss_tensor < 2*alpha) & (loss_tensor != 0)).to(torch.float)

def get_hard_mask(loss_tensor, alpha):
    return ((loss_tensor > 0) & (loss_tensor != 0)).to(torch.float)


def margin_loss_mining(embeddings, labels, alpha, mask_type):
    distance_matrix = get_distance_matrix(embeddings, embeddings)
    anch_pos_dist = torch.unsqueeze(distance_matrix, 2)
    anch_neg_dist = torch.unsqueeze(distance_matrix, 1)
    triplet_loss = anch_pos_dist - anch_neg_dist + alpha
    mask = get_mask(labels)
    triplet_loss = F.relu(triplet_loss*mask)
    if mask_type == 'semihard':
        mask2 = get_semihard_mask(triplet_loss, alpha)
    elif mask_type == 'hard':
        mask2 = get_hard_mask(triplet_loss, alpha)
    triplet_loss = triplet_loss*mask2
    num_positive_samples = torch.sum(mask2)
    loss = torch.sum(triplet_loss)/num_positive_samples
    return loss, num_positive_samples/len(labels)**3


def margin_loss(embeddings, labels, alpha):
    distance_matrix = get_distance_matrix(embeddings, embeddings)
    anch_pos_dist = torch.unsqueeze(distance_matrix, 2)
    anch_neg_dist = torch.unsqueeze(distance_matrix, 1)
    triplet_loss = anch_pos_dist - anch_neg_dist + alpha
    mask = get_mask(labels)
    triplet_loss = F.relu(triplet_loss*mask)
    num_positive_samples = torch.sum(mask)
    loss = torch.sum(triplet_loss)/num_positive_samples
    return loss, num_positive_samples/len(labels)**3

def exponent_loss(embeddings, labels):
    distance_matrix = get_distance_matrix(embeddings, embeddings)
    anch_pos_dist = torch.exp(torch.unsqueeze(distance_matrix, 2).clamp_max(50.0))
    anch_neg_dist = torch.exp(torch.unsqueeze(distance_matrix, 1).clamp_max(50.0))
    triplet_loss = anch_pos_dist/(anch_pos_dist + anch_neg_dist)
    mask = get_mask(labels)
    triplet_loss = triplet_loss*mask
    num_positive_samples = torch.sum(mask)
    loss = torch.sum(triplet_loss)/num_positive_samples
    return loss, num_positive_samples/len(labels)**3


if __name__ == '__main__':
    pass
