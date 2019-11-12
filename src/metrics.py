import torch


def average_precision(Y, ordered_pred, nb_examples_per_class=5000):
    cum_match = torch.cumsum(torch.eq(Y, ordered_pred).type(torch.float64), axis=1)
    prec = cum_match/(torch.arange(1, ordered_pred.shape[1]+1))
    recall = cum_match/nb_examples_per_class
    recall_diff = recall[:, 1:] - recall[:, :-1]
    ap = torch.sum(prec[:, 1:]*recall_diff, axis=1)
    return torch.sum(ap)/Y.shape[0]


if __name__ == '__main__':
    pass