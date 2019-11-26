import torch


def get_embeddings(model, loader, device):
    X = torch.tensor([])
    Y = torch.tensor([], dtype=torch.int64)
    for x, y in loader:
        X = torch.cat((X, model(x.to(device))[1].cpu().detach()))
        Y = torch.cat((Y, y))
    return X, Y


def get_distance_matrix(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0*torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def euclidean_knn(test_X, train_X, train_Y):
    dist = get_distance_matrix(test_X, train_X)
    predicted_classes = train_Y[torch.argsort(dist, dim=1)]
    return predicted_classes


if __name__ == '__main__':
    pass
