import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


from tools import get_embeddings, euclidean_knn
from metrics import average_precision
from models import give_me_resnet
from losses import (margin_loss, exponent_loss, margin_hard_loss, 
                    margin_semihard_loss, margin_hardest_loss)

import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

if __name__ == '__main__':
    data_path = '../data/'
    results_path = '../results/'
    device = torch.device('cuda:0')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(data_path, train=True,  transform=transform,
                       target_transform=None, download=True)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True,
                             num_workers=0)
    testset = CIFAR10(data_path, train=False, transform=transform,
                      target_transform=None, download=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=True,
                            num_workers=0)

    model = give_me_resnet(pretrained=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 1.0
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0
        for x, y in trainloader:
            optimizer.zero_grad()
            output, embeddings = model(x.to(device))
            loss, _ = margin_hard_loss(embeddings, y.to(device), alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {} \t Loss: {:.3f}'.format(epoch, total_loss))

    train_X, train_Y = get_embeddings(model, trainloader, device)
    test_X, test_Y = get_embeddings(model, testloader, device)

    reduced_umap = umap.UMAP().fit_transform(test_X.numpy())
    reduced_tsne = TSNE(n_components=2).fit_transform(test_X.numpy())

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.scatter(reduced_umap[:, 0], reduced_umap[:, 1], c=test_Y)
    ax2.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], c=test_Y)
    plt.tight_layout()
    plt.savefig(results_path+'scatter.pdf', format='pdf')
    plt.close()

    pred_class = euclidean_knn(test_X, train_X, train_Y)
    mAP = average_precision(test_Y.reshape(-1, 1), pred_class)
