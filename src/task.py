import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import plotly.graph_objects as go


from tools import get_embeddings, euclidean_knn
from metrics import average_precision
from models import give_me_resnet
from losses import margin_loss, exponent_loss, margin_loss_mining


import umap
from sklearn.manifold import TSNE

from hyperdash import Experiment


if __name__ == '__main__':
    data_path = '../data/'
    device = torch.device('cuda:0')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(data_path, train=True,  transform=transform,
                       target_transform=None, download=True)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True,
                             num_workers=2)
    testset = CIFAR10(data_path, train=False, transform=transform,
                      target_transform=None, download=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=True,
                            num_workers=2)

    model = give_me_resnet(pretrained=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 1.0

    exp = Experiment('margin loss')
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for x, y in trainloader:
            optimizer.zero_grad()
            output, embeddings = model(x.to(device))
            loss, valid_fraction = margin_loss(embeddings, y.to(device), 1.0)
            # loss, valid_fraction = margin_loss_mining(embeddings, y.to(device), 1.0, 'hard')
            # loss, valid_fraction = margin_loss_mining(embeddings, y.to(device), 1.0, 'semihard')
            # loss, valid_fraction = exponent_loss(embeddings, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {} \t Loss: {}'.format(epoch, total_loss))
        exp.metric('loss', total_loss)

    train_X, train_Y = get_embeddings(model, trainloader, device)
    test_X, test_Y = get_embeddings(model, testloader, device)

    # reduced_umap = umap.UMAP().fit_transform(test_X.numpy())
    # reduced_tsne = TSNE(n_components=2).fit_transform(test_X.numpy())


    # import matplotlib.pyplot as plt
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.scatter(reduced_umap[:, 0], reduced_umap[:, 1], c=test_Y)
    # ax2.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], c=test_Y)
    # plt.savefig('../results/classifier.pdf', format='pdf')


    pred_class = euclidean_knn(test_X, train_X, train_Y)
    mAP = average_precision(test_Y.reshape(-1, 1), pred_class)
    print(mAP)
