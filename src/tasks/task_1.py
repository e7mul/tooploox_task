import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


from tools import get_embeddings, euclidean_knn
from metrics import average_precision
from models import give_me_resnet


if __name__ == '__main__':
    data_path = '../data/'
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

    train_X, train_Y = get_embeddings(model, trainloader, device)
    test_X, test_Y = get_embeddings(model, testloader, device)

    pred_class = euclidean_knn(test_X, train_X, train_Y)
    mAP = average_precision(test_Y.reshape(-1, 1), pred_class)
