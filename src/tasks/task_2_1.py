import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


from tools import get_embeddings, euclidean_knn
from metrics import average_precision
from models import give_me_resnet
from losses import margin_loss


if __name__ == '__main__':
    data_path = '../data/'
    device = torch.device('cuda:0')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    epochs = 100
    alpha_list = [0.1, 1.0, 10.0, 100.0, 200.0, 400.0, 600.0]
    results = {}
    for alpha in alpha_list:
        results[str(alpha)] = []
    for _ in range(5):
        trainset = CIFAR10(data_path, train=True,  transform=transform,
                           target_transform=None, download=True)
        trainset, validationset = torch.utils.data.random_split(
            trainset, (int(len(trainset)*0.9),
                       len(trainset) - int(len(trainset)*0.9)))
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True,
                                 num_workers=0)
        validationloader = DataLoader(validationset, batch_size=256,
                                      shuffle=True, num_workers=0)

        model = give_me_resnet(pretrained=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters())
        for alpha in alpha_list:
            for epoch in range(epochs):
                total_loss = 0
                for x, y in trainloader:
                    optimizer.zero_grad()
                    output, embeddings = model(x.to(device))
                    loss = margin_loss(embeddings, y.to(device), alpha)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print('Epoch {} \t Loss: {}'.format(epoch, total_loss))
            train_X, train_Y = get_embeddings(model, trainloader, device)
            valid_X, valid_Y = get_embeddings(model, validationloader, device)

            pred_class = euclidean_knn(valid_X, train_X, train_Y)
            mAP = average_precision(valid_Y.reshape(-1, 1), pred_class)
            results[str(alpha)].append(mAP.item())
