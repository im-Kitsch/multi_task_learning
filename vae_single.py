from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Endcoder fc1, fc2, decoder fc3, fc4
        # share layer f2, f3
        self.fc1 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU()
        )
        self.fc2_mu = nn.Linear(100, 20)
        self.fc2_logstd = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Sequential(
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        return

    def forward(self, x):
        mu, logstd = self.encode(x)
        hid_z = self.reparameterize(mu, logstd)
        out = self.decode(hid_z)
        return mu, logstd, out

    def encode(self, x):
        feature = self.fc1(x)
        mu = self.fc2_mu(feature)
        logstd = self.fc2_logstd(feature)
        return mu, logstd

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(mu)
        hid_z = mu + std * eps
        return hid_z

    def decode(self, hid_z):
        out = self.fc3(hid_z)
        out = self.fc4(out)
        return out


def vae_loss(x_original, x_reconstruct, mu, logstd):
    bce_loss = F.binary_cross_entropy(x_reconstruct, x_original, reduction="sum")

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    kld_loss = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd) )
    return bce_loss + kld_loss


def vae_train():

    BATCH_SIZE = 128

    torch.manual_seed(1)

    writer = SummaryWriter(comment="_single_vae")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../mnist_data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../mnist_data', train=False, transform=transforms.ToTensor()),
    #     batch_size=8, shuffle=True, )

    vae_net = VAE()
    optimizer = optim.Adam(vae_net.parameters(), lr=1e-3)

    total_step = 0
    for epoc in range(10):
        train_loss_accu = 0.
        # train
        for batch_idx, (batch_data, _) in enumerate(train_loader):
            total_step += 1
            # batch_data = batch_data.flip(2)
            batch_data = batch_data.view(-1, 784)
            optimizer.zero_grad()
            mu, logstd, batch_data_reconstru = vae_net(batch_data)
            loss = vae_loss(batch_data, batch_data_reconstru, mu, logstd)
            loss.backward()
            optimizer.step()
            train_loss_accu += loss.detach().item()
            writer.add_scalar("loss/step_loss", loss/len(batch_data), total_step)
            if batch_idx % 10 == 0:
                print(f"epoc {epoc}, [{batch_idx *len(batch_data)}/{len(train_loader.dataset)}, "
                      f"({batch_idx/len(train_loader)*100:.0f}%)] loss {loss.item()/len(batch_data)}")
        print(f"===> Epoc {epoc}, Average_loss {train_loss_accu/len(train_loader.dataset)}")
        # test, use the last time's data
        batch_data = batch_data.view(-1, 1, 28, 28)
        batch_data_reconstru = batch_data_reconstru.view(-1, 1, 28, 28)
        test_result = torch.cat([batch_data[:8], batch_data_reconstru[:8]])

        writer.add_images("reconstruct", test_result, epoc)

    return

if __name__ == "__main__":
    vae_train()
    pass


