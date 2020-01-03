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
    def __init__(self, shared_layers=None):
        super(VAE, self).__init__()

        # Endcoder fc1, fc2, decoder fc3, fc4
        # share layer f2, f3
        self.fc1 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU()
        )

        if shared_layers is None:
            self.fc2_mu = nn.Linear(100, 20)
            self.fc2_logstd = nn.Linear(100, 20)
            self.fc3 = nn.Linear(20, 100)
        else:
            self.fc2_mu, self.fc2_logstd, self.fc3 = shared_layers

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


def train_one_step(net, optimizer, batch_data):
    optimizer.zero_grad()
    mu, logstd, batch_data_reconstruct = net(batch_data)
    loss = vae_loss(batch_data, batch_data_reconstruct, mu, logstd)
    loss.backward()
    optimizer.step()
    return loss


def vae_train():

    BATCH_SIZE = 128

    torch.manual_seed(1)

    writer = SummaryWriter(comment="_single_vae")

    train_loader_1 = torch.utils.data.DataLoader(
        datasets.MNIST('../mnist_data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)

    train_loader_2 = torch.utils.data.DataLoader(
        datasets.MNIST('../mnist_data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../mnist_data', train=False, transform=transforms.ToTensor()),
        batch_size=8, shuffle=True, )

    shared_layers = [nn.Linear(100, 20), nn.Linear(100, 20), nn.Linear(20, 100)]

    vae_net_1 = VAE(shared_layers)
    vae_net_2 = VAE(shared_layers)

    optimizer_1 = optim.Adam(vae_net_1.parameters(), lr=1e-3)
    optimizer_2 = optim.Adam(vae_net_2.parameters(), lr=1e-3)

    print("begin training")
    total_step = 0
    for epoc in range(10):
        total_loss_1 = 0.
        total_loss_2 = 0.
        # train

        ite_training_1 = iter(train_loader_1)
        ite_training_2 = iter(train_loader_2)

        assert len(ite_training_1) == len(ite_training_2)

        for i in range(len(ite_training_1)):
            total_step += 1
            batch_data_1, _ = next(ite_training_1)
            batch_data_2, _ = next(ite_training_2)
            batch_data_2 = batch_data_2.flip(2)

            loss_1 = train_one_step(vae_net_1, optimizer_1, batch_data_1.view(-1, 784))
            loss_2 = train_one_step(vae_net_2, optimizer_2, batch_data_2.view(-1, 784))

            total_loss_1 += loss_1
            total_loss_2 += loss_2

            writer.add_scalar("step_loss/loss_1", loss_1/len(batch_data_1), total_step)
            writer.add_scalar("step_loss/loss_2", loss_2/len(batch_data_2), total_step)

            print(f"epoc {epoc} [{i*len(batch_data_1)}/{len(train_loader_1.dataset)} "
                  f"{i/len(ite_training_1) * 100:.0f}%]"
                  f"loss {loss_1/len(batch_data_1):.2f}"
                  f"{loss_2/len(batch_data_2)}")


        ave_loss_1 = total_loss_1/len(train_loader_1.dataset)
        ave_loss_2 = total_loss_2/len(train_loader_2.dataset)
        writer.add_scalar("epoch_loss/loss_1", ave_loss_1, epoc)
        writer.add_scalar("epoch_loss/loss_2", ave_loss_2, epoc)
        print(f"epoch {epoc+1} ave loss {ave_loss_1} {ave_loss_2}")

        iter_test = iter(test_loader)
        batch_data_1, _ = next(iter_test)
        _, _, batch_data_reconstru_1 = vae_net_1(batch_data_1.view(-1, 784))

        batch_data_2 = batch_data_1.flip(2)
        _, _, batch_data_reconstru_2 = vae_net_2(batch_data_2.view(-1, 784))

        batch_data_reconstru_1 = batch_data_reconstru_1.view(-1, 1, 28, 28)
        batch_data_reconstru_2 = batch_data_reconstru_2.view(-1, 1, 28, 28)

        test_result = torch.cat([batch_data_1, batch_data_reconstru_1, batch_data_2, batch_data_reconstru_2])

        writer.add_images("reconstruct", test_result, epoc)
        writer.flush()

        sample = torch.randn(64, 20)
        result_1 = vae_net_1.decode(sample)
        result_2 = vae_net_2.decode(sample)

        result_1 = result_1.view(-1, 1, 28, 28)
        result_2 = result_2.view(-1, 1, 28, 28)

        writer.add_images("sample/1", result_1, epoc)
        writer.add_images("sample_2_original", result_2, epoc)
        writer.add_images("sample/2_flip", result_2.flip(2), epoc)

        # try:
        #     batch_data_2, _ = next(ite_training_2)
        # except StopIteration:
        #     pass
    return


if __name__ == "__main__":
    vae_train()
    pass


