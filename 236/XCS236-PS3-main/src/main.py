import argparse
import os
import tqdm
import numpy as np
import torch
import torchvision
from utils import *
from network import *
import torch.nn.functional as F
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Check if submission module is present.  If it is not, then main() will not be executed.
use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
    from submission.flow_network import MAF
    import submission.gan

FLOW_SEEDER = 777

def maf_train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0.0
    batch_idx = 0.0

    for data in tqdm.tqdm(train_loader):
        batch_idx += 1
        if isinstance(data, list):
            data = data[0]
        batch_size = len(data)
        data = data.view(batch_size, -1)
        data = data.to(device)

        # run MAF
        loss = model.loss(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save loss
        total_loss += loss.item()

    total_loss /= batch_idx + 1
    print("Average train log-likelihood: {:.6f}".format(-total_loss))

    return total_loss

def maf_test(model, split, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                data = data[0]
            batch_size = len(data)
            data = data.view(batch_size, -1)
            data = data.to(device)

            # run through model
            loss = model.loss(data)

            # save values
            total_loss += loss.item()

        total_loss /= batch_idx + 1
        print("Average {} log-likelihood: {:.6f}".format(split, -total_loss))

    return total_loss


def main(args):
    # NOTE: we are not supporting `torch.device("mps")` as it trains slower than cpu on Apple Silicon devices
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device: ", device)

    # create output directory
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.model == "flow":
        torch.manual_seed(FLOW_SEEDER)
        np.random.seed(FLOW_SEEDER)

        # load half-moons dataset
        train_loader, val_loader, test_loader = make_halfmoon_toy_dataset(args.n_samples, args.batch_size)

        # load model
        model = MAF(
            input_size=2, 
            hidden_size=args.hidden_size, 
            n_hidden=1, 
            n_flows=args.n_flows
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

        # logger
        best_loss = np.inf
        best_epoch = None
        train_loss_db = np.zeros(args.n_epochs + 1)
        val_loss_db = np.zeros(args.n_epochs + 1)

        # snippet of real data for plotting
        data_samples = test_loader.dataset

        for epoch in range(1, args.n_epochs + 1):
            print("Epoch {}:".format(epoch))
            train_loss = maf_train(model, optimizer, train_loader, device)
            val_loss = maf_test(model, "validation", val_loader, device)

            train_loss_db[epoch] = train_loss
            val_loss_db[epoch] = val_loss

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            if is_best:
                best_epoch = epoch
            print(
                "Best validation log-likelihood at epoch {}: {:.6f}".format(
                    best_epoch, -best_loss
                )
            )

            if epoch % 10 == 0:
                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "cmd_line_args": args,
                    },
                    is_best,
                    folder=args.out_dir,
                )

                # save samples
                samples = model.sample(device, n=1000)
                plot_samples(samples, data_samples, epoch, args)
        test_loss = maf_test(model, "test", test_loader, device)

        # save stuff
        np.save(os.path.join(args.out_dir, "train_loss.npy"), train_loss_db)
        np.save(os.path.join(args.out_dir, "val_loss.npy"), val_loss_db)
        np.save(os.path.join(args.out_dir, "test_loss.npy"), test_loss)

    elif args.model == "gan":
        dataset = torchvision.datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, drop_last=True
        )

        torchvision.utils.save_image(
            (next(iter(data_loader))[0] + 1) / 2.0, "%s/real.png" % args.out_dir
        )
        g = Generator().to(device)
        d = Discriminator().to(device)
        z_test = torch.randn(100, g.dim_z).to(device)

        g_optimizer = torch.optim.Adam(g.parameters(), 1e-3, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(d.parameters(), 1e-3, betas=(0.5, 0.999))

        global_step = 0

        # Override n_epochs to default to 1
        n_epochs = 1 if args.n_epochs == 50 else args.n_epochs

        for _ in range(n_epochs):
            for x_real, y_real in tqdm.tqdm(data_loader):
                x_real, y_real = x_real.to(device), y_real.to(device)

                if args.loss_type == "nonsaturating":
                    loss_d = submission.gan.loss_nonsaturating_d
                    loss_g = submission.gan.loss_nonsaturating_g
                elif args.loss_type == "wasserstein_gp":
                    loss_d = submission.gan.loss_wasserstein_gp_d
                    loss_g = submission.gan.loss_wasserstein_gp_g
                else:
                    raise NotImplementedError

                d_loss = loss_d(g, d, x_real, device=device)

                d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

                g_loss = loss_g(g, d, x_real, device=device)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                global_step += 1

                if global_step % 50 == 0:
                    with torch.no_grad():
                        g.eval()
                        x_test = (g(z_test) + 1) / 2.0
                        torchvision.utils.save_image(
                            x_test, "%s/fake_%04d.png" % (args.out_dir, global_step), nrow=10
                        )
                        g.train()

            with torch.no_grad():
                torch.save((g, d), "%s/model_%04d.pt" % (args.out_dir, global_step))
    elif args.model == "cgan":
        dataset = torchvision.datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, drop_last=True
        )

        torchvision.utils.save_image(
            (next(iter(data_loader))[0] + 1) / 2.0, "%s/real.png" % args.out_dir
        )

        g = ConditionalGenerator().to(device)
        d = ConditionalDiscriminator().to(device)
        z_test = torch.randn(10, 1, g.dim_z).repeat(1, 10, 1).reshape(100, g.dim_z).to(device)
        y_test = torch.arange(10).repeat(10).to(device)

        g_optimizer = torch.optim.Adam(g.parameters(), 1e-3, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(d.parameters(), 1e-3, betas=(0.5, 0.999))

        # Override n_epochs to default to 1
        n_epochs = 1 if args.n_epochs == 50 else args.n_epochs

        global_step = 0

        for _ in range(n_epochs):
            for x_real, y_real in tqdm.tqdm(data_loader):
                x_real, y_real = x_real.to(device), y_real.to(device)

                if args.loss_type == "nonsaturating":
                    loss_d = submission.gan.conditional_loss_nonsaturating_d
                    loss_g = submission.gan.conditional_loss_nonsaturating_g
                else:
                    raise NotImplementedError

                d_loss = loss_d(g, d, x_real, y_real, device=device)

                d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

                g_loss = loss_g(g, d, x_real, y_real, device=device)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                global_step += 1

                if global_step % 50 == 0:
                    with torch.no_grad():
                        g.eval()
                        x_test = (g(z_test, y_test) + 1) / 2.0
                        torchvision.utils.save_image(
                            x_test, "%s/fake_%04d.png" % (args.out_dir, global_step), nrow=10
                        )
                        g.train()

            with torch.no_grad():
                torch.save((g, d), "%s/model_%04d.pt" % (args.out_dir, global_step))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="flow", choices=['flow', 'gan', 'cgan'], help="Model to run")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'gpu'], help="GPU or CPU acceleration")
    parser.add_argument('--cache', action='store_true', help="Cache halfmoon data to avoid redownloading")

    # Flow Model Arguments
    parser.add_argument("--n_flows", default=5, type=int, help="number of planar flow layers")
    parser.add_argument("--hidden_size", default=100, type=int, help="number of hidden units in each flow layer")
    # NOTE: n_epochs will default to 1 for GAN and CGAN
    parser.add_argument("--n_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--n_samples", default=30000, type=int, help="total number of data points in toy dataset")
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--out_dir", type=str, default="maf", choices=['maf', 'gan_nonsat', 'gan_wass_gp', 'gan_nonsat_conditional'], help="path to output directory")

    # GAN, CGAN Arguments
    parser.add_argument("--loss_type", type=str, default="nonsaturating", choices=['nonsaturating', 'wasserstein_gp'], help="loss to train the GAN")

    args = parser.parse_args()

    if args.cache == True:
        # load fashion MNIST
        _ = torchvision.datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
    else:
        main(args)

