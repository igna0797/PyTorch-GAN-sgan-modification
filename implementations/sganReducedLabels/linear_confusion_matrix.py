import os
import torch
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sgan import Discriminator, Generator
from utils import parseArguments, NoiseAdder, get_opt_path


def load_dataset(opt):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST(
        root="../../data/mnist",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=True
    )


def load_discriminator(weights_path, device):
    model = Discriminator().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_generator(weights_path, device):
    model = Generator().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def evaluate_confusion(discriminator, generator, dataloader, device, opt):
    confusion = defaultdict(lambda: defaultdict(int))

    correct = 0
    total = 0

    FAKE_LABEL = opt.num_classes  # == 10

    with torch.no_grad():
        for images, labels in dataloader:
            # -------- REAL --------
            images = images.to(device)
            labels = labels.to(device)

            real_imgs,_= NoiseAdder.add_noise(images, opt)
            real_imgs = real_imgs.to(device)

            _, real_logits = discriminator(real_imgs)

            # -------- FAKE --------
            z = torch.randn(
                labels.size(0),
                opt.latent_dim,
                device=device
            )

            fake_imgs = generator(z)
            fake_imgs,_ = NoiseAdder.add_noise(fake_imgs, opt)
            fake_imgs = fake_imgs.to(device)

            _, fake_logits = discriminator(fake_imgs)

            # -------- CONCAT (IDENTICO A TRAINING) --------
            preds = torch.cat([real_logits, fake_logits], dim=0)

            fake_gt = torch.full(
                (labels.size(0),),
                FAKE_LABEL,
                device=device,
                dtype=labels.dtype
            )

            gt = torch.cat([labels, fake_gt], dim=0)

            pred_labels = torch.argmax(preds, dim=1)

            correct += torch.sum(pred_labels == gt).item()
            total += gt.size(0)

            # -------- CONFUSION --------
            for t, p in zip(gt, pred_labels):
                confusion[t.item()][p.item()] += 1

    accuracy = 100.0 * correct / total
    print(f"Accuracy (idéntico a training): {accuracy:.2f}%")

    plot_confusion(confusion, "confusion_matrix.png", percentage=False)
    plot_confusion(confusion, "confusion_matrix_percentage.png", percentage=True)

    return accuracy


def plot_confusion(confusion, filename, percentage=False):
    true_labels = sorted(confusion.keys())
    pred_labels = sorted({p for v in confusion.values() for p in v})

    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.float32)

    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            matrix[i, j] = confusion[t].get(p, 0)

    if percentage:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0) * 100
        fmt = ".1f"
        cmap = "viridis"
        cbar_label = "Percentage (%)"
    else:
        fmt = "g"
        cmap = "Blues"
        cbar_label = "Count"

    plt.figure(figsize=(12, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=pred_labels,
        yticklabels=true_labels,
        annot_kws={"size": 18},
        cbar=False
        #cbar_kws={"label": cbar_label}
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Etiqueta predicha", fontsize=14)
    plt.ylabel("Etiqueta Real", fontsize=14)
    plt.title("Matriz de confusión", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Saved {filename}")


if __name__ == "__main__":
    args = parseArguments()
    opt_path = get_opt_path(__file__, weights_path=args.weights_path)

    with open(opt_path, "rb") as f:
        opt = pickle.load(f)

    if not hasattr(opt, "noise_type"):
        opt.noise_type = "lines"
        print("using default noise : lines")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dir = os.path.dirname(args.weights_path)
    discriminator_path = os.path.join(weight_dir, "discriminator_weights.pth")
    generator_path = os.path.join(weight_dir, "generator_weights.pth")

    dataloader = load_dataset(opt)
    discriminator = load_discriminator(discriminator_path, device)
    generator = load_generator(generator_path, device)

    evaluate_confusion(discriminator, generator, dataloader, device, opt)
