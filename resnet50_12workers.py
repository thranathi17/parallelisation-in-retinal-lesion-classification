# Environment Setup and Library Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from torchvision.io import read_image
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import time
import matplotlib.pyplot as plt
import copy
from torch.distributed import all_reduce, ReduceOp
from tqdm import tqdm
import warnings
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# Ignore Warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ----- Custom Dataset Class -----
# Defines a custom dataset class to handle loading images from a directory structure.
# It includes preprocessing and transformations for the images.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initializes the custom dataset object.

        Args:
            root_dir (str): The root directory containing the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.imgs = self.make_dataset()

    def _find_classes(self, dir):
        """
        Identifies the classes in the dataset directory.

        Args:
            dir (str): The directory to search for classes.

        Returns:
            tuple: Contains a list of class names and a dictionary mapping class names to indices.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self):
        """
        Creates a list of image paths and their corresponding class indices.

        Returns:
            list: A list of tuples, each containing the path to an image and its class index.
        """
        images = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if not fname.startswith("."):  # Skip hidden files and folders
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        images.append(item)
        return images

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label at a given index.

        Args:
            idx (int): Index of the data item.

        Returns:
            tuple: Contains the image tensor, its class index, and the image path.
        """
        path, class_index = self.imgs[idx]
        image = read_image(path)

        # Convert grayscale to RGB if necessary
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)

        image = T.Resize((299, 299), antialias=True)(image)
        return image, class_index, path


# ----- Distributed Environment Setup -----
# Functions to set up and clean up the distributed environment for multi-GPU training.
# It initializes the process group for NCCL backend and sets the CUDA device based on the process rank.
def setup(rank, world_size):
    """
    Sets up the environment for distributed training.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes involved in the training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """
    Cleans up the distributed training environment.
    """
    dist.destroy_process_group()


# ----- DataLoader with DistributedSampler -----
# Prepares PyTorch DataLoader with a DistributedSampler to efficiently distribute the data across multiple GPUs with 12 workers.
def prepare_dataloader(dataset, batch_size, world_size, rank, num_workers=12):
    """
    Prepares a DataLoader with a DistributedSampler for distributed training.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        world_size (int): The total number of processes in the training.
        rank (int): The rank of the current process.
        num_workers (int, optional): The number of subprocesses to use for data loading.

    Returns:
        DataLoader: The DataLoader configured for distributed training.
    """
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader


# ----- Model Initialization Function -----
def initialize_resnet_model(num_classes=4):
    """
    Initializes and customizes the ResNet model for the specific dataset.

    Args:
        num_classes (int, optional): The number of classes in the dataset.

    Returns:
        model (torch.nn.Module): The modified ResNet model.
    """
    # Load the pre-trained ResNet model
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze some layers
    #     for param in list(model.parameters())[-6:]:
    #         param.requires_grad = True

    # Modify the final classification layer for the number of classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.1),
        nn.Linear(128, num_classes),
    )

    return model


# ----- Optimizer and Loss Function Initialization -----
# Initializes the RMSprop optimizer and the cross-entropy loss function.
def initialize_optimizer_and_criterion(model):
    """
    Initializes the optimizer and loss function for the model.

    Args:
        model (torch.nn.Module): The neural network model to be trained.

    Returns:
        tuple: Contains the optimizer (torch.optim.Optimizer) and the loss function (torch.nn.Module).
    """
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


# ----- Train Model -----
# Contains the main training loop for the model. It also handles validation and keeps track of the best model based on validation accuracy.
# The function employs distributed all-reduce operations to aggregate metrics across GPUs.
def train_resnet_model(
    model, criterion, optimizer, data_loaders, rank, world_size, num_epochs=3
):
    """
    Trains the model using the given data loaders, optimizer, and loss criterion.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        data_loaders (dict): Dictionary containing 'train' and 'val' DataLoader objects.
        rank (int): The rank of the current process in distributed training.
        world_size (int): Total number of processes involved in the training.
        num_epochs (int, optional): Number of epochs to train for.

    Returns:
        dict: A dictionary containing training metrics like loss and accuracy.
    """
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "time": [],
    }

    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Use tqdm only for rank 0
            data_iter = data_loaders[phase]
            if rank == 0:
                data_iter = tqdm(
                    data_iter,
                    desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}",
                )

            for inputs, labels, _ in data_iter:
                inputs = inputs.to(rank)
                labels = labels.to(rank)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            # Gather statistics from all processes
            loss_tensor = torch.tensor(running_loss / total_samples).to(rank)
            corrects_tensor = torch.tensor(running_corrects / total_samples).to(rank)
            all_reduce(loss_tensor, op=ReduceOp.SUM)
            all_reduce(corrects_tensor, op=ReduceOp.SUM)

            epoch_loss = loss_tensor.item() / world_size
            epoch_acc = corrects_tensor.item() / world_size

            metrics[f"{phase}_loss"].append(epoch_loss)
            metrics[f"{phase}_acc"].append(epoch_acc)

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Rank: {rank}"
            )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    metrics["time"].append(time_elapsed)

    if rank == 0:
        if world_size == 1:
            torch.save(
                best_model_wts, f"ResNet50_12W/resnet_model_best_{world_size}g_12w.pth"
            )
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))
        metrics["model"] = None  # Remove the model from metrics before saving
        with open(f"ResNet50_12W/training_metrics_{world_size}g_12w.json", "w") as f:
            json.dump(metrics, f)

        # Load best model weights
        model.load_state_dict(best_model_wts)

    metrics["model"] = model
    return metrics


# ----- Evaluate Model -----
# Evaluates the trained model on a test set and saves the predictions and true labels for further analysis.
def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model on a test set.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: Contains lists of true labels and predicted labels.
    """
    model.to("cuda")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(
                preds.cpu().numpy().astype(int).tolist()
            )  # Convert to Python int
            all_labels.extend(
                labels.cpu().numpy().astype(int).tolist()
            )  # Convert to Python int

    # Save to JSON file
    with open("ResNet50_12W/model_predictions_12w.json", "w") as f:
        json.dump({"labels": all_labels, "predictions": all_preds}, f)

    return all_labels, all_preds


# ----- Main Function for Training -----
# The entry point for the training process. It handles the spawning of processes for each GPU,
# sets up datasets and data loaders, initializes the model and loss/optimization functions, and calls the training function.
def main(rank, world_size, num_epochs, batch_size, train_dir, val_dir, test_dir):
    """
    Main function for distributed training. Sets up the environment, datasets, and model, and starts the training process.

    Args:
        rank (int): The rank of the current process.
        world_size (int): Total number of processes in distributed training.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        train_dir (str): Directory containing the training dataset.
        val_dir (str): Directory containing the validation dataset.
        test_dir (str): Directory containing the test dataset.
    """
    try:
        setup(rank, world_size)

        train_dataset = CustomDataset(root_dir=train_dir)
        val_dataset = CustomDataset(root_dir=val_dir)
        train_loader = prepare_dataloader(train_dataset, batch_size, world_size, rank)
        val_loader = prepare_dataloader(val_dataset, batch_size, world_size, rank)
        test_dataset = CustomDataset(root_dir=test_dir)
        test_loader = prepare_dataloader(test_dataset, batch_size, world_size, rank)

        model = initialize_resnet_model()
        model = model.to(rank)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )

        optimizer, criterion = initialize_optimizer_and_criterion(model)

        data_loaders = {"train": train_loader, "val": val_loader}

        metrics = train_resnet_model(
            model, criterion, optimizer, data_loaders, rank, world_size, num_epochs
        )
        if rank == 0 and world_size == 1:
            model.load_state_dict(
                torch.load(f"ResNet50_12W/resnet_model_best_{world_size}g_12w.pth")
            )
            evaluate_model(model, test_loader, rank)

    except Exception as e:
        print(f"Error in process {rank}: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    max_gpus = 4  # Maximum number of GPUs you want to use
    num_epochs = 5
    batch_size = 256
    train_dir = "/home/venugopalbalamurug.l/Dataset/train"
    val_dir = "/home/venugopalbalamurug.l/Dataset/val"
    test_dir = "/home/venugopalbalamurug.l/Dataset/test"
    for world_size in range(1, max_gpus + 1):
        print(f"Training with {world_size} GPU(s)")

        mp.spawn(
            main,
            args=(world_size, num_epochs, batch_size, train_dir, val_dir, test_dir),
            nprocs=world_size,
            join=True,
        )

        print(f"Completed training with {world_size} GPU(s)\n")
