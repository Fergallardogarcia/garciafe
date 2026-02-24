"""A function to load and split the desired dataset among clients."""

from .data_split import CustomDataset, split_data
import torchvision.transforms as transforms

def load_data(dataset_name: str, 
              dataset_path: str, 
              dataset_down: bool):

    assert dataset_name in ["MNIST", "FMNIST", "CIFAR-10", "STL-10", "EMNIST-DIGITS"], f"Invalid dataset {dataset_name} requested."
    custom_trainset, custom_testset = None, None

    if dataset_name == "MNIST":
        from .dt_mnist import load_mnist
        trainset, testset = load_mnist(data_root=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "EMNIST-DIGITS":
        from .dt_emnist import load_emnist
        trainset, testset = load_emnist(data_root=dataset_path, download=dataset_down, split="digits")

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "CIFAR-10":
        from .dt_cifar10 import load_cifar10
        trainset, testset = load_cifar10(data_root=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        # Modify data to have [S, C, H, W] format
        custom_trainset = CustomDataset(data=trainset.data.transpose((0, 3, 1, 2))/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.transpose((0, 3, 1, 2))/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "FMNIST":
        # Load Fashion-MNIST dataset
        from .dt_fmnist import load_fmnist
        trainset, testset = load_fmnist(data_root=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1)/255.0, targets=trainset.targets, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1)/255.0, targets=testset.targets, transform=ts_transform, target_transform=ts_target_transform)

    elif dataset_name == "STL-10":
        # Load STL-10 dataset
        from .dt_stl10 import load_stl10
        trainset, testset = load_stl10(out_dir=dataset_path, download=dataset_down)

        # Extract transforms from train and test set.
        tr_transform = transforms.Compose(trainset.transform.transforms) if trainset.transform else None
        tr_target_transform = transforms.Compose(trainset.target_transform.transforms) if trainset.target_transform else None
        ts_transform = transforms.Compose(testset.transform.transforms) if testset.transform else None
        ts_target_transform = transforms.Compose(testset.target_transform.transforms) if testset.target_transform else None

        # Build Custom datasets
        # For some weird reason STL-10 has labels instead 
        # of targets adding additional attribute targets 
        # to make it consistent with other datasets
        custom_trainset = CustomDataset(data=trainset.data, targets=trainset.labels, transform=tr_transform, target_transform=tr_target_transform)
        custom_testset = CustomDataset(data=testset.data, targets=testset.labels, transform=ts_transform, target_transform=ts_target_transform)

    else:
        raise ValueError(f"Invalid dataset {dataset_name} requested.")

    # Performing cleanups
    del trainset, testset

    return custom_trainset, custom_testset

def load_and_fetch_split(
        n_clients: int,
        dataset_conf: dict
    ):
    """A routine to load and split data."""

    # load the dataset requested
    trainset, testset \
        = load_data(dataset_name=dataset_conf["DATASET_NAME"],
                    dataset_path=dataset_conf["DATASET_PATH"],
                    dataset_down=dataset_conf["DATASET_DOWN"]
                   )

    # split the dataset if requested
    if dataset_conf["SPLIT"]:
        train_splits, split_labels \
            = split_data(
                train_data = trainset,
                num_partitions = n_clients,
                split_method = dataset_conf["SPLIT_METHOD"],
                dirichlet_alpha = dataset_conf["DIRICHLET_ALPHA"], 
                random_seed = dataset_conf["RANDOM_SEED"], 
                min_partition_size = dataset_conf["MIN_PARTITION_SIZE"],
                classes_per_worker = dataset_conf["CLASSES_PER_WORKER"]
            )

        # Performing cleanups
        del trainset

        return (train_splits, split_labels), testset

    else:
        return (trainset, None), testset
