from torch.utils.data import Dataset, DataLoader

class DatasetWithIndices(Dataset):
    # https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/12
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return index, data, target

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from torchvision import datasets, transforms

    mnist_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    indexed_dataset = DatasetWithIndices(mnist_dataset)
    loader = DataLoader(indexed_dataset,
                        batch_size=32,
                        shuffle=True)

    for batch_idx, (data, target, idx) in enumerate(loader):
        print(f"Batch idx {batch_idx}\n\tdataset index {idx}")
