from torchvision import datasets, transforms
from base import BaseDataLoader


class PlantVillageLoader(BaseDataLoader):
    """
    PlantVillage Disease Classification Challenge data loader using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=2,
        training=True,
    ):
        trsfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=trsfm)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
