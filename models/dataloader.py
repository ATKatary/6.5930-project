from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class Loader(LightningDataModule):
    def __init__(self, data_dir, batch_size, train_data, val_data, val_transforms, train_transforms, **kwargs):
        super().__init__()
        self.data_dir = data_dir

        self.batch_size = batch_size

        self.val_data = val_data 
        self.train_data = train_data

        self.val_transforms = val_transforms
        self.train_transforms = train_transforms

    def setup(self, stage = None):
        val_transforms = transforms.Compose(self.val_transforms)
        train_transforms = transforms.Compose(self.train_transforms)

        self.train_dataset = self.train_data(
            split = 'train',
            download = False,
            root = self.data_dir,
            transform=train_transforms,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = self.val_data(
            split = 'test',
            download = False,
            root = self.data_dir,
            transform=val_transforms,
        )
        
    def train_dataloader(self):
        return DataLoader(
            shuffle=True,
            dataset = self.train_dataset,
            batch_size = self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            shuffle=False,
            dataset = self.val_dataset,
            batch_size = self.batch_size,
        )
    
    def test_dataloader(self):
        return DataLoader(
            shuffle = True,
            batch_size = 144,
            dataset = self.val_dataset,
        )