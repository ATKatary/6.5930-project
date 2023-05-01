from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def worker(log_dir, name, epochs):
    logger = TensorBoardLogger(
        save_dir = log_dir,
        name = f"{name}_training"
    )
    trainer = Trainer(
        logger = logger,
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k = 1,
                dirpath = f"{log_dir}/checkpoints",
                monitor = f"{name}_loss",
                save_last = True
            )
        ],
        max_epochs = epochs
    )

    return trainer

def train(model, train_loader, log_dir, name, epochs = 100):
    worker(log_dir, name, epochs).fit(model, train_loader)

def test(model, test_loader, log_dir, name, epochs = 1):
    worker(log_dir, name, epochs).test(model, test_loader)