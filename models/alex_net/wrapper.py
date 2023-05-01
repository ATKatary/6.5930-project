import os
from torch import optim
from pytorch_lightning import LightningModule

class AlexNet_Wrapper(LightningModule):
    def __init__(self, model, lr, weight_decay):
        super(AlexNet_Wrapper, self).__init__()
        self.lr = lr
        self.model = model
        self.weight_decay = weight_decay
        try: self.hold_graph = self.params['retain_first_backpass']
        except: pass

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist = True)

        return train_loss['loss']
    
    def validation_step(self, batch):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist = True)
        
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        optims.append(optimizer)
        return optims
    