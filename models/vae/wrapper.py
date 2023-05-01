import os
from torch import optim
import torchvision.utils as vutils
from pytorch_lightning import LightningModule

class VAE_Wrapper(LightningModule):
    def __init__(self, model, lr, weight_decay):
        super(VAE_Wrapper, self).__init__()
        self.lr = lr
        self.model = model
        self.kld_weight = 0.00025
        self.weight_decay = weight_decay
        try: self.hold_graph = self.params['retain_first_backpass']
        except: pass

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results, M_N = self.kld_weight)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist = True)

        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results, M_N = self.kld_weight)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist = True)
    
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        if not os.path.exists(os.path.join(self.logger.log_dir, "reconstructions")):
            os.makedirs(os.path.join(self.logger.log_dir, "reconstructions"))

        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(
            recons.data,
            os.path.join(
              self.logger.log_dir , 
              "reconstructions", 
              f"recons_{self.logger.name}_epoch_{self.current_epoch}.png"),
            normalize = True,
            nrow = 12
        )

        try:
            samples = self.model.sample(144, self.curr_device, labels = test_label)
            if not os.path.exists(os.path.join(self.logger.log_dir, "samples")):
                os.makedirs(os.path.join(self.logger.log_dir, "samples"))
                
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.logger.log_dir, 
                    "samples",      
                    f"{self.logger.name}_epoch_{self.current_epoch}.png"
                ),
                normalize = True,
                nrow = 12
            )
        except Warning: pass
        except FileExistsError: pass
        
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        optims.append(optimizer)
        return optims
    