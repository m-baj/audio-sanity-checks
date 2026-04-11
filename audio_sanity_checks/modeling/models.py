import lightning as L
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.transforms import v2
import torch


class SpectrogramModel(L.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()

        weights = ResNet101_Weights.DEFAULT
        self.model = resnet101(weights=weights)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_epochs = 120
        self.preprocess = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        preds = y_hat.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        preds = y_hat.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
