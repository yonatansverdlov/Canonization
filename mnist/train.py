from statistics import mean, stdev

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.data_funcs import obtain, get_dataset
from torch.utils.data import DataLoader
from utils.models import AverageCNN, CNp4CNN, CNN
from argparse import ArgumentParser

def get_hyperparams():
    parser = ArgumentParser()
    ...
    parser.add_argument(
        "--model_type",
        type=str,
        default="average",
        choices=["average", "learned_can","cnn"],
        help="which model to run",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="How many seeds to run",
    )
    args = parser.parse_args()
    return args


class MNISTModel(pl.LightningModule):
    def __init__(self, lr=1e-3, model_type="cnn"):
        super().__init__()
        self.save_hyperparameters()

        self.im_shape = (1, 28, 28)
        self.loss_fn = nn.CrossEntropyLoss()
        if model_type == 'cnn':
            self.model = CNN(self.im_shape, out_channels=32, num_layers=6)
        elif model_type == "average":
            self.model = AverageCNN()
        elif model_type == 'learned_can': 
            self.model = CNp4CNN()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _prepare_x(self, x):
        return x.reshape(x.size(0), self.im_shape[0], self.im_shape[1], self.im_shape[2])

    def forward(self, x):
        x = self._prepare_x(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log("test/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


class DatasetDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=256, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def run_seed(seed, train_dataset, val_dataset, test_dataset, lr=1e-3, batch_size=256, model_type="canonized"):
    pl.seed_everything(seed, workers=True)

    datamodule = DatasetDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4,
    )

    model = MNISTModel(lr=lr, model_type=model_type)

    ckpt_dir = f"data/trained_models/{model_type}_seed_{seed}_checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename=f"best-{model_type}-seed-{seed}",
    )

    early_stopping = EarlyStopping(
        monitor="val/acc",
        mode="max",
        patience=20,
        min_delta=0.0,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stopping],
        check_val_every_n_epoch=10,
        limit_val_batches=100,
        inference_mode=False,
        logger=False,
    )

    trainer.fit(model, datamodule=datamodule)
    test_result = trainer.test(model, datamodule=datamodule, ckpt_path="best")[0]
    return {
        "seed": seed,
        "best_val_acc": checkpoint_callback.best_model_score.item(),
        "test_acc": test_result["test/acc"],
        "test_loss": test_result["test/loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }


if __name__ == "__main__":
    data_path = "data/rotated_mnist"
    obtain(data_path)

    train_dataset = get_dataset(data_path, split="train")
    val_dataset = get_dataset(data_path, split="valid")
    test_dataset = get_dataset(data_path, split="test")


    # choose one:
    parser = get_hyperparams()
    MODEL_TYPE = parser.model_type
    seeds = list(range(parser.num_seeds))  # you can change this to run fewer seeds for a quick test
    results = []

    for seed in seeds:
        print(f"\n===== Running {MODEL_TYPE}, seed {seed} =====")
        result = run_seed(
            seed,
            train_dataset,
            val_dataset,
            test_dataset,
            lr=1e-3,
            batch_size=256,
            model_type=MODEL_TYPE,
        )
        results.append(result)
        print(result)

    test_accs = [r["test_acc"] for r in results]

    print("\n===== Summary =====")
    print("Model type:", MODEL_TYPE)
    print("All test_accs:", test_accs)
    print("Mean test_acc:", mean(test_accs))
    if len(seeds) > 1:
        print("Std test_acc:", stdev(test_accs))