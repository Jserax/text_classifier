import argparse
import time

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from model import Model
from prepare_data import prepare_data, prepare_loader


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = scheduler
        self.train_loss = []
        self.val_loss = []
        self.model.to(self.device)

    def train(self, train_loader, val_loader, epochs):
        print("Start training")
        start_time = time.time()
        for _ in tqdm(range(epochs)):
            train_loss = self._train_epoch(train_loader)
            val_loss, f1 = self._validate(val_loader)
            if self.val_loss and min(self.val_loss) > val_loss:
                torch.save(self.model, "model/model.pt")
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            print(
                f" | Train_loss: {train_loss:.4f}, Val_loss: {val_loss:.4f}, Val_F1: {f1:.4f}"
            )
        total_time = time.time() - start_time
        print(
            f"End of training. Total time: {total_time:.2f} seconds, best val_loss: {min(self.val_loss):.4f}"
        )

    def _train_epoch(self, train_loader) -> float:
        self.model.train()
        avg_loss = 0.0
        for input, mask, target in train_loader:
            input = input.to(self.device)
            mask = mask.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(input, mask)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            avg_loss += loss.item()
        return avg_loss / len(train_loader)

    def _validate(self, val_loader):
        self.model.eval()
        avg_loss = 0.0
        avg_f1 = 0.0
        with torch.no_grad():
            for input, mask, target in val_loader:
                input = input.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)
                pred = self.model(input, mask)
                loss = self.criterion(pred, target)
                avg_f1 += f1_score(target.cpu(), pred.cpu())
                avg_loss += loss.item()
        return avg_loss / len(val_loader), avg_f1 / len(val_loader)


def train(
    max_len: int = 128,
    batch_size: int = 128,
    epochs: int = 20,
    min_lr: float = 3e-07,
    max_lr: float = 3e-03,
):
    train_loader, val_loader = prepare_loader(
        prepare_data(),
        max_len,
        batch_size,
    )
    model = Model()
    for i, param in enumerate(model.parameters()):
        if i < 35:
            param.requires_grad = False

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=min_lr,
        max_lr=max_lr,
        step_size_up=int(len(train_loader) / 2),
        mode="exp_range",
        gamma=0.86,
        cycle_momentum=False,
    )
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        scheduler,
    )
    trainer.train(train_loader, val_loader, epochs)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--max_len", type=int, default=128, help="max seq len")
    argparser.add_argument("--batch_size", type=int, default=128, help="batch size")
    argparser.add_argument("--epochs", type=int, default=30, help="max train epochs")
    argparser.add_argument("--min_lr", type=float, default=3e-07, help="min lr")
    argparser.add_argument("--max_lr", type=float, default=3e-03, help="max lr")
    args = argparser.parse_args()
    train(
        args.max_len,
        args.batch_size,
        args.epochs,
        args.min_lr,
        args.max_lr,
    )
