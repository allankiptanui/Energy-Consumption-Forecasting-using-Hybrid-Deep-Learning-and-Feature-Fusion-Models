from typing import Tuple, Dict, Optional
import os
import math
import random
import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 0):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """
    early stopping utility:
    - monitors validation loss
    - if no improvement after `patience` epochs, stop.
    """
    def __init__(self, patience: int = 8, min_delta: float = 1e-6):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, current_loss: float, model: torch.nn.Module):
        improved = current_loss + self.min_delta < self.best_loss
        if improved:
            self.best_loss = current_loss
            self.counter = 0
            # store a copy of best state_dict
            self.best_state = copy.deepcopy(model.state_dict())
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self) -> bool:
        return self.counter >= self.patience

    def best_weights(self):
        return self.best_state


def _make_dataloader(dataset, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0):
    """Small wrapper for DataLoader to keep calls compact and readable."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_model(model: torch.nn.Module,
                train_dataset,
                val_dataset,
                cfg: Dict,
                device: Optional[str] = None,
                save_path: str = "results/checkpoints/model_best.pth",
                seed: int = 0) -> Tuple[Dict, str]:
    """
    Train `model` using datasets and config `cfg`.

    cfg keys used:
      - batch_size
      - epochs
      - lr
      - weight_decay (optional)
      - early_stop_patience
      - num_workers (optional)
      - scheduler (optional): dict with keys name (e.g., 'StepLR') and params

    Returns:
      history: dict with train_loss and val_loss lists
      best_model_path: path where best model was saved (same as save_path)
    """
    # reproducibility
    set_seed(seed)

    # device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # dataloaders
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = _make_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _make_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # optimizer & optional scheduler
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_cfg = cfg.get("scheduler", None)
    scheduler = None
    if scheduler_cfg:
        name = scheduler_cfg.get("name", "").lower()
        params = scheduler_cfg.get("params", {})
        if name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
        elif name == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
         

    # loss
    criterion = torch.nn.MSELoss()

    # early stopping
    patience = int(cfg.get("early_stop_patience", 8))
    early_stopper = EarlyStopping(patience=patience, min_delta=cfg.get("min_delta", 1e-6))

    model.to(device)
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    epochs = int(cfg.get("epochs", 50))
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train", leave=False)
        for batch in loop:
            # expecting dataset to return (seq, static, target)
            seq, static, target = batch
            seq = seq.to(device)
            static = static.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            # outputs = model(seq, static)
            # # some models return (pred, extra) e.g. attention weights
            # if isinstance(outputs, tuple) or isinstance(outputs, list):
            #     preds = outputs[0]
            # else:
            #     preds = outputs
            # loss = criterion(preds, target)
            outputs = model(seq, static)
            # handle models returning (preds, extra)
            if isinstance(outputs, (tuple, list)):
                preds = outputs[0]
            else:
                preds = outputs

            # Normalize shapes make preds and target 1D (batch,)
            if preds.dim() > 1 and preds.size(-1) == 1:
                preds = preds.view(-1)
            if target.dim() > 1 and target.size(-1) == 1:
                target = target.view(-1)

            # final sanity cast to float tensors
            preds = preds.float()
            target = target.float()

            loss = criterion(preds, target)
            
   
            
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix({'batch_loss': loss.item()})

        avg_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float('nan')

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                seq, static, target = batch
                seq = seq.to(device)
                static = static.to(device)
                target = target.to(device)
                outputs = model(seq, static)
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    preds = outputs[0]
                else:
                    preds = outputs
                loss = criterion(preds, target)
                val_losses.append(loss.item())
        avg_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('nan')

        # scheduler step 
        if scheduler is not None:
            # if scheduler is ReduceLROnPlateau, call step with val loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        epoch_time = time.time() - t0
        tqdm.write(f"Epoch {epoch:03d} train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} time={epoch_time:.1f}s")

        # early stopping check
        improved = early_stopper.step(avg_val_loss, model)
        if improved:
            # save best weights (state held inside early_stopper.best_state)
            torch.save(early_stopper.best_weights(), save_path)
            best_val = avg_val_loss

        if early_stopper.should_stop():
            print(f"Early stopping at epoch {epoch}. Best val_loss={early_stopper.best_loss:.6f}")
            break

    # load best weights into model if available
    if early_stopper.best_weights() is not None:
        model.load_state_dict(early_stopper.best_weights())

    return history, save_path
