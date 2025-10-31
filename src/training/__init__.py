# src/training/__init__.py
"""
Training package.

Contains:
- trainer.py   : generic PyTorch training loop (train/val, early stopping, checkpointing)
- evaluator.py : metrics + plotting utilities (MAE, RMSE, MAPE, plots)

Student note:
Use these modules as a starting point. Understand each function and adapt for your experiments.
"""
__all__ = ["trainer", "evaluator"]
