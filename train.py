"""
Run this once to train all models and save them.
Usage: python train.py
"""
from backend.models import train_and_save_all

if __name__ == "__main__":
    train_and_save_all(data_path="data/creditcard.csv")