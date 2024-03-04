import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from core import WineGPTTrainer


if __name__ == "__main__":
    trainer = WineGPTTrainer(train_data_path="/content/drive/MyDrive/datasets/wines_processed/wines_vectors.csv",
                              model_output_dir="/content/drive/MyDrive/datasets/wines_processed/model/gpt2-finetuned/checkpoint-20000")
    trainer.train_model()
