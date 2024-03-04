# wine_gpt_trainer.py

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class WineGPTTrainer:
    def __init__(self, train_data_path, model_output_dir, num_train_epochs=3, per_device_train_batch_size=6, save_steps=10_000, save_total_limit=2):
        self.train_data_path = train_data_path
        self.model_output_dir = model_output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_training_data(self):
        wine_data = pd.read_csv(self.train_data_path)
        combined_text_data = wine_data['combined_text'].str.replace('\n', '').tolist()
        train_texts, test_val_texts = train_test_split(combined_text_data, test_size=0.2, random_state=42)
        test_texts, val_texts = train_test_split(test_val_texts, test_size=0.5, random_state=42)

        train_path = f"{self.model_output_dir}/train_data.txt"
        test_path = f"{self.model_output_dir}/test_data.txt"
        val_path = f"{self.model_output_dir}/val_data.txt"

        with open(train_path, 'w') as f:
            for text in train_texts:
                f.write(text + '\n')

        with open(test_path, 'w') as f:
            for text in test_texts:
                f.write(text + '\n')

        with open(val_path, 'w') as f:
            for text in val_texts:
                f.write(text + '\n')

        train_dataset = load_dataset('text', data_files={"train": train_path})
        train_dataset = train_dataset.map(lambda examples: self.tokenizer(examples["text"], return_tensors="pt"), batched=True)

        return train_dataset['train']

    def train_model(self):
        train_dataset = self.prepare_training_data()

        training_args = TrainingArguments(
            output_dir=self.model_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
        )

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            train_dataset=train_dataset,
        )

        trainer.train()
