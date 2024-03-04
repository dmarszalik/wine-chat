import os
os.kill(os.getpid(), 9)

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Wczytaj wytrenowany model
model_path = "/content/drive/MyDrive/datasets/wines_processed/model/gpt2-finetuned/checkpoint-20000"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_length=50)

# Load the validation set
from datasets import load_dataset
val_dataset = load_dataset('text', data_files={"/content/drive/MyDrive/datasets/wines_processed/val_data.txt"})#, split="train" , sample_by="paragraph"
with open("/content/drive/MyDrive/datasets/wines_processed/val_data.txt", "r", encoding="utf-8") as file:
    validation_text = file.readlines()

len(validation_text)

val_dataset = val_dataset.map(lambda examples: tokenizer(examples["text"], return_tensors="np"), batched=True)

val_dataset

validation_text[11]

# Generuj tekst za pomocą wytrenowanego modelu
generated_texts = []
generated_text = generator('Cabernet Sauvignon')[0]['generated_text']
generated_texts.append(generated_text)

generated_texts

