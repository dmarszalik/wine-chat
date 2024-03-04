import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

wine_data = pd.read_csv('/content/drive/MyDrive/datasets/wines_processed/wines_vectors.csv')

wine_data.sample(10)

# Krok 1: Przygotowanie danych treningowych
# Wczytanie danych
combined_text_data = wine_data['combined_text']

combined_text_data[0]

cleaned_texts = [text.replace('\n', '') for text in combined_text_data]

from sklearn.model_selection import train_test_split

train_texts, test_val_texts = train_test_split(cleaned_texts, test_size=0.2, random_state=42)
test_texts, val_texts = train_test_split(test_val_texts, test_size=0.5, random_state=42)

print("Liczba przykładów w zbiorze treningowym:", len(train_texts))
print("Liczba przykładów w zbiorze testowym:", len(test_texts))
print("Liczba przykładów w zbiorze walidacyjnym:", len(val_texts))

train_texts[11]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Ustawienie tokena paddingu na token końca sekwencji

tokenizer.save_pretrained("/content/drive/MyDrive/datasets/wines_processed/model/gpt2-finetuned/checkpoint-20000")

def count_tokens(text):
    # Tokenizacja tekstu
    tokens = tokenizer(text)['input_ids']
    # Zliczenie tokenów
    return len(tokens)

# Obliczenie liczby tokenów dla każdej sekwencji tekstu
wine_data['num_tokens'] = wine_data['combined_text'].apply(count_tokens)

# Obliczenie średniej liczby tokenów
average_tokens = wine_data['num_tokens'].mean()

# Obliczenie maksymalnej liczby tokenów
max_tokens = wine_data['num_tokens'].max()

print("Średnia liczba tokenów: ", average_tokens)
print("Maksymalna liczba tokenów: ", max_tokens)



# Zapisz dane do pliku tekstowego

train_path = '/content/drive/MyDrive/datasets/wines_processed/train_data.txt'
test_path = '/content/drive/MyDrive/datasets/wines_processed/test_data.txt'
val_path = '/content/drive/MyDrive/datasets/wines_processed/val_data.txt'

# Otwórz plik do zapisu
with open(train_path, 'w') as f:
    # Zapisz każdy tekst z listy do pliku
    for text in train_texts:
        f.write(text + '\n')


with open(test_path, 'w') as f:
    # Zapisz każdy tekst z listy do pliku
    for text in test_texts:
        f.write(text + '\n')


with open(val_path, 'w') as f:
    # Zapisz każdy tekst z listy do pliku
    for text in val_texts:
        f.write(text + '\n')

model_path = "/content/drive/MyDrive/datasets/wines_processed/model/gpt2-finetuned/checkpoint-20000"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# # Utwórz TextDataset
# train_dataset = TextDataset(
#     file_path='/content/drive/MyDrive/datasets/wines_processed/train_data.txt',
#     tokenizer=tokenizer,
#     block_size=128,  # Ustaw odpowiedni rozmiar bloku
# )
from datasets import load_dataset
train_dataset = load_dataset('text', data_files={"/content/drive/MyDrive/datasets/wines_processed/train_data.txt"})

train_dataset = train_dataset.map(lambda examples: tokenizer(examples["text"], return_tensors="np"), batched=True)

train_dataset

[]

# Krok 2: Wybór modelu GPT-2
# Załadowanie modelu
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Krok 3: Trenowanie modelu GPT-2
# Przygotowanie ustawień treningowych
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/datasets/wines_processed/model/gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=6,
    save_steps=10_000,
    save_total_limit=2,
)

# Inicjalizacja trainera
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=train_dataset['train'],
)



# Rozpoczęcie treningu
trainer.train()