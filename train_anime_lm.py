import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.model_selection import train_test_split

# === STEP 1: Load & Clean CSV ===
print("🔍 Loading dataset...")
df = pd.read_csv("lessreal-data.csv", delimiter=";")

# Keep only 'Quote' column and drop missing values
df = df[['Quote']].dropna()
df = df.rename(columns={"Quote": "text"})  # Rename for HuggingFace

# === Optional: Train-Test Split for Evaluation ===
train_texts, val_texts = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_texts)
eval_dataset = Dataset.from_pandas(val_texts)

print(f"✅ Loaded {len(train_dataset)} training and {len(eval_dataset)} eval quotes.")

# === STEP 2: Tokenization ===
print("✂️ Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Fix padding token issue

def tokenize_function(batch):
    tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# === STEP 3: Load Model ===
print("⚙️ Loading small language model...")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id  # Set pad token ID

# === STEP 4: Training Arguments ===
print("🚀 Starting training...")
training_args = TrainingArguments(
    output_dir="./anime-small-lm",
    eval_strategy="epoch",
    logging_strategy="steps",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    fp16=False,  # Set to True if you have a GPU with mixed precision
)

# === STEP 5: Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

trainer.train()

# === STEP 6: Save Model & Tokenizer ===
print("💾 Saving trained model...")
trainer.save_model("./anime-small-lm")
tokenizer.save_pretrained("./anime-small-lm")

# === STEP 7: Generate Example Output ===
print("✨ Generating sample text...")
generator = pipeline("text-generation", model="./anime-small-lm", tokenizer="./anime-small-lm")

sample = generator("Even if I fall,", max_length=30, do_sample=True, top_k=50)
print("🗯️ AnimeBot says:", sample[0]["generated_text"])