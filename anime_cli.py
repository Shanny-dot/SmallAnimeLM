import random
from transformers import pipeline

# Load the fine-tuned model
generator = pipeline("text-generation", model="./anime-small-lm", tokenizer="./anime-small-lm")

# Random anime-style prompts
prompts = [
    "Even if I fall,",
    "No matter how dark it gets,",
    "My strength comes from",
    "I won't forgive them because",
    "Friendship means",
    "I have a dream to",
    "When the rain falls, I remember",
    "Power without purpose is",
]

print("\n🎌 Welcome to AnimeBot CLI — Unleash the Anime Within 🎌")
print("Type your own prompt or just press Enter for a surprise!\n(Type 'exit' to quit)\n")

while True:
    user_input = input("📝 Prompt >>> ").strip()

    if user_input.lower() == "exit":
        print("\n👋 Bye buddy! Stay anime.\n")
        break

    if user_input == "":
        prompt = random.choice(prompts)
    else:
        prompt = user_input

    print("⚡ Generating...\n")
    result = generator(prompt, max_length=30, do_sample=True, top_k=50, truncation=True)
    print(f"🗯️ AnimeBot says:\n{result[0]['generated_text']}\n" + "-" * 50 + "\n")