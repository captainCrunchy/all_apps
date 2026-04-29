# Load the training data
with open('training_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Total characters:", len(text))
print("First 200 characters:")
print(text[:200])

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary size:", vocab_size)

# Create mapping from characters to numbers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode the text
encoded = for ch in text :20])