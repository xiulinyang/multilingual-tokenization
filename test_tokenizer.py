from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("models/EN_5000_41")

print(tokenizer.backend_tokenizer.normalizer)
# Encode both "L" and "l"
encoded_l = tokenizer.encode("l")
encoded_L = tokenizer.encode("L")

print("Token IDs for 'l':", encoded_l)
print("Token IDs for 'L':", encoded_L)

# Optional: print decoded tokens
print("Decoded from 'l':", tokenizer.decode(encoded_l))
print("Decoded from 'L':", tokenizer.decode(encoded_L))
