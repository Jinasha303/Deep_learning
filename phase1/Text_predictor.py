import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# ── 1. Download dataset ──────────────────────────────────────────────────────
shakespeare = keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# ── 2. Read and tokenize ─────────────────────────────────────────────────────
with open(shakespeare, encoding='utf-8') as f:
    text = f.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

vocab_size = len(tokenizer.word_index) + 1  # +1 for padding index 0
print(f"Vocabulary size: {vocab_size}")

input_seq = []
for line in text.split('\n'):
    if not line.strip():
        continue
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i + 1]
        input_seq.append(n_gram_seq)

print(f"Total n-gram sequences: {len(input_seq)}")

max_sequence_len = max([len(seq) for seq in input_seq])
padded_input = pad_sequences(input_seq, maxlen=max_sequence_len, padding='pre')

x = padded_input[:, :-1]          # all tokens except last
y = padded_input[:, -1]           # last token is the label
y = to_categorical(y, num_classes=vocab_size)

input_length = max_sequence_len - 1   # matches x shape
print(f"x shape: {x.shape}, y shape: {y.shape}, input_length: {input_length}")

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=input_length),
    LSTM(150, return_sequences=True),   # return_sequences=True required for stacked LSTM
    LSTM(150),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
model.fit(x, y, epochs=100, verbose=1)