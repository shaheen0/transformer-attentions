import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, batch, seq_len, d_model):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, X, debug=True):
        if debug:
            print(f"Input X shape: {X.shape}")
            
        Q = self.Wq(X)  # Pass input through Query layer (x @ W_q)
        K = self.Wk(X)
        V = self.Wv(X)
        
        if debug:
            print(f"Query shape: {Q.shape}")
            print(f"Key shape: {K.shape}")
            print(f"Value shape: {V.shape}")  
            
        Attention_score = torch.bmm(Q, K.transpose(-2, -1))
        
        if debug:
            print(f"Attention_scores (Q @ K^T) shape: {Attention_score.shape}")
            
        scaling = Attention_score / (self.d_model ** 0.5)
        Attention_weight = F.softmax(scaling, dim=-1) 
        
        if debug:
            print(f"Attention weights shape: {Attention_weight.shape}")
        context = torch.bmm(Attention_weight, V)
        if debug:
            print(f"Context Vector shape: {context.shape}")
        return context


# --- Vocabulary Setup ---
SOS_token = 0
EOS_token = 1

# Index2word: Computer → Human (convert numbers back to text)
index2words = {
    SOS_token: "SOS",
    EOS_token: "EOS"
}

# Create a very small vocabulary/dataset
words = "how are you doing ? i am good and you ?"
words_list = set(words.lower().split(' '))  

for word in words_list:
    # len(index2words) starts at 2, so new words get added at index 2, 3, etc.
    index2words[len(index2words)] = word   

# Word2index: Human → Computer (convert text to numbers)
word2index = {w: i for i, w in index2words.items()}


# --- Tensor Conversion ---
def convt2tensor(sentence):
    sentence = sentence.lower().split(' ')
    indexes = [word2index[word] for word in sentence]
    # view(1, -1) reshapes into 1 row and auto-computes columns
    result = torch.tensor(indexes, dtype=torch.long).view(1, -1)   
    return result


# --- Test the Attention ---
Batch_size = 1   # number if sentences sent  parallel
seq_length = 10    # Number of Tokens/words
d_model = 10      # Each word = 10-dimensional vector
VOACAB = len(word2index)

embedding = nn.Embedding(VOACAB, d_model)
attention = Attention(Batch_size, seq_length, d_model)

sentence = "how are you doing ?"
input_tensor = convt2tensor(sentence)
embeded = embedding(input_tensor)   

print(f"Embedding size: {embeded.size()}")  # torch.Size([1, 5, 10]) -> 1 batch, 5 tokens, 10 dims

# Set debug=False here so it doesn't spam your console, change to True if you need to debug
output = attention(embeded, debug=True) 
print(f"Output contextual vector: {output}")