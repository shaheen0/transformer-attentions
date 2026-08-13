import torch
import torch.nn as nn
import torch.nn.functional as F

# RULE: d_out MUST be divisible by num_heads!
# If d_out=512 and num_heads=8, head_dim = 64 
class MultiHeadAttention(nn.Module):
    def __init__(self , n_head,d_model):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model must be divisible by num_heads"
        self.Q = nn.Linear(d_model , d_model )
        self.K = nn.Linear(d_model , d_model )
        self.V = nn.Linear(d_model , d_model )
        self.Output = nn.Linear(d_model , d_model )

    def forward(self , x):
        B, T, C = x.shape  # B=Batch, T=Sequence Length (10), C=d_model (512)
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        # The Scissors (Slice into heads)
        # We reshape (B, T, 512) into (B, T, 8_heads, 64_dim)
        Q = Q.view(B, T, self.n_head, self.head_dim)
        K = K.view(B, T, self.n_head, self.head_dim)
        V = V.view(B, T, self.n_head, self.head_dim)
        # The Scissors (Transpose)
        Q = Q.transpose(-2, -3)    # (B,Heads,T,D)
        K = K.transpose(-2, -3)    # (B,Heads,T,D)
        V = V.transpose(-2, -3)
        # The Scissors (Attention)
        # we get exact order (Batch , Humber of heads, d_in(sequence),d_out(dimention))
        attention_score = Q @ K.transpose(-1,-2)    # (B,H,T,D)@(B,H,D,T) 
        scale = self.head_dim ** 0.5
        attention_scores = attention_score / scale
        # Softmax across the last dimension (the T dimension)
        Attention_weights = F.softmax(attention_scores, dim=-1)
        Context = Attention_weights @ V
        # The Glue (Put it back together)
        # Move Heads back to the middle: (B, Heads, T, Dim) -> (B, T, Heads, Dim)
        context_matrix = Context.transpose(-2, -3)

        # Flatten the Heads and Dim back together: (B, T, Heads, Dim) -> (B, T, 512)
        context_matrix = context_matrix.contiguous().view(B, T, self.d_model)
        output = self.Output(context_matrix)
        return output

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
num_heads = 2

VOACAB = len(word2index)

embedding = nn.Embedding(VOACAB, d_model)
attention = MultiHeadAttention( num_heads, d_model)

sentence = "how are you doing ?"
input_tensor = convt2tensor(sentence)
embeded = embedding(input_tensor)   

print(f"Embedding size: {embeded.size()}")  # torch.Size([1, 5, 10]) -> 1 batch, 5 tokens, 10 dims

# Set debug=False here so it doesn't spam your console, change to True if you need to debug
output = attention(embeded) 
print(f"Output contextual vector: {output}")