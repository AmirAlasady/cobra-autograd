
# this is js for development of the module that i decided to keep for the sake of cleanliness 
"""
import numpy as np

from activations import *
from conv import Conv2D, Flatten, MaxPool2D
from custom import BaseModel
from dense import Dense, parse_dtype
from loss import *
from optimizer import *
from sequential import Sequential
from tensor import Tensor, has_cupy
import cupy as cp # type: ignore


# Removed duplicate definition of Embedding class

class PositionalEncoding(BaseModel):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dtype=np.float32):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        
        # Initialize positional encoding matrix
        self.pos_enc = self._create_positional_encoding()
        self._modules['pos_enc'] = self.pos_enc  # Register buffer

    def _create_positional_encoding(self):
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(-(np.arange(0, self.d_model, 2) * np.log(10000.0) / self.d_model))
        
        pos_enc = np.zeros((self.max_seq_len, self.d_model), dtype=self.dtype)
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        
        return Tensor(pos_enc, 
                    requires_grad=False,
# Removed duplicate definition of PositionalEncoding class
        return self.forward(query, key, value, mask)

    def split_heads(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=self.dtype)
        if x.dtype != self.dtype:
            x = x.astype(self.dtype)
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)  # New shape: (batch, num_heads, seq, head_dim)

    def combine_heads(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=self.dtype)
        if x.dtype != self.dtype:
            x = x.astype(self.dtype)
        x = x.transpose(0, 2, 1, 3)  # Back to (batch, seq, num_heads, head_dim)
        return x.reshape(x.shape[0], -1, self.d_model)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        if not isinstance(q, Tensor):
            q = Tensor(q, dtype=self.dtype)
        if not isinstance(k, Tensor):
            k = Tensor(k, dtype=self.dtype)
        if not isinstance(v, Tensor):
# Removed duplicate definition of MultiHeadAttention class
        self.positional = PositionalEncoding(d_model,dtype=dtype)
        self.attention = MultiHeadAttention(d_model, num_heads, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x)
        x = self.positional(x)
        # Now works because attention expects 3 inputs
        return self.attention(x, x, x)  

# Instantiate and use
transformer = Transformer(vocab_size=10000, d_model=512, num_heads=8,dtype=np.float64)
input_ids = Tensor([[1, 2, 3], [4, 5, 0]], dtype=np.int64)
output = transformer(input_ids)
print('--------')
print(transformer.attention.device)  # Expected: cpu
print(transformer.attention.dtype)  # Expected: float32
print(output.device)  # Expected: cpu
print(output.dtype)  # Expected: float32

# Test with sample data
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 5

# Initialize
attn = MultiHeadAttention(d_model, num_heads)
x = Tensor(np.random.randn(batch_size, seq_len, d_model))

# Forward pass
output = attn(x, x, x)
print(output.shape)  # Should be (2, 5, 512)




# Full transformer component initialization
class Embedding(BaseModel):
    def __init__(self, vocab_size, d_model, dtype=np.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype
        # Xavier initialization using parent's device/dtype
        init_std = np.sqrt(2.0 / (vocab_size + d_model))
        weight_data = np.random.randn(vocab_size, d_model).astype(self.dtype) * init_std
        
        self.weight = Tensor(weight_data, 
                           device=self.device,  # Use BaseModel's device
                           dtype=self.dtype,
                           requires_grad=True)

    def forward(self, x):
        # Ensure input is Tensor and on correct device
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device, dtype=np.int64)
        else:
            x = x.to(self.device).astype(np.int64)  # Force device/dtype
        return self.weight[x]
    


class PositionalEncoding(BaseModel):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dtype=np.float32):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_enc = self._create_positional_encoding()
        self.dtype = dtype

    def _create_positional_encoding(self):
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(-(np.arange(0, self.d_model, 2) * np.log(10000.0) / self.d_model))
        
        pos_enc = np.zeros((self.max_seq_len, self.d_model), dtype=self.dtype)
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        
        return Tensor(pos_enc, 
                     device=self.device,
                     requires_grad=False,
                     dtype=self.dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Get proper numerical library
        xp = x.xp
        
        # Get positional encoding for sequence length
        seq_len = x.shape[1]
        pos_enc = self.pos_enc[:seq_len]
        
        # Reshape positional encoding for broadcasting
        # From (seq_len, d_model) to (1, seq_len, d_model)
        pos_enc = pos_enc.reshape(1, seq_len, self.d_model)
        
        # Expand to match batch dimension
        # Result shape: (batch_size, seq_len, d_model)
        pos_enc = xp.broadcast_to(pos_enc.data, (x.shape[0], seq_len, self.d_model))
        
        return x + Tensor(pos_enc, device=self.device, dtype=self.dtype)

class MultiHeadAttention(BaseModel):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dtype=np.float32):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.dtype = dtype
        # All layers inherit device/dtype from parent
        self.Wq = Dense(d_model, d_model,dtype=self.dtype)
        self.Wk = Dense(d_model, d_model,dtype=self.dtype)
        self.Wv = Dense(d_model, d_model,dtype=self.dtype)
        self.Wo = Dense(d_model, d_model,dtype=self.dtype)

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        return self.forward(query, key, value, mask)

    def split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)\
               .transpose(0, 2, 1, 3)

    def combine_heads(self, x: Tensor) -> Tensor:
        return x.transpose(0, 2, 1, 3)\
               .reshape(x.shape[0], -1, self.d_model)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.where(mask, -1e9)
        attn_weights = Softmax(axis=-1)(attn_scores)
        return attn_weights @ v

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        q = self.split_heads(self.Wq(query))
        k = self.split_heads(self.Wk(key))
        v = self.split_heads(self.Wv(value))
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        return self.Wo(self.combine_heads(attn_output))



class LayerNorm(BaseModel):
    def __init__(self, features: int, eps: float = 1e-5, dtype=np.float32):

        super().__init__()
        self.features = features
        self.eps = eps
        self.dtype = dtype

        # Learnable parameters
        self.gamma = Tensor(np.ones(features), 
                           dtype=dtype,
                           requires_grad=True)
        self.beta = Tensor(np.zeros(features),
                          dtype=dtype,
                          requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Ensure input is on correct device
        x = x.to(self.device).astype(self.dtype)
        
        # Calculate statistics
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True, unbiased=False)  # Use biased variance
        
        # Normalize
        x_normalized = (x - mean) / (var + self.eps).sqrt()
        #print('====================>')
        #print(self.gamma.dtype)
        #print(self.gamma.device)
        #print(x_normalized.dtype)
        #print(x_normalized.device)
        # Scale and shift
        return self.gamma * x_normalized + self.beta

    @property
    def parameters(self):

        return [self.gamma, self.beta]

    def state_dict(self):
        return {
            "gamma": self.gamma.data.copy(),
            "beta": self.beta.data.copy(),
            "features": self.features,
            "eps": self.eps,
            "device": self.device
        }

    def load_state_dict(self, state_dict):
        self.gamma.data = state_dict['gamma']
        self.beta.data = state_dict['beta']
        self.features = state_dict.get('features', self.features)
        self.eps = state_dict.get('eps', self.eps)

def test_layer_norm():
    # Test 1: Basic functionality on CPU
    ln_cpu = LayerNorm(features=64, dtype=np.float32)
    x_cpu = Tensor(np.random.randn(2, 10, 64), dtype=np.float32)
    out_cpu = ln_cpu(x_cpu)
    assert out_cpu.device == 'cpu', "CPU test failed device check"
    assert out_cpu.dtype == np.float32, "CPU test failed dtype check"

    # Test 2: Device propagation
    ln_gpu = LayerNorm(features=64, dtype=np.float32)
    ln_gpu.set_device('cuda')
    
    print("\nAfter set_device('cuda'):")
    print("Gamma device:", ln_gpu.gamma.device)  # Should be 'cuda'
    print("Beta device:", ln_gpu.beta.device)    # Should be 'cuda'
    
    x_gpu = Tensor(np.random.randn(2, 10, 64), device='cuda', dtype=np.float32)
    out_gpu = ln_gpu(x_gpu)
    assert out_gpu.device == 'cuda', "GPU test failed device check"

    # Test 3: Mixed device input
    x_mixed = Tensor(np.random.randn(2, 10, 64), dtype=np.float32)  # Default CPU
    out_mixed = ln_gpu(x_mixed)  # Should auto-convert to CUDA
    assert out_mixed.device == 'cuda', "Mixed device test failed"

    # Test 4: Dtype consistency
    ln_fp64 = LayerNorm(features=64, dtype=np.float64)
    ln_fp64.set_device('cuda')
    x_fp64 = Tensor(np.random.randn(2, 10, 64), device='cuda', dtype=np.float64)
    out_fp64 = ln_fp64(x_fp64)
    assert out_fp64.dtype == np.float64, "Float64 test failed"

    # Test 5: Gradient flow
    x = Tensor(np.random.randn(2, 10, 64), device='cuda', dtype=np.float32)
    x.requires_grad = True
    out = ln_gpu(x)
    loss = out.sum()
    loss.backward()
    
    assert ln_gpu.gamma.grad is not None, "Gamma gradients not calculated"
    assert ln_gpu.beta.grad is not None, "Beta gradients not calculated"
    assert ln_gpu.gamma.grad.device == 'cuda', "Gamma grad wrong device"
    assert ln_gpu.beta.grad.device == 'cuda', "Beta grad wrong device"

    print("\nAll LayerNorm tests passed!")

# Run the test
test_layer_norm()

print('========wdwdwd=======>')
bb=LayerNorm(512)
print(bb.gamma.device)
print(bb.beta.device)
print(bb.state_dict())
bb.set_device('cuda')
print(bb.gamma.device)
print(bb.beta.device)
print(bb.state_dict())
print('========wdwdwd=======>')
# Full encoder component initialization
# ==========================================================================================:>
class Encoder(BaseModel):
    def __init__(self, vocab_size, d_model, num_heads, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.embeddings = Embedding(vocab_size, d_model, dtype=self.dtype)
        self.positional = PositionalEncoding(d_model, dtype=self.dtype)
        self.attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm1 = LayerNorm(d_model, dtype=self.dtype)
        self.norm2 = LayerNorm(d_model, dtype=self.dtype)
        self.feedforward = Sequential([
            Dense(d_model, d_model * 4, dtype=self.dtype),
            ReLU(),
            Dense(d_model * 4, d_model, dtype=self.dtype)])

    def forward(self, x: Tensor) -> Tensor:
        # Embeddings and positional encoding
        x = self.embeddings(x)
        x = self.positional(x)
        
        # Attention block
        residual = x
        x = self.attention(x, x, x)
        x = residual + x
        x = self.norm1(x)
        
        # Feedforward block
        residual = x
        x = self.feedforward(x)
        x = residual + x
        x = self.norm2(x)
        
        return x
    


class Decoder(BaseModel):
    def __init__(self, vocab_size, d_model, num_heads, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.embeddings = Embedding(vocab_size, d_model, dtype=self.dtype)
        self.positional = PositionalEncoding(d_model, dtype=self.dtype)
        
        # Self attention with masking
        self.self_attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm1 = LayerNorm(d_model, dtype=self.dtype)
        
        # Encoder-Decoder attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm2 = LayerNorm(d_model, dtype=self.dtype)
        
        # Feedforward network
        self.feedforward = Sequential([
            Dense(d_model, d_model * 4, dtype=self.dtype),
            ReLU(),
            Dense(d_model * 4, d_model, dtype=self.dtype)
        ])
        self.norm3 = LayerNorm(d_model, dtype=self.dtype)

    def create_causal_mask(self, seq_len):
       
        xp = np if self.device == 'cpu' else cp
        mask = xp.ones((1, seq_len, seq_len), dtype=self.dtype)
        mask = xp.tril(mask)  # Lower triangular matrix
        return Tensor(mask, device=self.device, requires_grad=False)

    def __call__(self, target: Tensor, encoder_output: Tensor) -> Tensor:
        return self.forward(target, encoder_output)

    def forward(self, target: Tensor, encoder_output: Tensor) -> Tensor:
        # Embed target sequence
        x = self.embeddings(target)
        x = self.positional(x)
        
        # Self attention with causal masking
        residual = x
        seq_len = x.shape[1]
        mask = self.create_causal_mask(seq_len)
        x = self.self_attention(x, x, x, mask)
        x = residual + x
        x = self.norm1(x)
        
        # Encoder-decoder attention
        residual = x
        x = self.cross_attention(x, encoder_output, encoder_output)
        x = residual + x
        x = self.norm2(x)
        
        # Feedforward block
        residual = x
        x = self.feedforward(x)
        x = residual + x
        x = self.norm3(x)
        
        return x


class Transformer(BaseModel):
    def __init__(self, vocab_size, d_model, num_heads, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.embeddings = Embedding(vocab_size, d_model, dtype=self.dtype)
        self.positional = PositionalEncoding(d_model, dtype=self.dtype)
        self.attention = MultiHeadAttention(d_model, num_heads, dtype=self.dtype)
        self.norm = LayerNorm(d_model, dtype=self.dtype)  # Add layer norm
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x)
        x = self.positional(x)
        x = self.attention(x, x, x)
        return self.norm(x)  # Apply normalization 

transformer = Transformer(vocab_size=10000, d_model=512, num_heads=8, dtype=np.float64)
print("Before set_device:")
print(transformer.norm.gamma.device)  # cpu (default)
print(transformer.norm.beta.device)   # cpu

transformer.set_device('cuda')

print("\nAfter set_device:")
print(transformer.norm.gamma.device)  # cuda
print(transformer.norm.beta.device)   # cuda

# Instantiate model
transformer = Transformer(vocab_size=10000, d_model=512, num_heads=8,dtype=np.float64)
print('====================LL')
print(transformer.embeddings.weight.device)  # Expected: cuda
print(transformer.positional.pos_enc.device)  # Expected: cuda
print(transformer.attention.Wk.device)  # Expected: cuda
print(transformer.norm.gamma.device)  # Expected: cuda
print(transformer.norm.beta.device)  # Expected: cuda
print('====================>')
transformer.set_device('cuda')  # Propagates to all components
print(transformer.embeddings.weight.device)  # Expected: cuda
print(transformer.positional.pos_enc.device)  # Expected: cuda
print(transformer.attention.Wk.device)  # Expected: cuda
print(transformer.norm.gamma.device)  # Expected: cuda
print(transformer.norm.beta.device)  # Expected: cuda

# Create input on GPU
input_ids = Tensor([[1, 2, 3], [4, 5, 0]], 
                  dtype=np.int64,
                  device='cuda')  # Explicit device

# Forward pass
output = transformer(input_ids)
print(transformer.state_dict())





# Test backpropagation
def test_transformer_backprop():
    transformer = Transformer(vocab_size=1000, d_model=64, num_heads=4, dtype=np.float64)
    transformer.set_device('cuda')
    
    # Create test data
    input_ids = Tensor([[1, 2, 3, 0], [4, 5, 0, 0]], 
                     dtype=np.int64,
                     device='cuda')
    
    # Forward + backward pass
    output = transformer(input_ids)
    loss = output.mean()  # Simple scalar loss
    loss.backward()
    
    # Verify gradients
    print("Embedding grad shape:", transformer.embeddings.weight.grad.shape)
    print("Attention Wq grad shape:", transformer.attention.Wq.weights.grad.shape)
    print(transformer.state_dict())
test_transformer_backprop()


def test_transformer_backprop():
    transformer = Transformer(vocab_size=1000, d_model=64, num_heads=4, dtype=np.float64)
    transformer.set_device('cuda')
    
    # Create test data
    input_ids = Tensor([[1, 2, 3, 0], [4, 5, 0, 0]], 
                     dtype=np.int64,
                     device='cuda')
    
    # Forward + backward pass
    output = transformer(input_ids)
    loss = output.mean()
    loss.backward()
    
    # Verify layer norm gradients
    print("Gamma grad shape:", transformer.norm.gamma.grad.shape)  # Should be (64,)
    print("Beta grad shape:", transformer.norm.beta.grad.shape)    # Should be (64,)

test_transformer_backprop()



# Test backpropagation
def test_transformer_backprop():
    transformer = Transformer(vocab_size=1000, d_model=64, num_heads=4, dtype=np.float64)
    transformer.set_device('cpu')
    
    # Create test data
    input_ids = Tensor([[1, 2, 3, 0], [4, 5, 0, 0]], 
                     dtype=np.int64,
                     device='cpu')
    
    # Forward + backward pass
    output = transformer(input_ids)
    loss = output.mean()  # Simple scalar loss
    loss.backward()
    
    # Verify gradients
    print("Embedding grad shape:", transformer.embeddings.weight.grad.shape)
    print("Attention Wq grad shape:", transformer.attention.Wq.weights.grad.shape)
    print(transformer.state_dict())
test_transformer_backprop()


def test_transformer_backprop():
    transformer = Transformer(vocab_size=1000, d_model=64, num_heads=4, dtype=np.float64)
    transformer.set_device('cpu')
    
    # Create test data
    input_ids = Tensor([[1, 2, 3, 0], [4, 5, 0, 0]], 
                     dtype=np.int64,
                     device='cpu')
    
    # Forward + backward pass
    output = transformer(input_ids)
    loss = output.mean()
    loss.backward()
    
    # Verify layer norm gradients
    print("Gamma grad shape:", transformer.norm.gamma.grad.shape)  # Should be (64,)
    print("Beta grad shape:", transformer.norm.beta.grad.shape)    # Should be (64,)

test_transformer_backprop()













# Test with:
encoder = Encoder(vocab_size=10000, d_model=512, num_heads=8)

input_ids = Tensor([[1, 2, 3], [4, 5, 0]], dtype=np.int64,device='cuda')

print('========================================================================================mmmmmmmmmmmmmm')
print(encoder.embeddings.weight.device)  # Expected: cpu
print(encoder.positional.pos_enc.device)  # Expected: cpu
print(encoder.attention.Wk.device)  # Expected: cpu
print(encoder.norm1.gamma.device)  # Expected: cpu
print(encoder.norm1.beta.device)  # Expected: cpu
print(encoder.norm2.gamma.device)  # Expected: cpu
print(encoder.norm2.beta.device)  # Expected: cpu
print(encoder.feedforward.layers)  # Expected: cpu
print(encoder.state_dict())
print('<====================>')
encoder.set_device('cuda')  # Propagates to all components

print(encoder.embeddings.weight.device)  # Expected: cuda
print(encoder.positional.pos_enc.device)  # Expected: cuda
print(encoder.attention.Wk.device)  # Expected: cuda
print(encoder.norm1.gamma.device)  # Expected: cuda
print(encoder.norm1.beta.device)  # Expected: cuda
print(encoder.norm2.gamma.device)  # Expected: cuda
print(encoder.norm2.beta.device)  # Expected: cuda
print(encoder.feedforward.layers)  # Expected: cuda
print(encoder.state_dict())
output = encoder(input_ids)
print('<=><=>')
print(encoder.state_dict())
assert output.shape == (2, 3, 512), f"Bad shape: {output.shape}"
assert output.device == input_ids.device, "Device mismatch"
assert output.dtype == np.float32, "Dtype mismatch"


# First test the Encoder
def test_encoder():
    # Initialize with CPU
    encoder = Encoder(vocab_size=10000, d_model=512, num_heads=8)
    
    # Test initial device placement
    assert encoder.embeddings.weight.device == 'cpu', "Embeddings not on CPU"
    assert encoder.positional.pos_enc.device == 'cpu', "Positional encoding not on CPU"
    assert encoder.attention.Wq.weights.device == 'cpu', "Attention weights not on CPU"
    assert encoder.norm1.gamma.device == 'cpu', "LayerNorm gamma not on CPU"
    
    # Move to CUDA
    encoder.set_device('cuda')
    
    # Verify device propagation
    assert encoder.embeddings.weight.device == 'cuda', "Embeddings not moved to CUDA"
    assert encoder.positional.pos_enc.device == 'cuda', "Positional encoding not moved"
    assert encoder.attention.Wk.weights.device == 'cuda', "Attention weights not moved"
    assert encoder.norm2.beta.device == 'cuda', "LayerNorm beta not moved"
    
    # Create test input
    input_ids = Tensor([[1, 2, 3], [4, 5, 0]], dtype=np.int64, device='cuda')
    
    # Forward pass test
    output = encoder(input_ids)
    assert output.shape == (2, 3, 512), f"Bad shape: {output.shape}"
    assert output.device == 'cuda', "Output device mismatch"
    assert output.dtype == np.float32, f"Wrong dtype: {output.dtype}"
    
    # Backward pass test
    dummy_target = Tensor.randn(*output.shape, device='cuda')
    loss = (output - dummy_target).square().mean()
    loss.backward()
    
    # Verify gradients exist
    assert encoder.embeddings.weight.grad is not None, "No gradients in embeddings"
    assert encoder.attention.Wv.bias.grad is not None, "No attention gradients"
    assert encoder.norm1.gamma.grad is not None, "No LayerNorm gradients"
    
    print("Encoder tests passed!")

# Then test the Decoder
def test_decoder():
    # Initialize components
    decoder = Decoder(vocab_size=10000, d_model=512, num_heads=8)
    encoder = Encoder(vocab_size=10000, d_model=512, num_heads=8)
    
    # Move everything to CUDA
    decoder.set_device('cuda')
    encoder.set_device('cuda')
    
    # Create test inputs
    target_ids = Tensor([[5, 6, 7], [8, 9, 10]], dtype=np.int64, device='cuda')
    src_ids = Tensor([[1, 2, 3], [4, 5, 0]], dtype=np.int64, device='cuda')
    
    # Get encoder output
    encoder_output = encoder(src_ids)
    
    # Test decoder initialization
    assert decoder.embeddings.weight.device == 'cuda', "Decoder embeddings wrong device"
    assert decoder.cross_attention.Wq.weights.device == 'cuda', "Cross-attention not on CUDA"
    
    # Forward pass
    decoder_output = decoder(target_ids, encoder_output)
    assert decoder_output.shape == (2, 3, 512), f"Bad decoder shape: {decoder_output.shape}"
    assert decoder_output.device == 'cuda', "Decoder output device mismatch"
    
    # Backward pass
    dummy_target = Tensor.randn(*decoder_output.shape, device='cuda')
    loss = (decoder_output - dummy_target).square().mean()
    loss.backward()
    
    # Verify gradients
    assert decoder.embeddings.weight.grad is not None, "No decoder embedding gradients"
    assert decoder.cross_attention.Wk.weights.grad is not None, "No cross-attention gradients"
    assert decoder.norm3.gamma.grad is not None, "No final LayerNorm gradients"
    
    # Test causal masking
    batch_size, seq_len = 2, 5
    mask = decoder.create_causal_mask(seq_len)
    assert mask.shape == (1, seq_len, seq_len), f"Bad mask shape: {mask.shape}"
    assert mask.device == 'cuda', "Mask not on CUDA"
    
    # Verify triangular structure
    mask_data = mask.data
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask_data[0,i,j] == 0, "Mask not upper-triangular"
            else:
                assert mask_data[0,i,j] == 1, "Mask incorrectly applied"

    print("Decoder tests passed!")

# Run the tests
test_encoder()
test_decoder()



def test_decoder():
    decoder = Decoder(vocab_size=10000, d_model=512, num_heads=8)
    encoder = Encoder(vocab_size=10000, d_model=512, num_heads=8)
    
    # Test initialization
    assert isinstance(decoder.cross_attention, MultiHeadAttention), "Missing cross-attention"
    assert len(decoder.feedforward.layers) == 3, "Wrong FFN structure"
    
    # Test device propagation
    decoder.set_device('cuda')
    encoder.set_device('cuda')
    
    input_ids = Tensor([[1, 2, 3], [4, 5, 0]], dtype=np.int64, device='cuda')
    target_ids = Tensor([[5, 6, 7], [8, 9, 10]], dtype=np.int64, device='cuda')
    
    encoder_output = encoder(input_ids)
    decoder_output = decoder(target_ids, encoder_output)  # This should now work
    
    assert decoder_output.shape == (2, 3, 512), f"Bad shape: {decoder_output.shape}"
    print("Decoder forward pass successful!")
    
    # Test backward pass
    dummy_target = Tensor.randn(*decoder_output.shape, device='cuda')
    loss = (decoder_output - dummy_target).square().mean()
    loss.backward()
    
    assert decoder.embeddings.weight.grad is not None, "Missing embedding gradients"
    assert decoder.cross_attention.Wq.weights.grad is not None, "Missing attention gradients"
    print("Decoder backward pass successful!")

test_decoder()



class Transformer(BaseModel):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                 num_encoder_layers=6, num_decoder_layers=6, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.d_model = d_model
        
        # Encoder stack
        self.encoders = [
            Encoder(
                vocab_size=src_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                dtype=dtype
            ) for _ in range(num_encoder_layers)
        ]
        
        # Decoder stack
        self.decoders = [
            Decoder(
                vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                dtype=dtype
            ) for _ in range(num_decoder_layers)
        ]
        
        # Shared embeddings and final projection
        self.src_embed = Embedding(src_vocab_size, d_model, dtype=dtype)
        self.tgt_embed = Embedding(tgt_vocab_size, d_model, dtype=dtype)
        self.pos_enc = PositionalEncoding(d_model, dtype=dtype)
        self.final_proj = Dense(d_model, tgt_vocab_size, dtype=dtype)

    def __call__(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.forward(src, tgt)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # Encoder processing
        x = self.pos_enc(self.src_embed(src))
        for encoder in self.encoders:
            x = encoder(x)
        encoder_out = x
        
        # Decoder processing
        x = self.pos_enc(self.tgt_embed(tgt))
        for decoder in self.decoders:
            x = decoder(x, encoder_out)
        
        return self.final_proj(x)
    




    def set_device(self, device):
       # Propagate device setting to all components
        super().set_device(device)
        for encoder in self.encoders:
            encoder.set_device(device)
        for decoder in self.decoders:
            decoder.set_device(device)
        self.src_embed.set_device(device)
        self.tgt_embed.set_device(device)
        self.pos_enc.set_device(device)
        self.final_proj.set_device(device)




def test_transformer():
    # Initialize model
    transformer = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=15000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    
    # Initial device checks
    print("Initial device checks:")
    print(transformer.encoders[0].embeddings.weight.device)  # cpu
    print(transformer.decoders[0].cross_attention.Wk.device)  # cpu
    print(transformer.final_proj.weights.device)  # cpu
    
    # Move to CUDA
    transformer.set_device('cuda')
    print("\nAfter device change:")
    print(transformer.encoders[3].norm2.gamma.device)  # cuda
    print(transformer.decoders[5].feedforward.layers[0].weights.device)  # cuda
    print(transformer.pos_enc.pos_enc.device)  # cuda
    
    # Test forward pass
    src = Tensor([[1,2,3,0], [4,5,6,7]], dtype=np.int64, device='cuda')
    tgt = Tensor([[8,9,10,0], [11,12,13,14]], dtype=np.int64, device='cuda')
    output = transformer(src, tgt)
    
    print("\nOutput checks:")
    print(output.shape)  # (2, 4, 15000)
    print(output.device)  # cuda
    print(output.dtype)  # float32
    
    # Test backward pass
    dummy_target = Tensor.randn(2, 4, 15000, device='cuda')
    loss = (output - dummy_target).square().mean()
    loss.backward()
    
    print("\nGradient checks:")
    print(transformer.encoders[0].attention.Wq.weights.grad is not None)  # True
    print(transformer.decoders[3].norm3.beta.grad is not None)  # True
    print(transformer.final_proj.bias.grad is not None)  # True
    
    print("\nAll transformer tests passed!")

test_transformer()


"""