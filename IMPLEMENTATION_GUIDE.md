# Implementation Guide - Next Upgrades

## Quick Implementation Examples

### 1. Subword Tokenization with tiktoken

```python
# Add to imports
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")

# In main() function, replace character tokenization:

if HAS_TIKTOKEN:
    # Use GPT-2 style tokenizer (50k vocab)
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode text
    data = np.array(enc.encode(sample_text))
    vocab_size = enc.n_vocab
    
    # For generation, use decoder
    def decode_tokens(token_ids):
        return enc.decode(token_ids.tolist())
    
    def encode_text(text):
        return np.array(enc.encode(text))
    
    print(f"Using subword tokenization: {vocab_size:,} tokens")
else:
    # Fallback to character-level
    chars = sorted(list(set(sample_text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = np.array([char_to_idx[ch] for ch in sample_text])
    
    def decode_tokens(token_ids):
        return ''.join([idx_to_char[i] for i in token_ids])
    
    def encode_text(text):
        return np.array([char_to_idx[ch] for ch in text])
```

### 2. Mixed Precision Training

```python
# Add to config
class Config:
    # ... existing config ...
    use_amp = True  # Automatic Mixed Precision
    grad_scale = 2.0 ** 16  # For gradient scaling

# Modify forward pass to use FP16
def forward_amp(self, idx, training=True):
    # Cast inputs to FP16
    if self.config.use_amp and training:
        # Convert to float16 for computation
        # Keep critical operations in float32
        pass

# Add gradient scaling before backward
def train(model, train_data, val_data, config):
    # ... existing training code ...
    
    if config.use_amp:
        # Scale loss to prevent underflow
        loss = loss * config.grad_scale
    
    model.backward(dlogits * config.grad_scale if config.use_amp else dlogits)
    
    # Unscale gradients for optimizer
    if config.use_amp:
        grad_norm = model.clip_gradients(config.grad_clip / config.grad_scale)
        # Divide gradients by scale before updating
    else:
        grad_norm = model.clip_gradients(config.grad_clip)
```

### 3. SwiGLU Activation

```python
def swish(x):
    """Swish activation: x * sigmoid(x)"""
    return x * (1 / (1 + np.exp(-x)))

def swiglu(x):
    """SwiGLU: Swish-Gated Linear Unit"""
    # Split input in half
    x1, x2 = np.split(x, 2, axis=-1)
    return x1 * swish(x2)

def swiglu_derivative(x):
    """Derivative for backprop"""
    x1, x2 = np.split(x, 2, axis=-1)
    sig = 1 / (1 + np.exp(-x2))
    swish_x2 = x2 * sig
    dswish = sig * (1 + x2 * (1 - sig))
    return np.concatenate([swish_x2, x1 * dswish], axis=-1)

# In FeedForward class:
def forward(self, x, training=True):
    """Forward pass with SwiGLU"""
    h = x @ self.W1 + self.b1
    # Use SwiGLU instead of GELU
    h_act = swiglu(h)  # Instead of gelu(h)
    h_act, drop_mask = dropout(h_act, self.dropout_p, training)
    out = h_act @ self.W2 + self.b2
    
    self.cache = {'x': x, 'h': h, 'h_act': h_act, 'drop_mask': drop_mask}
    return out

def backward(self, dout):
    """Backward pass with SwiGLU"""
    # ... existing code ...
    dh = dh_act * swiglu_derivative(h)  # Instead of gelu_derivative(h)
    # ... rest of backward pass ...
```

### 4. Gradient Checkpointing

```python
class TransformerBlock:
    def forward(self, x, training=True, checkpoint=False):
        """Forward with optional checkpointing"""
        if checkpoint and training:
            # Save only input, recompute later
            return self._forward_checkpointed(x, training)
        else:
            return self._forward_normal(x, training)
    
    def _forward_checkpointed(self, x, training):
        """Checkpointed forward - saves memory"""
        # Only save input, not intermediate activations
        # Recompute during backward
        pass

# In model forward:
def forward(self, idx, training=True, checkpoint=False):
    # ... existing code ...
    for block in self.blocks:
        x = block.forward(x, training, checkpoint=checkpoint)
    # ... rest ...
```

### 5. Better Weight Initialization

```python
def init_weights(layer_size, fan_in, init_type='xavier'):
    """Better weight initialization"""
    if init_type == 'xavier':
        scale = np.sqrt(2.0 / (fan_in + layer_size))
    elif init_type == 'kaiming':
        scale = np.sqrt(2.0 / fan_in)
    elif init_type == 'gpt_style':
        scale = 0.02  # GPT-style small initialization
    else:
        scale = 0.1
    
    return np.random.randn(*layer_size) * scale

# Use in model initialization:
self.W_q = init_weights((config.n_embd, config.n_embd), config.n_embd, 'xavier')
```

### 6. Learning Rate Finder

```python
def find_learning_rate(model, train_data, config, lr_min=1e-6, lr_max=1.0, num_steps=100):
    """Find optimal learning rate"""
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), num_steps)
    losses = []
    
    for lr in lrs:
        # One forward/backward step
        xb, yb = get_batch(train_data, config.block_size, config.batch_size)
        logits = model.forward(xb, training=True)
        loss, dlogits = cross_entropy_loss(logits, yb)
        model.backward(dlogits)
        model.update_params(lr)
        
        losses.append(loss)
    
    # Find steepest descent point
    best_idx = np.argmin(np.gradient(losses))
    best_lr = lrs[best_idx]
    
    return best_lr, lrs, losses
```

### 7. Data Loading Improvements

```python
def load_text_corpus(file_paths):
    """Load and process multiple text files"""
    all_text = ""
    
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Clean text
            text = clean_text(text)
            all_text += text + "\n"
    
    return all_text

def clean_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove special characters (optional)
    # Normalize unicode
    return text

# In main():
corpus_files = [
    "data/shakespeare.txt",
    "data/wikipedia_sample.txt",
    # ... more files
]

sample_text = load_text_corpus(corpus_files)
```

---

## Testing Checklist

After implementing each upgrade:

- [ ] Model still trains without errors
- [ ] Loss decreases during training
- [ ] Generated text makes sense
- [ ] No NaN or Inf values
- [ ] Memory usage is reasonable
- [ ] Training speed is acceptable
- [ ] Validation loss improves

---

## Performance Benchmarks

Track these metrics before/after each upgrade:

1. **Training Speed**: Tokens/second
2. **Memory Usage**: Peak GPU/RAM usage
3. **Loss Convergence**: How fast loss decreases
4. **Generation Quality**: Perplexity, coherence
5. **Context Length**: Max sequence length
6. **Model Size**: Number of parameters

---

## Troubleshooting

### If training becomes unstable:
- Check learning rate (might be too high)
- Verify gradient clipping is working
- Check for NaN values
- Reduce batch size or model size

### If out of memory:
- Use gradient checkpointing
- Reduce batch size
- Use mixed precision
- Reduce model size

### If training is slow:
- Enable mixed precision
- Use larger batch sizes
- Optimize data loading
- Consider distributed training

