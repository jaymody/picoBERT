import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps=1e-3):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v):
    return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    out_heads = [attention(q, k, v) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = layer_norm(x + mha(x, **attn, n_head=n_head), **ln_1)
    x = layer_norm(x + ffn(x, **mlp), **ln_2)
    return x

def bert(input_ids, segment_ids, wte, wpe, wse, ln_e, blocks, pooler, n_head):
    x = wte[input_ids] + wpe[range(len(input_ids))] + wse[segment_ids]
    x += layer_norm(x, **ln_e)

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    seq_output = x
    pooled_output = np.tanh(linear(x[0], **pooler))
    return seq_output, pooled_output
