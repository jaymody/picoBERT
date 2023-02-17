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

    layer_embeddings = []
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
        layer_embeddings.append(x)

    pooled_embedding = np.tanh(linear(x[0], **pooler))
    return layer_embeddings, pooled_embedding


def nsp_head(x, fc):
    return linear(x, **fc)


def mlm_head(x, wte, fc, ln, bias):
    x = linear(x, **fc)
    x = layer_norm(x, **ln)
    return linear(x, wte.T, bias)


def main(
    text_a: str,
    text_b: str = None,
    model_name: str = "bert-base-uncased",
    models_dir: str = "models",
    mask_prob: float = 0.15,
    seed: int = 123,
):
    np.random.seed(seed)
    from utils import load_tokenizer_hparams_and_params, tokenize

    tokenizer, hparams, params = load_tokenizer_hparams_and_params(
        model_name,
        models_dir,
    )

    tokens, input_ids, segment_ids, masked_tokens, masked_input_ids = tokenize(
        tokenizer,
        text_a,
        text_b,
        mask_prob,
    )

    assert len(input_ids) <= hparams["max_position_embeddings"]
    layer_embeddings, pooled_embedding = bert(
        masked_input_ids if mask_prob > 0 else input_ids,
        segment_ids,
        **params["bert"],
        n_head=hparams["num_attention_heads"],
    )

    if mask_prob > 0:
        mlm_logits = mlm_head(layer_embeddings[-1], params["bert"]["wte"], **params["mlm"])
        correct = [
            input_ids[i] == np.argmax(mlm_logits[i])
            for i, token in enumerate(masked_tokens)
            if token == "[MASK]"
        ]
        print(f"mlm_accuracy = {sum(correct) / len(correct)}")

    if text_b:
        nsp_logits = nsp_head(pooled_embedding, **params["nsp"])
        print(f"is_next_sentence = {np.argmax(nsp_logits) == 0}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
