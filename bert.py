import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-3):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + eps) + beta


def linear(x, kernel, bias):
    return x @ kernel + bias


def ffn(x, intermediate, output):
    x = linear(x, **intermediate["dense"])
    x = gelu(x)
    x = linear(x, **output["dense"])
    return x


def attention(q, k, v):
    return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v


def mha(x, self, output, n_head):
    q, k, v = linear(x, **self["query"]), linear(x, **self["key"]), linear(x, **self["value"])
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), [q, k, v]))
    out_heads = [attention(q, k, v) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **output["dense"])
    return x


def encoder_block(x, attention, intermediate, output, n_head):
    x = layer_norm(x + mha(x, **attention, n_head=n_head), **attention["output"]["LayerNorm"])
    x = layer_norm(x + ffn(x, intermediate, output), **output["LayerNorm"])
    return x


def bert(input_ids, token_type_ids, params, n_head, nsp=False, mlm=False):
    output = {"embeddings": []}

    # embeddings
    x = params["bert"]["embeddings"]["word_embeddings"][input_ids]
    x += params["bert"]["embeddings"]["token_type_embeddings"][token_type_ids]
    x += params["bert"]["embeddings"]["position_embeddings"][range(len(input_ids))]
    x += layer_norm(x, **params["bert"]["embeddings"]["LayerNorm"])

    # encoder stack
    for block in params["bert"]["encoder"]:
        x = encoder_block(x, **block, n_head=n_head)
        output["embeddings"].append(x)

    # next sentence prediction
    if nsp:
        _x = np.tanh(linear(x[0], **params["bert"]["pooler"]["dense"]))
        output["nsp"] = linear(
            _x,
            params["cls"]["seq_relationship"]["output_weights"].T,
            params["cls"]["seq_relationship"]["output_bias"],
        )

    # masked language modeling
    if mlm:
        _x = linear(x, **params["cls"]["predictions"]["transform"]["dense"])
        _x = layer_norm(_x, **params["cls"]["predictions"]["transform"]["LayerNorm"])
        output["mlm"] = _x @ params["bert"]["embeddings"]["word_embeddings"].T

    return output


def tokenize(tokenizer, text_a, text_b=None):
    tokens = ["[CLS]"] + tokenizer.tokenize(text_a) + ["[SEP]"]
    token_type_ids = [0] * len(tokens)

    if text_b is not None:
        tokens_b = tokenizer.tokenize(text_b) + ["[SEP]"]
        tokens += tokens_b
        token_type_ids += [1] * len(tokens_b)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, input_ids, token_type_ids


def main(
    text_a: str,
    text_b: str = None,
    model_name: str = "uncased_L-12_H-768_A-12",
    models_dir: str = "models",
):
    from utils import load_tokenizer_config_and_params

    tokenizer, config, params = load_tokenizer_config_and_params(model_name, models_dir)
    tokens, input_ids, token_type_ids = tokenize(tokenizer, text_a, text_b)

    assert len(input_ids) <= config["max_position_embeddings"]

    output = bert(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        params=params,
        n_head=config["num_attention_heads"],
        nsp=True,
        mlm=True,
    )

    return "yes" if np.argmax(output["nsp"]) == 0 else "no"


if __name__ == "__main__":
    import fire

    fire.Fire(main)
