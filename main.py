import numpy as np

from bert import bert, mlm_head, nsp_head
from utils import load_tokenizer_hparams_and_params


def tokenize(tokenizer, text_a, text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b) if text_b else []

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    segment_ids = [0] + [0] * len(tokens_a) + [0] + [1] * len(tokens_b) + [1]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, input_ids, segment_ids


def mask_tokens(tokens, p):
    masked_tokens = tokens[:]
    for i in np.random.choice(len(tokens), int(len(tokens) * p), replace=False):
        if tokens[i] in [["[CLS]", "[SEP]"]]:  # skip for cls or sep tokens
            continue
        masked_tokens[i] = "[MASK]"
    return masked_tokens


def main(
    text_a: str,
    text_b: str = None,
    model_name: str = "bert-tiny-uncased",
    models_dir: str = "models",
    do_nsp: bool = True,
    do_mlm: bool = True,
    mask_prob: float = 0.15,
    seed: int = 123,
):
    np.random.seed(seed)

    tokenizer, hparams, params = load_tokenizer_hparams_and_params(model_name, models_dir)

    tokens, input_ids, segment_ids = tokenize(tokenizer, text_a, text_b)
    if do_mlm:
        masked_tokens = mask_tokens(tokens, mask_prob)
        input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    assert len(input_ids) <= hparams["max_position_embeddings"]
    layer_embeddings, pooled_embedding = bert(
        input_ids,
        segment_ids,
        **params["bert"],
        n_head=hparams["num_attention_heads"],
    )

    if do_mlm:
        mlm_logits = mlm_head(layer_embeddings[-1], params["bert"]["wte"], **params["mlm"])
        correct = [
            tokens[i] == tokenizer.inv_vocab[np.argmax(mlm_logits[i])]
            for i, token in enumerate(masked_tokens)
            if token == "[MASK]"
        ]
        print(f"mlm_accuracy = {sum(correct) / len(correct)}")

    if do_nsp and text_b:
        nsp_logits = nsp_head(pooled_embedding, **params["nsp"])
        print(f"is_next_sentence = {np.argmax(nsp_logits) == 0}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
