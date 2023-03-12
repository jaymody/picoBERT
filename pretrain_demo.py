import numpy as np

from bert import bert, layer_norm, linear
from utils import load_tokenizer_hparams_and_params, mask_tokens, tokenize


def mask_tokens(tokenizer, input_ids, mask_prob):
    CLS_ID, SEP_ID, MASK_ID = tokenizer.vocab["[CLS]"], tokenizer.vocab["[SEP]"], tokenizer.vocab["[MASK]"]

    masked_indices = np.random.choice(len(input_ids), int(len(input_ids) * mask_prob), replace=False)
    masked_indices = list(filter(lambda i: i not in {CLS_ID, SEP_ID}, masked_indices))  # dont mask cls or sep

    masked_input_ids = input_ids[:]
    for i in masked_indices:
        masked_input_ids[i] = MASK_ID

    return masked_input_ids, masked_indices


def nsp_head(pooled_output, fc):
    return linear(pooled_output, **fc)


def mlm_head(seq_output, wte, fc, ln, bias):
    x = linear(seq_output, **fc)
    x = layer_norm(x, **ln)
    return linear(x, wte.T, bias)


def main(
    text_a: str,
    text_b: str = None,
    model_name: str = "bert-base-uncased",
    models_dir: str = "models",
    mask_prob: float = 0.15,
    seed: int = 123,
    verbose: bool = False,
):
    np.random.seed(seed)

    tokenizer, hparams, params = load_tokenizer_hparams_and_params(model_name, models_dir)

    tokens, input_ids, segment_ids = tokenize(tokenizer, text_a, text_b)
    masked_input_ids, masked_indices = mask_tokens(tokenizer, input_ids, mask_prob)

    assert len(input_ids) <= hparams["max_position_embeddings"]
    seq_output, pooled_output = bert(
        masked_input_ids,
        segment_ids,
        **params["bert"],
        n_head=hparams["num_attention_heads"],
    )

    if mask_prob > 0:
        mlm_logits = mlm_head(seq_output, params["bert"]["wte"], **params["mlm"])
        correct = [input_ids[i] == np.argmax(mlm_logits[i]) for i in masked_indices]

        if verbose:
            preds = np.argmax(mlm_logits, axis=-1)
            print(f"input = {tokenizer.convert_ids_to_tokens(masked_input_ids)}\n")
            for i in sorted(masked_indices):
                print(f"actual: {tokens[i]}\npred: {tokenizer.inv_vocab[preds[i]]}\n")

        print(f"mlm_accuracy = {sum(correct)}/{len(correct)} = {sum(correct)/len(correct)} ")
    if text_b:
        nsp_logits = nsp_head(pooled_output, **params["nsp"])
        print(f"is_next_sentence = {np.argmax(nsp_logits) == 0}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
