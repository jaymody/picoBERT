import json
import os
import re
import zipfile

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from tokenization import FullTokenizer

model_name_to_url = {
    "bert-tiny-uncased": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip",
    "bert-mini-uncased": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip",
    "bert-small-uncased": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip",
    "bert-medium-uncased": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip",
    "bert-base-uncased": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "bert-base-cased": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
    "bert-large-uncased": "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
    "bert-large-cased": "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",
    "bert-base-multilingual-cased": "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
}

params_key_map = {
    # block
    "attention/output/LayerNorm/gamma": "ln_1/g",
    "attention/output/LayerNorm/beta": "ln_1/b",
    "attention/output/dense/kernel": "attn/c_proj/w",
    "attention/output/dense/bias": "attn/c_proj/b",
    "attention/self/query/kernel": "attn/q/w",
    "attention/self/query/bias": "attn/q/b",
    "attention/self/key/kernel": "attn/k/w",
    "attention/self/key/bias": "attn/k/b",
    "attention/self/value/kernel": "attn/v/w",
    "attention/self/value/bias": "attn/v/b",
    "intermediate/dense/kernel": "mlp/c_fc/w",
    "intermediate/dense/bias": "mlp/c_fc/b",
    "output/dense/kernel": "mlp/c_proj/w",
    "output/dense/bias": "mlp/c_proj/b",
    "output/LayerNorm/gamma": "ln_2/g",
    "output/LayerNorm/beta": "ln_2/b",
    # top level
    "bert/embeddings/LayerNorm/gamma": "bert/ln_e/g",
    "bert/embeddings/LayerNorm/beta": "bert/ln_e/b",
    "bert/embeddings/position_embeddings": "bert/wpe",
    "bert/embeddings/word_embeddings": "bert/wte",
    "bert/embeddings/token_type_embeddings": "bert/wse",
    "bert/pooler/dense/kernel": "bert/pooler/w",
    "bert/pooler/dense/bias": "bert/pooler/b",
    "cls/predictions/output_bias": "mlm/bias",
    "cls/predictions/transform/dense/kernel": "mlm/fc/w",
    "cls/predictions/transform/dense/bias": "mlm/fc/b",
    "cls/predictions/transform/LayerNorm/gamma": "mlm/ln/g",
    "cls/predictions/transform/LayerNorm/beta": "mlm/ln/b",
    "cls/seq_relationship/output_weights": "nsp/fc/w",
    "cls/seq_relationship/output_bias": "nsp/fc/b",
}


def download_bert_files(model_name, model_dir):
    url = model_name_to_url[model_name]
    zip_fpath = os.path.join(model_dir, os.path.basename(url))

    r = requests.get(url, stream=True)
    r.raise_for_status()
    file_size = int(r.headers["content-length"])
    chunk_size = 1000
    with open(zip_fpath, "wb") as f:
        with tqdm(ncols=100, desc=f"downloading zip files.", total=file_size, unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)

    with zipfile.ZipFile(zip_fpath) as f:
        # we do this hack instead of simply f.extractall(model_dir) since the older
        # zipfiles released by google are nested inside a folder
        for finfo in f.infolist():
            if finfo.filename[-1] == "/":
                continue
            finfo.filename = os.path.basename(finfo.filename)
            f.extract(finfo, model_dir)


def load_bert_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"bert": {"blocks": [{} for _ in range(hparams["num_hidden_layers"])]}}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        if name.startswith("bert/encoder/layer_"):
            m = re.match(r"bert/encoder/layer_([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = params_key_map[m[2]]
            set_in_nested_dict(params["bert"]["blocks"][n], sub_name.split("/"), array)
        else:
            name = params_key_map[name]
            set_in_nested_dict(params, name.split("/"), array)

    # combine the q, k, v weights and biases
    for i, block in enumerate(params["bert"]["blocks"]):
        attn = block["attn"]
        q, k, v = attn.pop("q"), attn.pop("k"), attn.pop("v")
        params["bert"]["blocks"][i]["attn"]["c_attn"] = {}
        params["bert"]["blocks"][i]["attn"]["c_attn"]["w"] = np.concatenate([q["w"], k["w"], v["w"]], axis=-1)
        params["bert"]["blocks"][i]["attn"]["c_attn"]["b"] = np.concatenate([q["b"], k["b"], v["b"]], axis=-1)

    # we need to transpose the following parameter
    params["nsp"]["fc"]["w"] = params["nsp"]["fc"]["w"].T

    return params


def load_tokenizer_hparams_and_params(model_name, models_dir):
    assert model_name in model_name_to_url

    is_uncased = "uncased" in model_name

    model_dir = os.path.join(models_dir, model_name)
    hparams_path = os.path.join(model_dir, "bert_config.json")
    vocab_path = os.path.join(model_dir, "vocab.txt")
    tf_ckpt_path = os.path.join(model_dir, "bert_model.ckpt")

    if not os.path.isfile(hparams_path):  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_bert_files(model_name, model_dir)

    tokenizer = FullTokenizer(vocab_path, do_lower_case=is_uncased)
    hparams = json.load(open(hparams_path))
    params = load_bert_params_from_tf_ckpt(tf_ckpt_path, hparams)

    return tokenizer, hparams, params


def tokenize(tokenizer, text_a, text_b=None, mask_prob=0.0):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b) if text_b else []

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] + [0] * len(tokens_a) + [0] + [1] * len(tokens_b) + [1]

    masked_input_ids = input_ids[:]
    masked_tokens = tokens[:]
    for i in np.random.choice(len(tokens), int(len(tokens) * mask_prob), replace=False):
        if tokens[i] in [["[CLS]", "[SEP]"]]:  # skip for cls or sep tokens
            continue
        masked_tokens[i] = "[MASK]"
        masked_input_ids[i] = tokenizer.vocab["[MASK]"]

    return tokens, input_ids, segment_ids, masked_tokens, masked_input_ids
