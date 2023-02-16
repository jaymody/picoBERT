import json
import os
import re
import zipfile

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from tokenization import FullTokenizer


def download_bert_files(model_name, model_dir):
    filename = f"{model_name}.zip"
    url = f"https://storage.googleapis.com/bert_models/2020_02_20/{filename}"
    file_path = os.path.join(model_dir, filename)

    r = requests.get(url, stream=True)
    r.raise_for_status()
    file_size = int(r.headers["content-length"])
    chunk_size = 1000
    with open(file_path, "wb") as f:
        with tqdm(
            ncols=100, desc="download bert files.", total=file_size, unit_scale=True
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)

    with zipfile.ZipFile(file_path, "r") as f:
        f.extractall(model_dir)


def load_bert_params_from_tf_ckpt(tf_ckpt_path, config):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    init_vars = tf.train.list_variables(tf_ckpt_path)
    params = {"bert": {"encoder": [{} for _ in range(config["num_hidden_layers"])]}}
    for name, _ in init_vars:
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        if name.startswith("bert/encoder/layer_"):
            m = re.match(r"bert/encoder/layer_([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["bert"]["encoder"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_tokenizer_config_and_params(model_name, models_dir):
    # TODO assert model_name in [""]

    is_uncased = "uncased" in model_name

    model_dir = os.path.join(models_dir, model_name)
    config_path = os.path.join(model_dir, "bert_config.json")
    vocab_path = os.path.join(model_dir, "vocab.txt")
    tf_ckpt_path = os.path.join(model_dir, "bert_model.ckpt")

    if not os.path.isfile(config_path):  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_bert_files(model_name, model_dir)

    tokenizer = FullTokenizer(vocab_path, do_lower_case=is_uncased)
    config = json.load(open(config_path))
    params = load_bert_params_from_tf_ckpt(tf_ckpt_path, config)

    return tokenizer, config, params
