import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bert import bert
from utils import load_tokenizer_hparams_and_params, mask_tokens, tokenize


def main(
    dataset_name: str = "imdb",
    N: int = 1000,
    test_ratio: float = 0.2,
    model_name: str = "bert-base-uncased",
    models_dir: str = "models",
    seed: int = 123,
):
    np.random.seed(seed)

    # load tokenizer, hparams, and params
    tokenizer, hparams, params = load_tokenizer_hparams_and_params(model_name, models_dir)
    n_head = hparams["num_attention_heads"]
    max_len = hparams["max_position_embeddings"]

    # load dataset
    dataset = load_dataset(dataset_name, split="train").shuffle()

    # extract bert features
    X, y = [], []
    for text, label in tqdm(zip(dataset[:N]["text"], dataset[:N]["label"]), total=N):
        _, input_ids, segment_ids = tokenize(tokenizer, text)
        input_ids, segment_ids = input_ids[:max_len], segment_ids[:max_len]
        _, pooled_output = bert(input_ids, segment_ids, **params["bert"], n_head=n_head)
        X.append(pooled_output)
        y.append(label)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y)

    # train classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # predictions
    preds = classifier.predict(X_test)
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
