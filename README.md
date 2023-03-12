# picoBERT
Like [picoGPT](https://github.com/jaymody/picoGPT), but for [BERT](https://arxiv.org/pdf/1810.04805.pdf).

#### Dependencies
```bash
pip install -r requirements.txt
```
Tested on `Python 3.9.10`.

#### Usage
* `bert.py` contains the actual BERT model code.
* `utils.py` includes utility code to download, load, and tokenize stuff.
* `tokenization.py` includes BERT WordPiece tokenizer code.
* `pretrain_demo.py` code to demo BERT doing pre-training tasks (MLM and NSP).
* `classify_demo.py` code to demo training an SKLearn classifier using the BERT output embeddings as input. This is not the same as actually fine-tuning the BERT model.

To demo BERT on pre-training tasks:

```bash
python pretrain_demo.py \
    --text_a "The apple doesn't fall far from the tree." \
    --text_b "Instead, it falls on Newton's head." \
    --model_name "bert-base-uncased" \
    --mask_prob 0.20
```

Which outputs:

```text
mlm_accuracy = 0.75
is_next_sentence = True
```

If we add the `--verbose` flag, we can also see where the model went wrong with masked language modeling:

```text
input = ['[CLS]', 'the', 'apple', 'doesn', "'", '[MASK]', 'fall', 'far', 'from', 'the', 'tree', '.', '[SEP]', 'instead', ',', 'it', 'falls', 'on', '[MASK]', "'", '[MASK]', '[MASK]', '.', '[SEP]']

actual: t
pred: t

actual: newton
pred: one

actual: s
pred: s

actual: head
pred: head
```

Instead of predicting the word "newton", it predicted the word "one", which still gives a valid sentence "Instead, it falls on one's head.".

For a demo of training an SKLearn classifier for the [IMDB dataset](https://huggingface.co/datasets/imdb), using BERT output embeddings as input to the classifier:
```bash
python classify_demo.py
    dataset_name "imdb" \
    N 1000 \
    test_ratio 0.2 \
    model_name "bert-base-uncased" \
    models_dir "models"
```

Which outputs (note, it takes a while to run the BERT model and extract all the embeddings):

```text
              precision    recall  f1-score   support

           0       0.78      0.85      0.81       104
           1       0.82      0.74      0.78        96

    accuracy                           0.80       200
   macro avg       0.80      0.79      0.79       200
weighted avg       0.80      0.80      0.79       200
```

Not bad, 80% accuracy using only 800 training examples and a simple SKLearn model. Of course, fine-tuning the entire model over all the training examples would yield much better results.
