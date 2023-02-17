# picoBERT
Like [picoGPT](https://github.com/jaymody/picoGPT), but for [BERT](https://arxiv.org/pdf/1810.04805.pdf).

#### Dependencies
```bash
pip install -r requirements.txt
```
Tested on `Python 3.9.10`.

#### Usage
```bash
python bert.py \
    --text_a "In a hole in the ground there lived a hobbit." \
    --text_b "Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort." \
    --model_name "bert-base-uncased" \
    --mask_prob 0.15
```

Which outputs:

```python
mlm_accuracy = 0.35714285714285715
is_next_sentence = True
```
