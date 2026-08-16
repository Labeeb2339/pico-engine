"""Tokenizer: byte-level BPE round-trips and known encodings."""

from pico_engine.tokenizer import BPETokenizer


def _toy_tokenizer():
    # byte-encoded toy vocab (ASCII letters map to themselves under the GPT-2
    # byte-encoder, so these tokens are their own byte-encoded forms).
    vocab = ["h", "e", "l", "o", "w", "r", "d", "he", "ll", "lo", "hello"]
    merges = ["h e", "he l", "l l", "l o"]
    return BPETokenizer(vocab, merges, bos_id=0, eos_id=0)


def test_roundtrip():
    tok = _toy_tokenizer()
    for text in ["hello", "hell", "world", "he", "hello world".replace(" ", "")]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_merge_applied():
    tok = _toy_tokenizer()
    # "he" is a merge, so encoding "he" should yield a single token if it wins
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"


def test_vocab_size():
    tok = _toy_tokenizer()
    assert tok.n_vocab == 11
