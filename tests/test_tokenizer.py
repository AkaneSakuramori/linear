from src.tokenizer import CharTokenizer, SPECIAL_TOKENS


def test_encode_decode_roundtrip(tokenizer: CharTokenizer, text: str):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_vocab_has_specials(tokenizer: CharTokenizer):
    for s in SPECIAL_TOKENS:
        assert s in tokenizer.stoi
        assert s in tokenizer.itos.values()
    assert tokenizer.vocab_size == len(tokenizer.itos)
    assert tokenizer.vocab_size == len(tokenizer.stoi)


def test_all_corpus_chars_known(tokenizer: CharTokenizer, text: str):
    ids = tokenizer.encode(text)
    assert tokenizer.unk_id not in ids


def test_unknown_char_maps_to_unk(tokenizer: CharTokenizer):
    assert tokenizer.encode("zzz") == [tokenizer.unk_id] * 3 \
        if "z" not in tokenizer.chars else True
    # every character in the raw text maps to a known id
    unknown = set(SPECIAL_TOKENS)
    bad = [ch for ch in "xyz" if ch not in tokenizer.itos]
    for ch in bad:
        assert tokenizer.unk_id == tokenizer.stoi[tokenizer.itos[tokenizer.unk_id]]


def test_deterministic(tokenizer: CharTokenizer, text: str):
    other = CharTokenizer(text)
    assert tokenizer.vocab_size == other.vocab_size
    assert tokenizer.encode(text) == other.encode(text)


def test_save_load_roundtrip(tokenizer: CharTokenizer, text: str, tmp_path):
    p = str(tmp_path / "tok.json")
    tokenizer.save(p)
    loaded = CharTokenizer.load(p)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text