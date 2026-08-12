import torch
import torch.nn.functional as F

from src.model import ModelConfig, Transformer


def test_deterministic_initialization(tiny_cfg: ModelConfig):
    torch.manual_seed(42)
    m1 = Transformer(tiny_cfg)
    torch.manual_seed(42)
    m2 = Transformer(tiny_cfg)
    for (k1, p1), (k2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert k1 == k2
        assert torch.equal(p1.data, p2.data)


def test_init_flags_affect_init(tiny_cfg: ModelConfig):
    torch.manual_seed(1)
    m = Transformer(tiny_cfg)
    torch.manual_seed(1)
    n = Transformer(tiny_cfg)
    torch.manual_seed(2)
    o = Transformer(tiny_cfg)
    for p_q, p_o in zip(m.parameters(), o.parameters()):
        if p_q.numel() > 0:
            assert not torch.equal(p_q.data, p_o.data)
            break
    for p_q, p_n in zip(m.parameters(), n.parameters()):
        assert torch.equal(p_q.data, p_n.data)


def test_parameter_count_in_range():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(vocab_size=29, n_layer=2, d_model=256, n_head=4,
                                context_length=64, ffn_mult=4))
    n = m.n_params
    assert 1_000_000 <= n <= 5_000_000, f"param count {n} outside 1-5M"


def test_forward_shape(tiny_model: Transformer, toy_batch):
    x, _ = toy_batch
    B, T = x.shape
    logits = tiny_model(x)
    V = tiny_model.cfg.vocab_size
    assert logits.shape == (B, T, V)


def test_forward_deterministic(tiny_model: Transformer, toy_batch):
    x, _ = toy_batch
    a = tiny_model(x)
    b = tiny_model(x)
    assert torch.equal(a, b)


def test_causal_mask(tiny_model: Transformer, toy_batch):
    # output at position i must not depend on tokens after position i
    x, _ = toy_batch
    idx = 8
    x2 = x.clone()
    x2[:, idx:] = (x2[:, idx:] + 5) % tiny_model.cfg.vocab_size
    out1 = tiny_model(x)
    out2 = tiny_model(x2)
    assert torch.equal(out1[:, :idx], out2[:, :idx])
    assert not torch.equal(out1[:, idx:], out2[:, idx:])


def test_loss_matches_manual(tiny_model: Transformer, toy_batch):
    x, y = toy_batch
    logits = tiny_model(x)
    manual = F.cross_entropy(logits.reshape(-1, tiny_model.cfg.vocab_size),
                             y.reshape(-1))
    assert tiny_model.loss(logits, y).item() == manual.item()


def test_out_of_context_raises(tiny_model: Transformer):
    bad = torch.zeros(1, tiny_model.cfg.context_length + 1, dtype=torch.long)
    try:
        tiny_model(bad)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for over-long sequence")


def test_config_roundtrip(tiny_cfg: ModelConfig):
    d = tiny_cfg.to_dict()
    cfg2 = ModelConfig.from_dict(d)
    assert cfg2 == tiny_cfg