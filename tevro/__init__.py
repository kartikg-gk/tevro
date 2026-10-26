"""Tevro: a small decision model. Give it a state, a question and options; it returns a probability per option."""

__version__ = "0.1.2"
__all__ = ["DEFAULT_MODEL", "Decider", "load"]


def __getattr__(name):
    # Lazy, so `import tevro` and `tevro --help` do not load the model stack or the tokenizer
    if name in __all__:
        from tevro import predict

        return getattr(predict, name)
    raise AttributeError(f"module 'tevro' has no attribute {name!r}")
