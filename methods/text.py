"""Legacy runtime encoders are intentionally unavailable in optimized forward paths."""
class FrozenText:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use prepared_text.prepare_text before constructing the model; inline encoding is forbidden")
