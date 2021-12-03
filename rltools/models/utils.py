

class LossSink:
    def __init__(self):
        self.total_loss = 0.

    def __add__(self, t):
        assert t.requires_grad is True
        self.total_loss += t
        return self

    def __call__(self):
        return self.total_loss

    def __repr__(self):
        return self.total_loss.__repr__()

    def zero_grad(self):
        self.total_loss = 0.
        return self

    def backward(self, *args, **kwargs):
        self.total_loss.backward(*args, **kwargs)
