import torch


class Sym:
    """Tiny symbolic expression tree for feature and parameter references."""

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __add__(self, other):
        return Sym("add", self, to_sym(other))

    def __sub__(self, other):
        return Sym("sub", self, to_sym(other))

    def __mul__(self, other):
        return Sym("mul", self, to_sym(other))

    def __truediv__(self, other):
        return Sym("fdiv", self, to_sym(other))

    def __neg__(self):
        return Sym("neg", self)

    def __radd__(self, other):
        return to_sym(other).__add__(self)

    def __rsub__(self, other):
        return to_sym(other).__sub__(self)
    def __rmul__(self, other):
        return to_sym(other).__mul__(self)

    def sin(self):
        """Wrap this expression in a sine node."""
        return Sym("sin", self)

    def cos(self):
        """Wrap this expression in a cosine node."""
        return Sym("cos", self)

    def eval(self, x, t)->torch.Tensor:
        """Evaluate the expression using feature tensor `x` and parameter tensor `t`."""
        if self.op == "x":
            return x[..., self.args[0]]
        elif self.op == "t":
            return t[..., self.args[0]]
        elif self.op == "add":
            return self.args[0].eval(x, t) + self.args[1].eval(x, t)
        elif self.op == "sub":
            return self.args[0].eval(x,t) - self.args[1].eval(x,t)
        elif self.op == "mul":
            return self.args[0].eval(x, t) * self.args[1].eval(x, t)
        elif self.op == "fdiv":
            return self.args[0].eval(x, t) / self.args[1].eval(x, t)
        elif self.op == "const":
            ref = t if t is not None else x
            return torch.as_tensor(self.args[0], dtype=ref.dtype, device=ref.device)
        elif self.op == "neg":
            return -self.args[0].eval(x, t)
        elif self.op == "sin":
            return torch.sin(self.args[0].eval(x, t))
        elif self.op == "cos":
            return torch.cos(self.args[0].eval(x, t))
        else:
            raise ValueError("Unknown op")

def to_sym(other):
    """Convert numbers and existing expressions into `Sym` objects."""
    if isinstance(other, Sym):
        return other
    else:
        return Sym("const", float(other))

class Xsym:
    """Index helper for input features, e.g. `Xsym()[0]`."""

    def __getitem__(self, idx):
        return Sym("x", idx)

class Tsym:
    """Index helper for trainable parameters, e.g. `Tsym()[0]`."""

    def __getitem__(self, idx):
        return Sym("t", idx)

pii = to_sym(torch.pi)


def sin(a):
    """Symbolic sine."""
    return to_sym(a).sin()

def cos(a):
    """Symbolic cosine."""
    return to_sym(a).cos()

__all__ = ["Xsym", "Tsym"]
