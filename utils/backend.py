use_torch = True

if use_torch:
    import torch as xp
    dtype_complex = xp.complex64

    def tensor(data, dtype=None, device=None):
        if dtype is None:
            dtype = dtype_complex
        t = xp.tensor(data, dtype=dtype)
        if device:
            t = t.to(device)
        return t

    matmul = xp.matmul
    conj = xp.conj
    transpose = xp.transpose
    reshape = xp.reshape
    sin = xp.sin
    cos = xp.cos
    exp = xp.exp
    real = xp.real
    imag = xp.imag
    abs = xp.abs
    sum = xp.sum
    mean = xp.mean
    stack = xp.stack
    trace = xp.trace
    eye = lambda n: xp.eye(n, dtype=dtype_complex)
    zeros = lambda shape: xp.zeros(shape, dtype=dtype_complex)
    ones = lambda shape: xp.ones(shape, dtype=dtype_complex)

    # PyTorch doesn't have kron natively, so define it
    def kron(a, b):
        a_dim = a.dim()
        b_dim = b.dim()
        a_reshape = a.reshape(-1, 1)
        b_reshape = b.reshape(1, -1)
        return (a_reshape @ b_reshape).reshape(a.size() + b.size()).permute(
            *range(0, a_dim), *(range(a_dim, a_dim + b_dim))
        ).reshape(a.size(0) * b.size(0), a.size(1) * b.size(1))
    
else:
    import numpy as xp
    dtype_complex = complex

    def tensor(data, dtype=None):
        if dtype is None:
            dtype = dtype_complex
        return xp.array(data, dtype=dtype)

    matmul = xp.matmul
    kron = xp.kron
    conj = xp.conj
    transpose = xp.transpose
    reshape = xp.reshape
    sin = xp.sin
    cos = xp.cos
    exp = xp.exp
    real = xp.real
    imag = xp.imag
    abs = xp.abs
    sum = xp.sum
    mean = xp.mean
    stack = xp.stack
    trace = xp.trace
    eye = lambda n: xp.eye(n, dtype=dtype_complex)
    zeros = lambda shape: xp.zeros(shape, dtype=dtype_complex)
    ones = lambda shape: xp.ones(shape, dtype=dtype_complex)
