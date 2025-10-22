from qml_lib.utils import backend as be  # assuming backend.py is in the same folder

def test_tensor_creation():
    print("Testing tensor creation...")
    t = be.tensor([[1, 2], [3, 4]], dtype=be.dtype_complex)
    print("Tensor:\n", t)
    print("Type:", type(t))
    assert t.shape == (2, 2), "Shape mismatch"
    assert t.dtype == be.tensor([1]).dtype, "Dtype mismatch"
    print("Tensor creation test passed!\n")

def test_matmul():
    print("Testing matrix multiplication...")
    a = be.tensor([[1, 0], [0, 1]], dtype=be.dtype_complex)
    b = be.tensor([[0, 1], [1, 0]], dtype=be.dtype_complex)
    c = be.matmul(a, b)
    print("Result of matmul:\n", c)
    assert c.shape == (2, 2), "Shape mismatch"
    print("Matmul test passed!\n")

def test_kron():
    print("Testing Kronecker product...")
    a = be.tensor([[1, 0], [0, 1]], dtype=be.dtype_complex)
    b = be.tensor([[0, 1], [1, 0]], dtype=be.dtype_complex)
    c = be.kron(a, b)
    print("Result of kron:\n", c)
    expected_shape = (a.shape[0]*torch.shape[0], a.shape[1]*torch.shape[1])
    assert c.shape == expected_shape, "Shape mismatch"
    print("Kron test passed!\n")

def test_functions():
    print("Testing other functions...")
    x = be.tensor([0, 3.14159/2], dtype=be.dtype_complex)
    s = be.sin(x)
    c = be.cos(x)
    print("sin(x):", s)
    print("cos(x):", c)
    assert abs(be.abs(s[0] - 0)) < 1e-6, "Sin(0) should be 0"
    print("Functions test passed!\n")

def main():
    print("Running backend tests...\n")
    test_tensor_creation()
    test_matmul()
    test_kron()
    test_functions()
    print("All backend tests passed successfully!")

if __name__ == "__main__":
    main()
