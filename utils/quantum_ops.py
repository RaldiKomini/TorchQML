from qml_lib.utils import backend as b

def density_matrix(mystate):
    p1 = b.outer(mystate, b.conjugate(mystate))
    return p1