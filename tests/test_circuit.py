from qml_lib.circuits.circuit import Circuit
from qml_lib.gates.gate import Gate as g
from qml_lib.utils import backend as b
from qml_lib.gates import expand as e
from qml_lib.wrappers.gatewrapper import GateWrapper
from qml_lib.gates.gate import Rotx, Roty, Rotz

c = Circuit(num_qubits = 1)

#print(c.state)

#c.add_gates([g.H, g.I])
#state2 = c.run()
#print(state2)
#c.add_gates(g.CNOT)


#final_state = c.run()
#print(final_state)

#c.reset(gates=1)
#print(c)


state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.cfloat)

x = [torch.tensor(0.3), torch.tensor(0.7)]
t = [torch.tensor(1.0), torch.tensor(0.2)]

# Create wrappers that pull values from x and t
# Qubit 0 will get Rotx(x[0]), Qubit 1 will get Roty(t[0])
gw1 = GateWrapper(Rotx, ("x", 0))
gw2 = GateWrapper(Roty, ("t", 0))

# Another layer: Rotz(t[1]) and Rotx(x[1])
gw3 = GateWrapper(Rotz, ("t", 1))
gw4 = GateWrapper(Rotx, ("x", 1))

# Build a 2-qubit circuit
circuit = Circuit(num_qubits=2)

# Add two layers: each a list of 2 GateWrapper instances
circuit.add_gates([gw1, gw2])
circuit.add_gates([gw3, gw4])

# Apply the circuit to the initial state
out = circuit.apply_to(state, x=x, t=t)

print("Final state:\n", out)

# Optional: manually verify one layer
#expected_manual torch.tensor_product([Rotx(x[0]), Roty(t[0])])
#manual_result = expected_manual(state)
#print("\nManual result after first layer:\n", manual_result)



