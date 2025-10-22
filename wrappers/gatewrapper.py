class GateWrapper:
    def __init__(self, gate, *args):
        self.gate = gate
        self.args = args
    
    def resolve(self, x, t):
        values = []
        for source, idx in self.args:
            if source == "x":
                values.append(x[idx])
            elif source == "t":
                values.append(t[idx])
                
        return self.gate(*values)

    @property
    def name(self):
        return self.gate.__name__