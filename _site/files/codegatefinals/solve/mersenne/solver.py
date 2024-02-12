class Solver:
    def __init__(self):
        self.equations = []
        self.outputs = []
        self.xors = []
    def insert(self, equation, output):
        for eq, o in zip(self.equations, self.outputs):
            lsb = eq & -eq
            if equation & lsb:
                equation ^= eq
                output ^= o
        
        if equation == 0:
            if output != 0:
                assert True == False

        lsb = equation & -equation
        self.xors.append(0)
        for i in range(len(self.equations)):
            if self.equations[i] & lsb:
                self.equations[i] ^= equation
                self.outputs[i] ^= output
                self.xors[i] ^= 1 << i
        self.equations.append(equation)
        self.outputs.append(output)
        return True
    
    def solve(self):
        num = 0
        for i, eq in enumerate(self.equations):
            if self.outputs[i]:
                # Assume every free variable is 0
                num |= eq & -eq
        
        state = [ (num >> (32 * i)) & 0xFFFFFFFF for i in range(624) ]
        return state