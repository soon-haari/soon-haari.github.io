from z3 import *
import rbtree_mersenne
import random
import time
from tqdm import tqdm

def rbtree_solve(outputs):
    twister = rbtree_mersenne.Twister()
    solver = rbtree_mersenne.Solver()

    for tup in tqdm(outputs):
        res, bits = tup
        equation = twister.getrandbits(bits)

        for i in range(bits):
            solver.insert(equation[i], (res >> (bits - 1 - i)) & 1)

    state = solver.solve()
    return (3, tuple(state + [0]), None)



def z3_solve(outputs):
    MT = [BitVec(f'm{i}', 32) for i in range(624)]
    s = Solver()
    
    def tamper(y):
        y ^= LShR(y, 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return y ^ LShR(y, 18)
        
    def getnext():
        x = Concat(Extract(31, 31, MT[0]), Extract(30, 0, MT[1]))
        y = If(x & 1 == 0, BitVecVal(0, 32), 0x9908B0DF)
        MT.append(MT[397] ^ LShR(x, 1) ^ y)
        return tamper(MT.pop(0))
        
    def getrandbits(n):
        return Extract(31, 32 - n, getnext())
    
    for op in outputs:
        if op[1] == 0:
            getnext()
        else:
            s.add(getrandbits(op[1]) == op[0])
    assert(s.check() == sat)
    state = [s.model()[x].as_long() for x in [BitVec(f'm{i}', 32) for i in range(624)]]
    
    return (3, tuple(state + [0]), None)


if __name__ == "__main__":

    bits = 26
    number = 1248
    # change these values for different tests

    output = []
    for _ in range(number):
        output.append((random.getrandbits(bits), bits))
    check = []
    for _ in range(10000):
        check.append(random.getrandbits(bits))


    st = time.time()
    random.setstate(rbtree_solve(output))
    for _ in range(number):
        assert output[_][0] == random.getrandbits(bits)
    for _ in range(10000):
        assert check[_] == random.getrandbits(bits)
    fin = time.time()
    print(f"rbtree took {fin - st} seconds.")


    st = time.time()
    random.setstate(z3_solve(output))
    for _ in range(number):
        assert output[_][0] == random.getrandbits(bits)
    for _ in range(10000):
        assert check[_] == random.getrandbits(bits)
    fin = time.time()
    print(f"z3 took {fin - st} seconds.")