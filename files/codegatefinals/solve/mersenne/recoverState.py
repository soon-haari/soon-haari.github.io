from solver import Solver
from twister import Twister
import random
def recoverState(witch_spell):
    round = 1
    outputs = []
    bits = []
    num = 666
    witch_rand = random.Random()
    witch_rand.seed(int(witch_spell, 16))
    for i in range(num):
        witch_hand = witch_rand.randint(0, 2)
        my_hand = (witch_hand + 1) % 3
        outputs.append(my_hand)
        bits.append(2)

    twister = Twister()
    solver = Solver()
    e = []
    for i in range(num):
        e.append(twister.getrandbits(bits[i]))
    solver.insert(1 << 31,1) # set state's MSB
    print(f"Recovering State ... ")
    for i in range(num):
        eq = e[i]
        for j in range(bits[i]):
            solver.insert(eq[j], (outputs[i] >> (bits[i] - 1 - j)) & 1)

    state = solver.solve()
    recovered_state = (3, tuple(state + [624]), None)
    return recovered_state