#!/usr/bin/env python3

# MathGolf-Warmup challenge by shalaamum for KalmarCTF 2024

import signal
import sys
import time
import sage.all

def out_of_time(signum, frame):
    print("\nTime is up!")
    sys.exit(1)

signal.signal(signal.SIGALRM, out_of_time)
signal.alarm(60)


def sequence_slow(n, b, c, a0, a1, p):
    if n == 0:
        return a0
    elif n == 1:
        return a1
    else:
        return (b*sequence(n - 1, b, c, a0, a1, p) + c*sequence(n - 2, b, c, a0, a1, p)) % p

from lib import sequence_fast
sequence = sequence_fast

from lib import ProblemGenerator
generator = ProblemGenerator()

def get_number():
    return int(input().strip()[2:], 16)


def sequence_from_parameters(n, b, c, a0, a1, p, parameters):
    poly = parameters[0:2]
    phi = parameters[2:4]
    psi = parameters[4:6]
    const_phi = parameters[6:8]
    const_psi = parameters[8:10]
    
    Fp = sage.all.GF(p)
    RFp = sage.all.PolynomialRing(Fp, ['t'])
    F = sage.all.GF(p**2, name='t', modulus=RFp(poly + [1]))
    phi = F(phi)
    psi = F(psi)
    const_phi = F(const_phi)
    const_psi = F(const_psi)

    answer = list(phi**n * const_phi - psi**n * const_psi)
    if answer[1] != 0:
        print("That can't be right...")
        sys.exit(1)
    return int(answer[0])

for i in range(100):
    print(f'Solved {i} of 100')
    n, b, c, a0, a1, p  = generator.get()
    print(f'b  = 0x{b:016x}')
    print(f'c  = 0x{c:016x}')
    print(f'a0 = 0x{a0:016x}')
    print(f'a1 = 0x{a1:016x}')
    print(f'p  = 0x{p:016x}')

    parameters = []
    print('Polynomial: ')
    parameters.append(get_number())
    parameters.append(get_number())
    print('phi: ')
    parameters.append(get_number())
    parameters.append(get_number())
    print('psi: ')
    parameters.append(get_number())
    parameters.append(get_number())
    print('const_phi: ')
    parameters.append(get_number())
    parameters.append(get_number())
    print('const_psi: ')
    parameters.append(get_number())
    parameters.append(get_number())

    print('Checking...')
    answer = sequence_from_parameters(n, b, c, a0, a1, p, parameters)
    correct = sequence(n, b, c, a0, a1, p)
    if answer != correct:
        print(f'Incorrect! Correct answer was 0x{correct:016x}')
        sys.exit(1)

print(open('flag.txt', 'r').read())
