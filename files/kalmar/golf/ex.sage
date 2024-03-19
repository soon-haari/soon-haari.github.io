from pwn import *
from tqdm import trange
import time

io = remote("mathgolf.chal-kalmarc.tf", "3470")

def recv():
	io.recvuntil(b"= ")
	return int(io.recvline(), 16)

def send(v):
	for i in range(2):
		io.sendline(hex(int(v[i])).encode())
		
for rnd in trange(100):
	st_recv = time.time()
	b, c, a0, a1, p = [recv() for _ in range(5)]
	en_recv = time.time()
	print(f"recv: {en_recv - st_recv:.3f}")

	st_calc = time.time()
	Fp = GF(p)

	rfp.<t> = PolynomialRing(Fp)

	for i in range(1, 100):
		if kronecker(-i, p) != 1:
			break
	mod = t^2 + i
	

	Fp2.<t> = GF(p^2, modulus=mod)
	F.<x> = PolynomialRing(Fp2)
	poly = ((b + x) * x - c)
	roots = poly.roots()
	r1 = roots[0][0]
	r2 = roots[1][0]

	send(mod)
	send(r1 + b)
	send(r2 + b)
	send((a1 + r1 * a0) / (r1 - r2))
	send((a1 + r2 * a0) / (r1 - r2))

	en_calc = time.time()
	print(f"calc: {en_calc - st_calc:.3f}")

io.interactive()