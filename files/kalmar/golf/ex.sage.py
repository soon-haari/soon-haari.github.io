

# This file was *autogenerated* from the file ex.sage
from sage.all_cmdline import *   # import sage library

_sage_const_16 = Integer(16); _sage_const_2 = Integer(2); _sage_const_100 = Integer(100); _sage_const_5 = Integer(5); _sage_const_1 = Integer(1); _sage_const_0 = Integer(0)
from pwn import *
from tqdm import trange
import time

io = remote("mathgolf.chal-kalmarc.tf", "3470")

def recv():
	io.recvuntil(b"= ")
	return int(io.recvline(), _sage_const_16 )

def send(v):
	for i in range(_sage_const_2 ):
		io.sendline(hex(int(v[i])).encode())
		
for rnd in trange(_sage_const_100 ):
	st_recv = time.time()
	b, c, a0, a1, p = [recv() for _ in range(_sage_const_5 )]
	en_recv = time.time()
	print(f"recv: {en_recv - st_recv:.3f}")

	st_calc = time.time()
	Fp = GF(p)

	rfp = PolynomialRing(Fp, names=('t',)); (t,) = rfp._first_ngens(1)

	for i in range(_sage_const_1 , _sage_const_100 ):
		if kronecker(-i, p) != _sage_const_1 :
			break
	mod = t**_sage_const_2  + i
	

	Fp2 = GF(p**_sage_const_2 , modulus=mod, names=('t',)); (t,) = Fp2._first_ngens(1)
	F = PolynomialRing(Fp2, names=('x',)); (x,) = F._first_ngens(1)
	poly = ((b + x) * x - c)
	roots = poly.roots()
	r1 = roots[_sage_const_0 ][_sage_const_0 ]
	r2 = roots[_sage_const_1 ][_sage_const_0 ]

	send(mod)
	send(r1 + b)
	send(r2 + b)
	send((a1 + r1 * a0) / (r1 - r2))
	send((a1 + r2 * a0) / (r1 - r2))

	en_calc = time.time()
	print(f"calc: {en_calc - st_calc:.3f}")

io.interactive()

