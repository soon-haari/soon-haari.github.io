from intarg import *
from server import NUMBER as N, check_proof

num = 2**NUML
merk = Merkle([str(i) for i in range(num)])

tx = Transcript(N)

for i in range(10):
	tx.com(merk)

value = 0

coefs = [tx.challenge() for i in range(10)]

tx.value(0)

poss = [tx.challenge() % num for i in range(QUERIES)]

prs = [PRIMES[i] for i in poss]
mod = prod(prs)


acf = coefs[2:6]
bcf = coefs[6:10]

from tqdm import tqdm, trange

pairs = []

for i, p in tqdm(enumerate(prs), total=len(prs)):

	Fp = GF(p)
	P.<x> = PolynomialRing(Fp)

	while True:
		p_val = Fp(randrange(1, num))
		q_val = N / p_val
		if ZZ(q_val) >= num:
			continue

		tg = ZZ(q_val^2 - 4)

		bs = list(four_squares(tg))
		if any(ZZ(b) > num for b in bs):
			tg += p
			continue
		

		atar = p_val^2 - 4

		a3 = Fp(randrange(num))
		a4 = Fp(randrange(num))
		atar -= a3^2 + a4^2
		c1, c2 = acf[:2]

		star = -sum(aa * bb for aa, bb in zip([p_val, q_val, a3, a4] + bs, coefs[:2] + coefs[4:10]))

		a2 = (star - c1 * x) / c2
		poly = x^2 + a2^2 - atar
		roots = poly.roots()
		if len(roots) == 0:
			continue

		root = roots[0][0]
		a1 = root
		a2 = a2(x=root)

		if ZZ(a1) >= num or ZZ(a2) >= num:
			continue

		break

	pair = [p_val, q_val, a1, a2, a3, a4] + bs
	pair = list(map(int, pair))
	pairs.append(pair)


for i in range(QUERIES):
	p = prs[i]
	Fp = GF(p)
	p_val, q_val, a1, a2, a3, a4, b1, b2, b3, b4 = pairs[i]
	assert all(v < num for v in pairs[i])
	assert Fp(p_val * q_val - N) == 0
	assert Fp(a1^2 + a2^2 + a3^2 + a4^2 - (p_val^2 - 4)) == 0
	assert Fp(b1^2 + b2^2 + b3^2 + b4^2 - (q_val^2 - 4)) == 0
	assert Fp(sum(v1 * v2 for v1, v2 in zip(pairs[i], coefs))) == 0 == value

root = []
open = []
for i in range(10):
	rt = merk.root
	op = []

	for j in range(QUERIES):
		p = prs[j]
		v = pairs[j][i]
		oop = merk.open(int(v))
		assert int(oop[0]) == v
		op.append(oop)

	root.append(rt)
	open.append(op)

import json

proof = {
	"root": root,
	"open": open,
	"poss": [int(pos) for pos in poss],
	"vals": [int(value)]
}
proof = {'N': int(N), 'pf': proof}

assert check_proof(proof) == N

from pwn import *

io = remote("zzkaok.chal-kalmarc.tf", 9001r)
# io = process(["python3", "server.py"])
prf = json.dumps(proof).encode()
print(len(prf))
io.sendline(prf)

io.interactive()
