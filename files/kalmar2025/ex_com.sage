from intarg import *
from server import NUMBER as N, check_proof
from tqdm import trange, tqdm

num = 2**NUML

a_vals = [four_squares((-3 % pr)) for pr in PRIMES]
a_arrs = [[a[i] for a in a_vals] for i in range(4)]
a_coms = [Merkle([str(aa) for aa in a]) for a in a_arrs]

b_vals = four_squares(N^2 - 4)
b_coms = [Comm(b) for b in b_vals]
b_arrs = [b.cord for b in b_coms]

p, q = 1, N
p_com, q_com = Comm(p), Comm(q)
p_arr, q_arr = p_com.cord, q_com.cord

all_coms = [p_com, q_com] + a_coms + b_coms
all_arrs = [p_arr, q_arr] + a_arrs + b_arrs

tx = Transcript(N)
for com in all_coms:
	tx.com(com)

coefs = [tx.challenge() for i in range(11)]

value = 0
tx.value(value)

poss = [int(tx.challenge() % num) for i in range(QUERIES)]

final_arr = []
for i in range(num):
	pr = PRIMES[i]

	if i in poss:
		val = -sum(coefs[j] * int(all_arrs[j][i]) for j in range(10))
		val *= pow(coefs[-1], -1, pr)
		val %= pr
		final_arr.append(str(val))
	else:
		final_arr.append("0")

final_com = Merkle(final_arr)

all_coms.append(final_com)
root = [com.root for com in all_coms]
open = [[com.open(pos) for pos in poss] for com in all_coms]

import json

proof = {
	"root": root,
	"open": open,
	"poss": poss,
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
