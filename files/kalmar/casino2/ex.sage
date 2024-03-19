from pwn import *

from Pedersen_commitments import gen, commit, verify

from sage.rings.factorint import factor_trial_division

from tqdm import trange

while True:
	# io = process(["python3", "casino.py"])
	io = remote("casino-2.chal-kalmarc.tf", "13337")

	io.recvuntil(b"q = ")
	q = int(io.recvline())
	io.recvuntil(b"g = ")
	g = int(io.recvline())
	io.recvuntil(b"h = ")
	h = int(io.recvline())

	order = factor_trial_division(q - 1, 100000000)
	tot = q - 1
	Fq = GF(q)
	h = Fq(h)
	g = Fq(g)
	lar = order[-1][0]

	h_ord = (q - 1) // lar
	g_ord = (q - 1) // lar

	# print(h_ord)

	for base, exp in order[:-1]:
		for i in range(exp):
			ex = tot // base^i

			if h^ex != 1:
				break
			else:
				m = i

		h_ord //= base^m



	for base, exp in order[:-1]:
		for i in range(exp):
			ex = tot // base^i

			if g^ex != 1:
				break
			else:
				m = i

		g_ord //= base^m
	assert h^(h_ord * lar) == 1
	assert g^(g_ord * lar) == 1

	diff = (q - 1) // (g_ord * lar)

	print(h_ord)
	print(g_ord)
	print(diff)

	if g_ord // gcd(h_ord, g_ord) >= 6 and diff > 1:
		break

	io.close()


io.sendline(b"d")

for _ in trange(250):
	io.recvuntil(b"Commitment: ")
	com = Fq(int(io.recvline()))

	com = com^(h_ord * lar)
	hh = h^(h_ord * lar)
	gg = g^(h_ord * lar)

	assert hh == 1
	assert gg != 1

	realgord = g_ord // gcd(h_ord, g_ord)

	assert gg^realgord == 1
	assert realgord >= 6

	print(realgord)

	for i in range(1, realgord):
		assert gg^i != 1

	for i in range(1, 7):
		if gg^i == com:
			break
	else:
		print("fucked")
		exit()
	ans = i
	# print(f"ans = {ans}")
	io.sendline("y")
	ans = ans + lar * g_ord
	# print(ans)
	# print(q)

	assert 1 < ans < q

	
	io.sendline(str(ans).encode())
























io.interactive()