from pwn import *
from tqdm import trange, tqdm
import hashlib


TEST = False

if TEST:
	io = process(["sage", "task.sage"])
else:
	io = remote("52.79.59.27", 17776)

sys.set_int_max_str_digits(100000)

def decode_num(num, n):
	num_cp = num
	res = []
	for _ in range(n):
		res.append((num_cp % 3) - 1)
		num_cp //= 3
	res.extend([0] * (n - len(res)))
	return res

def encode_num(n, arr):
    res = 0
    for i in range(n):
        assert -1 <= arr[i] <= 1
        res += (arr[i] + 1) * (3 ** i)
    return res 

def fmt3(n):
	return ((int(n) + 1) % 3) - 1

def solve_pow():
	pfx = bytes.fromhex(io.recvline()[:-1].decode())
	assert len(pfx) == 8
	for i in trange(2**25):
		sfx = str(i)
		sfx = "0" * (24 - len(sfx)) + sfx
		sfx = sfx.encode()

		if hashlib.sha256(pfx + sfx).digest()[:3] == b"\x00" * 3:
			break

	io.sendline(bytes.hex(sfx))


if TEST == False:
	solve_pow()

def solve_task1():

	n, D = 2411, 83
	
	P = PolynomialRing(ZZ, 'x')
	x = P.gen()

	T = P.change_ring(GF(3)).quotient(x ** n - 1)
	x = T.gen()

	if TEST:
		ffff = P(decode_num(int(io.recvline()[:-1]), n))
		ffff3 = P(decode_num(int(io.recvline()[:-1]), n))


		mul = T(ffff * ffff3)
		assert mul == 1

	sel1 = eval(io.recvline()[:-1])
	sel2 = eval(io.recvline()[:-1])

	f = decode_num(int(io.recvline()[:-1]), n)
	f3 = decode_num(int(io.recvline()[:-1]), n)
	assert len(f) == n and len(f3) == n

	# print(f.count(-1))
	# print(f.count(0) - 83)
	# print(f.count(1))

	for s in sel1:
		assert f[s] == 0

	for s in sel2:
		assert f3[s] == 0

	var_num = 83 * 2
	M = []
	R = [0] * n
	R[0] = 1

	doubles = []
	for _ in range(n):
		M.append([0] * var_num)
		doubles.append([])


	for i in trange(n):
		for j in range(n):
			deg = (i + j) % n
			if i in sel1 and j in sel2:
				doubles[deg].append([sel1.index(i), sel2.index(j)])
				continue

			elif i in sel1:
				M[deg][sel1.index(i)] += f3[j]

			elif j in sel2:
				M[deg][sel2.index(j) + D] += f[i]
			
			else:
				R[deg] -= f[i] * f3[j]

	zero = []
	one = []
	for i in range(n):
		if len(doubles[i]) == 0:
			zero.append(i)
		if len(doubles[i]) == 1:
			one.append(i)

	cnt = [0] * 83

	# print(len(zero), len(one))

	for i in one:
		cnt[doubles[i][0][0]] += 1

	cnt = [cnt[i] * 1000000 + i for i in range(83)]

	cnt.sort()

	for i in range(1, 100):
		# print(f"asdf {i}")
		tot = len(zero)
		for k in cnt[-i:]:
			tot += k // 1000000
			tot += 1

		if tot >= 170:
			# print(tot)
			known = tot
			bf = i
			break

	bfs = [k % 1000000 for k in cnt[-bf:]]

	# print(bf)

	print(bfs)

	bf = len(bfs)


	M_default = []
	res_default = []

	for i in range(n):
		if i in zero:
			M_default.append(M[i])
			res_default.append(R[i])

	print(len(M_default))

	

	if TEST:

		correct_root = vector(GF(3), [ffff[t] for t in sel1] + [ffff3[t] for t in sel2])

		assert T(ffff) * T(ffff3) == 1

		roots = vector(GF(3), [correct_root[k] for k in bfs])

		to_add_M = []
		to_add_res = []

		for i in range(bf):
			a = [0] * 166
			a[bfs[i]] = 1
			
			to_add_M.append(a)
			to_add_res.append(roots[i])

		for i in range(n):
			if i not in one:
				continue

			a = doubles[i][0][0]
			b = doubles[i][0][1]

			# print(a, b)

			if a not in bfs:
				continue

			k = bfs.index(a)

			mm = M[i][:]

			mm[b + D] += roots[k]
			assert len(mm) == 166
			to_add_M.append(mm)
			to_add_res.append(R[i])

		MM = Matrix(GF(3), M_default + to_add_M)
		RRR = vector(GF(3), res_default + to_add_res)

		# assert len(MM) == len(RRR) == known

		root = MM.solve_right(RRR)

		assert root == correct_root
		print(root)
		print(correct_root)
		print()

		f_scene = f[:]
		f3_scene = f3[:]

		for i in range(83):
			f_scene[sel1[i]] = fmt3(root[i])
			f3_scene[sel2[i]] = fmt3(root[i + D])


		if T(f_scene) * T(f3_scene) == 1:
			print("FUCK YEAH")



	sent = False

	for it in trange(3**bf):
		it_ = it
		roots = []
		for i in range(bf):
			roots.append((it_ % 3) - 1)
			it_ //= 3

		to_add_M = []
		to_add_res = []

		for i in range(bf):
			a = [0] * 166
			a[bfs[i]] = 1
			
			to_add_M.append(a)
			to_add_res.append(roots[i])

		for i in range(n):
			if i not in one:
				continue

			a = doubles[i][0][0]
			b = doubles[i][0][1]

			# print(a, b)

			if a not in bfs:
				continue

			k = bfs.index(a)

			mm = M[i][:]

			mm[b + D] += roots[k]
			assert len(mm) == 166
			to_add_M.append(mm)
			to_add_res.append(R[i])

		MM = Matrix(GF(3), M_default + to_add_M)
		RRR = vector(GF(3), res_default + to_add_res)

		# assert len(MM) == len(RRR) == known

		try:
			root = MM.solve_right(RRR)

			# print(root)
			# print(correct_root)
			# print()


		except:
			continue

		if TEST:
			chk = True
			for k in bfs:
				if GF(3)(root[k]) != GF(3)(correct_root[k]):
					chk = False

			if chk:
				print(correct_root)
				print(root)
				assert correct_root == root

		f_scene = f[:]
		f3_scene = f3[:]

		for i in range(83):
			f_scene[sel1[i]] = fmt3(root[i])
			f3_scene[sel2[i]] = fmt3(root[i + D])

		if T(f_scene) * T(f3_scene) == 1:
			# print("FUCK YEAH")
			io.sendline(str(encode_num(n, f_scene)))
			io.sendline(str(encode_num(n, f3_scene)))

			sent = True

			break

	if sent == False:
		print("UNnlucky fuck")
		exit()






def solve_task2():
	n, D = 8501, 2125
	seed_str = open("seed.txt", "rb").read()
	seed = int(seed_str)

	import random

	random.seed(seed)

	sel = set(range(n - D, n))

	assert sel == set(random.sample(range(n), D))
	assert sel == set(random.sample(range(n), D))


	io.sendline(seed_str)

	assert sel == set(eval(io.recvline()))
	assert sel == set(eval(io.recvline()))

	f = decode_num(int(io.recvline()[:-1]), n)
	f3 = decode_num(int(io.recvline()[:-1]), n)

	assert f[-D:] == [0] * D
	assert f3[-D:] == [0] * D

	M = []
	for _ in range(4252):
		M.append([0] * D * 2)
	r = [0] * 4252
	r[0] = 1

	unk = set(range(6376, 8501))

	for i in trange(4251):
		for j in range(n):
			a, b = j, (i - j) % n
			if a in unk and b in unk:
				print("Fuck")
				exit()

			elif a in unk:
				M[i][a - 6376] += f3[b]
			elif b in unk:
				M[i][b - 6376 + D] += f[a]
			else:
				r[i] -= f[a] * f3[b]

	for j in range(n):
		a, b = j, (8500 - j) % n
		if a in unk and b in unk:
			print("Fuck")
			exit()

		elif a in unk:
			M[4251][a - 6376] += f3[b]
		elif b in unk:
			M[4251][b - 6376 + D] += f[a]
		else:
			r[4251] -= f[a] * f3[b]


	M = Matrix(GF(3), M)
	r = vector(GF(3), r)

	root = M.solve_right(r)

	rk = M.rank()

	if rk != 4250:
		print("Fuckfuck")
		exit()

	for i in range(2125):
		f[i + 6376] = fmt3(root[i])
		f3[i + 6376] = fmt3(root[i + D])

	f_num = encode_num(n, f)
	f3_num = encode_num(n, f3)

	io.sendline(str(f_num))
	io.sendline(str(f3_num))

	

for _ in range(8):
	print(f"Round {_ + 1}")
	solve_task1()

for _ in range(8):
	print(f"Round {_ + 1}")
	solve_task2()

io.interactive()