import random
import os
from Crypto.Hash import Poly1305
from Crypto.Cipher import ChaCha20
from tqdm import trange

TEST = False

dat = ['b19dd2', 'aa43fc', '2ff840', 'f944d4', 'e383bf', 'e8ff57', '2c3cad', '70bdad', 'cc8f65', '59a2c1', '51521b', 'b11505', '71bf95', '76bd04', '190e9e', '22a76c', '4f4b46', '0b0366', '6e5b74', '683d66', '0df575', '697683', '91a11f', '2c4a88', 'b40c52', '9fec93', '3ea23e', '64dab3', 'a66cca', 'ca0c19', 'd93c0a', '9fee90', 'fd1fb8', '4eeb7b', 'b0f7ed', 'f19fe4', '26d00e', 'f34a69', '1e988c', 'c25981', 'd9769d', '45c5f6', 'a1e567', 'e9a267', 'bad1bb', 'a93814', '76dfa1', 'd1654d', 'bee544', 'c4411e', 'b85110', 'fe920e', 'becb14', '3bc932', '932bac', '79b1a4', '5c4a1f', 'de129a', 'e4a860', '843b24', '0ea4e2', 'b8cfc0', '0e3430', 'f4b9b6', '1b4e1c', 'ba5b9b', '08265b', '864033', 'b608a8', '528292', 'b584de', 'f5a8d7', 'd1755e', 'da6933', 'dd4d1c', '535056', 'afdbf7', '851752', '88c665', '96767b', 'a08d9c', '32fba1', '0908c4', '89988f', '14d67f', 'ba2351', '8e61e8', '0dd35b', '4d9602', '9d856f', '243164', 'da0d61', 'ac1f97', '93b4b1', '48e574', '1d5541', 'c07b6a', 'ab87a2', 'dab5f2', '2696a5', '8131d0', 'af1274', 'e708a5', '86cc20', '27f6e5', '25561e', '184d0b', 'f4a48c', '7f42ae', '3669ec', 'cb2cd9', 'ca3bd7', 'c2ac38', 'd369c9', '3d11db', '0f46eb', 'd93f87', '8c55dc', '95f747', 'a75315', '852372', '84ffdb', '94d2c7', '9e3b85', 'c2bbad', 'dfb4e8', '311889', 'dcc5e6', '7eb10f', '59dffc', '1957a8', 'df10b3', 'ebfd25', 'fa1c6e', 'fe5802', '909042', 'aba5bf', 'db1403', 'acd4d7', '5ec9c6', 'e35e93', '0e0735', '77c6ca', '05a40e', 'b1b21b', '439f40', '6b5c30', '98e5c2', '1428df', '61726d', 'f6a253', '175142', 'd18960', 'ec8811', '5f1daa', '4c7d05', '92da84', '67c25c', '4967e9', '22c20e', 'ad68ef', '6523a7', '9dacd4', 'eb3b78', '4bf3dc', '020b44', '8bda25', 'f514ed', 'bc59b4', '59fba2', '9849c0', '547f6a', '4fa5f8', '6db1c8', '1ae1ba', '446db3', '1e9321', '2d966c', '93ade6', '8bc51b', '3a3e1a', '69c550', '8461c5', 'a1bc1c', '968377', '560eaa', 'e2f7be', '49e8cd', '3eaebc', '82729b', 'ed6bd4', '2d0b4f', '9ab070', '8e4610', '7545e9', '535e09', '6ed07d', '06a370', '2eb836', 'a4420d', 'c55236', 'fe9434', '44ed45', '4eb761', '193dbe', '8b8e4d', '1b1c78', 'fa05bb', 'f259d5', 'bcf9c7', '250f78', 'e3f928', '2e6c84', '34bab5', 'aacbf7', '837b84', '7b26be', '44693d', '2f1d0d', '106705', '3c9b72', 'f07768', '01bba5', 'ace1cf', '3344cc', 'a802bd', '77c9f8', '7da7ef', 'ebbf7d', '1cf462', 'b97592', '039f98', '6cdecf', '4a2c7b', 'bd8209', '2bfb88', '0bd392', '85a9be', 'c5c893', '1fc88e']
dat = [int.from_bytes(bytes.fromhex(d), "little") for d in dat]

p = 2^130 - 5
assert is_prime(p)

key, nonce = os.urandom(32), os.urandom(8)
st = os.urandom(16)
rs = ChaCha20.new(key=key, nonce=nonce).encrypt(b'\x00' * 32)
r, s = rs[:16], rs[16:]
r = int.from_bytes(r, "little")
s = int.from_bytes(s, "little")
r &= 0x0ffffffc0ffffffc0ffffffc0fffffff

def poly1305_hash(data):
	hsh = Poly1305.new(key=key, cipher=ChaCha20, nonce=nonce)
	hsh.update(data=data)
	return hsh.digest()

ps = []
real = []

n = 240

if TEST:
	for i in range(n):
		ps.append(int.from_bytes(st[10:13], "little"))
		real.append(int.from_bytes(st, "little"))

		st = poly1305_hash(st)
else:
	ps = dat

if TEST:
	for i in range(100):
		tst = (real[i] + 2^128) * r % p
		tst += s
		tst %= 2^128
		assert tst == real[i + 1]

reals = []
Ps = []
for i in range(n - 1):
	P = 2^106 * (ps[i] - ps[i + 1]) % p
	Ps.append(P)
	# print(hex(P))

	if TEST:
		Pr = 2^26 * (real[i] - real[i + 1]) % p
		reals.append(Pr)

		# print(hex(Pr))
		assert abs(Ps[i] - reals[i]).bit_length() <= 106
		# print(abs(P - Pr).bit_length())
# exit()

if TEST:
	diffs = []

	for i in range(n - 2):
		diff = reals[i] * r % p - reals[i + 1]
		# print(hex(diff))
		assert diff % 0x5000000 == 0
		diff //= 0x5000000
		assert diff in range(-4, 5)

		diffs.append(diff)

	rr = r

cnt = 0

MM = []
real_Ps = Ps[:]

bss = []

for i in range(2):
	base = 116 * i


	size = 30
	l = size * 4 - 2
	r = 3 * size
	d = size

	M = Matrix(r + d, r + d)

	for i in range(r):
		M[i, i] = 2^106

	for i in range(d):
		M[r + i, r + i] = p

	for i in range(r):
		for j in range(d):
			M[i, r + j] = Ps[base + i + j]

	M = M.LLL()


	v = M[0][:r]
	v = [k // 2^106 for k in v]

	MM = []

	def dot(aa, bb):
		assert len(aa) == len(bb)
		res = 0
		for a, b in zip(aa, bb):
			res += a * b
		return res

	for i in range(size - 1):
		v = M[i][:r]
		v = [k // 2^106 for k in v]

		# print(hex(dot(v, reals[i:i + r]) % p))
		for t in range(size - 1):
			if TEST:
				# print(dot(v, reals[t + base:t + r + base]) % p)
				assert dot(v, reals[t + base:t + r + base]) % p == 0
			cnt += 1

			app = [0] * t + v
			app += [0] * (l - len(app))
			MM.append(app)

	MM = Matrix(GF(p), MM)
	bs = MM.right_kernel().basis()
	assert len(bs) == 1
	bs = bs[0]
	if TEST:
		aa = (bs * reals[0 + base])[:20]
		for i in range(20):
			assert aa[i] == reals[i + base]

	M = Matrix(22, 22)
	for i in range(20):
		M[0, i] = ZZ(bs[i]) * 2^24
		M[i + 1, i] = p * 2^24
		M[21, i] = Ps[i + base] * 2^24


	wt = 2^135
	M[0, 20] = 1
	M[21, 21] = wt

	M = M.LLL()
	for v in M:
		if v[21] == wt:
			break
	else:
		print("no")
		exit()

	bs = -bs * v[20]
	bs = [ZZ(k) % p for k in bs]

	# print(len(bs))
	bss.append(bs)

assert bss[0][-2:] == bss[1][:2]
# print(bss)
bs = bss[0] + bss[1][2:]

if TEST:
	assert bs[:10] == reals[:10]

# print(len(bs))
assert len(bs) == 234

pos = None

for i in range(10):
	ss = set()
	for j in range(-4, 5):
		pp = ((bs[i + 1] + j * 0x5000000) / bs[i]) % p
		ss.add(pp)
	if pos == None:
		pos = ss
	else:
		pos = pos & ss

assert len(pos) == 1
# print(pos)
r_final = pos.pop()
# print(r_final)

if TEST:
	assert rr == r_final

assert r_final & 0x0ffffffc0ffffffc0ffffffc0fffffff == r_final
r = r_final
# print(r_final)

bs = [(-k * pow(2, -26, p)) % p for k in bs]

Ps = real_Ps

assert len(bs) == 234








toadd = []

todolen = 210
bs = bs[:todolen]

for i in range(todolen):
	toadd.append(sum(bs[:i]) % p)

if TEST:
	for i in range(todolen):
		assert (toadd[i] + real[0]) % p == real[i]




if TEST:
	s0 = s % 2^104
	sa = s >> 104
	assert s == 2^104 * sa + s0
else:
	init = '4b2b8f015ac838013b2330496b'
	init = bytes.fromhex(init)
	init = int.from_bytes(init, "little")
	s0 = init - (((int.from_bytes(b"init\x01", "little") * r) % p) % 2^104)

vvs = set()

for i in range(todolen - 1):
	vv = toadd[i + 1] - (2^128 + toadd[i]) * r - s0
	vv %= p
	vvs.add(vv)

assert len(vvs) == 5

for vv in vvs:
	if ((vv - 5) % p) in vvs:
		break

if TEST:
	assert (real[0] * (r - 1) + 2^104 * sa - vv) % p == 0



rrs = [0] * 500

cnt = 0

flagenc = '4a1891d571e9f122afcc0203d0aeeda9e66bc125df5c883e84fd6eeef23ecefc4efeacd0a612'
ct = bytes.fromhex(flagenc)

for saa in trange(2^24):
	# saa = sa
	mys = s0 + 2^104 * saa
	t = (vv - 2^104 * saa) / (r - 1)
	t %= p

	# assert mys == s
	# assert t == real[0]
	
	fail = False
	for i in range(todolen):
		# print(f"{i = }")
		rrs[i] = (t + toadd[i]) % p

		# print(rrs[0].bit_length())

		if rrs[i] >= 2^128:
			fail = True
			break

		
		if i >= 1:
			# assert rrs[i] == real[i]
			# assert rrs[i - 1] == real[i - 1]

			tst = (rrs[i - 1] + 2^128) * r % p
			tst += mys
			tst %= 2^128
			if rrs[i] != tst:
				fail = True
				break
		
	if fail:
		continue

	for i in range(todolen, todolen + 100):
		prev = rrs[i - 1]
		res = (prev + 2^128) * r % p
		res += mys
		res %= 2^128
		rrs[i] = res

	cnt += 1
	# print(cnt)

	fin = rrs[240:251]
	fin = [int(kk).to_bytes(16, "little")[10:13] for kk in fin]
	key = b"".join(fin)[:32]

	
	cipher = ChaCha20.new(key=key, nonce=b'\0'*8)
	pt = cipher.decrypt(ct)

	if b"mar" in pt:
		print(pt)
		break

