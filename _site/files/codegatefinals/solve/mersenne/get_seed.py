from MersenneDestroyer import rbtree_solve
import random
from state2seed import state2seed

def myrandbelow(t):
	k = t.bit_length()
	r = random.getrandbits(k)
	while r >= t:
		r = random.getrandbits(k)
	return r


def mysample(n, k):
	lst = list(range(n))
	res = []

	for i in range(k):
		rb = myrandbelow(n - i)

		ress = lst[rb]
		res.append(ress)

		# print(ress)
		lst[rb] = lst[n - i - 1]
	return res

random.seed(1234)
sp = random.sample(range(8501), 2125)

random.seed(1234)
sp2 = mysample(8501, 2125)

assert sp == sp2

output = []

tot_bit = 0

for i in range(624):
	output.append([0, 0])


for _ in range(2):
	tot_num = 0

	for i in range(310):
		pfx = "0111"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(1024):
		pfx = "1101"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(512):
		pfx = "11001"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(128):
		pfx = "110010"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(64):
		pfx = "1100100"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(32):
		pfx = "11001000"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(16):
		pfx = "110010000"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(8):
		pfx = "1100100000"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(4):
		pfx = "11001000000"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(2):
		pfx = "110010000000"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(1):
		pfx = "1100100000000"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(16):
		pfx = "1100011101"
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	for i in range(8):
		# pfx = "1100011101000"
		pfx = bin(6383 - i)[2:]
		output.append((int(pfx, 2), len(pfx)))
		tot_bit += len(pfx)
		tot_num += 1

	assert tot_num == 2125


print(tot_bit)

# output = []

st = rbtree_solve(output)

print(st)

exit()

random.setstate(st)

for _ in range(624):
	random.getrandbits(32)

sel1 = set(random.sample(range(8501), 2125))
sel2 = set(random.sample(range(8501), 2125))

nice = set(range(8501 - 2125, 8501))

assert sel1 == nice and sel2 == nice


print(st)