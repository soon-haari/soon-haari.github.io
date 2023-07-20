import hashlib
import os
import signal
from Crypto.Util.number import *
import random
from tqdm import trange


class SusCipher:
	S = [
		43,  8, 57, 53, 48, 39, 15, 61,
		 7, 44, 33,  9, 19, 41,  3, 14,
		42, 51,  6,  2, 49, 28, 55, 31,
		 0,  4, 30,  1, 59, 50, 35, 47,
		25, 16, 37, 27, 10, 54, 26, 58,
		62, 13, 18, 22, 21, 24, 12, 20,
		29, 38, 23, 32, 60, 34,  5, 11,
		45, 63, 40, 46, 52, 36, 17, 56
	]

	P = [
		21,  8, 23,  6,  7, 15,
		22, 13, 19, 16, 25, 28,
		31, 32, 34, 36,  3, 39,
		29, 26, 24,  1, 43, 35,
		45, 12, 47, 17, 14, 11,
		27, 37, 41, 38, 40, 20,
		 2,  0,  5,  4, 42, 18,
		44, 30, 46, 33,  9, 10
	]

	P_inv = [37, 21, 36, 16, 39, 38, 3, 4, 1, 46, 47, 29, 25, 7, 28, 5, 9, 27, 41, 8, 35, 0, 6, 2, 20, 10, 19, 30, 11, 18, 43, 12, 13, 45, 14, 23, 15, 31, 33, 17, 34, 32, 40, 22, 42, 24, 44, 26]

	ROUND = 3
	BLOCK_NUM = 8
	MASK = (1 << (6 * BLOCK_NUM)) - 1

	@classmethod
	def _divide(cls, v: int) -> list[int]:
		l: list[int] = []
		for _ in range(cls.BLOCK_NUM):
			l.append(v & 0b111111)
			v >>= 6
		return l[::-1]

	@staticmethod
	def _combine(block: list[int]) -> int:
		res = 0
		for v in block:
			res <<= 6
			res |= v
		return res

	@classmethod
	def _sub(cls, block: list[int]) -> list[int]:
		return [cls.S[v] for v in block]

	@classmethod
	def _perm(cls, block: list[int]) -> list[int]:
		bits = ""
		for b in block:
			bits += f"{b:06b}"

		buf = ["_" for _ in range(6 * cls.BLOCK_NUM)]
		for i in range(6 * cls.BLOCK_NUM):
			buf[cls.P[i]] = bits[i]

		permd = "".join(buf)
		return [int(permd[i : i + 6], 2) for i in range(0, 6 * cls.BLOCK_NUM, 6)]

	@classmethod
	def _perm_inv(cls, block: list[int]) -> list[int]:
		bits = ""
		for b in block:
			bits += f"{b:06b}"

		buf = ["_" for _ in range(6 * cls.BLOCK_NUM)]
		for i in range(6 * cls.BLOCK_NUM):
			buf[cls.P_inv[i]] = bits[i]

		permd = "".join(buf)
		return [int(permd[i : i + 6], 2) for i in range(0, 6 * cls.BLOCK_NUM, 6)]

	@staticmethod
	def _xor(a: list[int], b: list[int]) -> list[int]:
		return [x ^ y for x, y in zip(a, b)]

	def __init__(self, key: int):
		assert 0 <= key <= self.MASK

		keys = [key]
		for _ in range(self.ROUND):
			v = hashlib.sha256(str(keys[-1]).encode()).digest()
			v = int.from_bytes(v, "big") & self.MASK
			keys.append(v)

		self.subkeys = [self._divide(k) for k in keys]

	def encrypt(self, inp: int) -> int:
		block = self._divide(inp)

		block = self._xor(block, self.subkeys[0])
		for r in range(self.ROUND):
			block = self._sub(block)
			block = self._perm(block)
			block = self._xor(block, self.subkeys[r + 1])

		return self._combine(block)

	# TODO: Implement decryption
	def decrypt(self, inp: int) -> int:
		raise NotImplementedError()




from pwn import *

io = process(["python3", "task.py"])

def get(sends):
	to_send = sends[:]
	res = []
	while len(to_send):
		payload = to_send[:256]
		to_send = to_send[256:]

		io.sendlineafter("> ", str(payload)[1:-1])
		res.extend([int(recv) for recv in io.recvline().decode().split(",")])

	return res

def not0count(a):
	res = 0
	for k in a:
		if k:
			res += 1

	return res

send_n = 1000
sample_num = 1000

final_key = []

for idx in range(0, 8):

	send1 = []
	send2 = []

	st = []

	for i in range(send_n):
		s1 = [random.randrange(0, 64) for _ in range(8)]
		s2 = s1[:]

		a = random.randrange(0, 64)
		b = random.randrange(0, 64)

		s1[idx] = a
		s2[idx] = b

		send1.append(SusCipher._combine(s1))
		send2.append(SusCipher._combine(s2))

		st.append([a, b])

	res1 = get(send1)
	res2 = get(send2)

	key_chance = [0] * 64

	weight = [0] * 64

	for i in trange(sample_num):
		a = [random.randrange(0, 64) for _ in range(8)]
		for k in range(64):
			b = a[:]
			b[idx] ^= k

			a_ = a[:]
			b_ = b[:]

			a_ = SusCipher._perm(a_)
			b_ = SusCipher._perm(b_)

			a_ = SusCipher._sub(a_)
			b_ = SusCipher._sub(b_)

			a_ = SusCipher._perm(a_)
			b_ = SusCipher._perm(b_)

			weight[k] += not0count(SusCipher._xor(a_, b_))

	for i in range(64):
		weight[i] /= sample_num

	for i in range(send_n):
		a, b = st[i]
		final = not0count(SusCipher._perm_inv(SusCipher._divide(res1[i] ^ res2[i])))

		for k in range(64):
			init_dif = SusCipher.S[a ^ k] ^ SusCipher.S[b ^ k]

			key_chance[k] += abs(weight[init_dif] - final)

	smallest = key_chance[0]
	key = 0
	for i in range(1, 64):
		if key_chance[i] < smallest:
			smallest = key_chance[i]
			key = i

	final_key.append(key)

io.sendlineafter("> ", str(SusCipher._combine(final_key)))


io.interactive()
