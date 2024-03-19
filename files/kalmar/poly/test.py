from Crypto.Hash import Poly1305
from Crypto.Cipher import ChaCha20
import os

def poly1305_hash(data, key, nonce):
	hsh = Poly1305.new(key=key, cipher=ChaCha20, nonce=nonce)
	hsh.update(data=data)
	return hsh.digest()

def divceil(a, b):
	return (a + b - 1) // b

def mypoly1305_hash(data, key, nonce):
	rs = ChaCha20.new(key=key, nonce=nonce).encrypt(b'\x00' * 32)
	r, s = rs[:16], rs[16:]

	r = int.from_bytes(r, "little")
	s = int.from_bytes(s, "little")
	r &= 0x0ffffffc0ffffffc0ffffffc0fffffff
	P = 0x3fffffffffffffffffffffffffffffffb

	res = 0
	for i in range(0, divceil(len(data), 16)):
		block = data[i*16:(i+1)*16] + b'\x01'
		res += int.from_bytes(block, "little")
		res = (r * res) % P
	res += s
	res %= 2**128
	res = res.to_bytes(16, "little")
	return res
	

for _ in range(10):
	msg = os.urandom(os.urandom(1)[0])
	key = os.urandom(32)
	nonce = os.urandom(8)
	res = poly1305_hash(msg, key, nonce)
	res2 = mypoly1305_hash(msg, key, nonce)

	assert res == res2