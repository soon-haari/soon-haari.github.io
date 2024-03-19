from pwn import *
import random
from untemper import untemper
from tqdm import trange


# io = process(["python3", "casino.py"])
io = remote("chal-kalmarc.tf", 9)
io.sendline(b"a")

ress = []
state = []

for i in trange(624):
	io.sendline(b"n")

for i in trange(624):
	
	io.recvuntil(b"commited value was ")
	res = int(io.recvline())
	ress.append(res)

	state.append(untemper(res))

state = tuple([3, tuple(state + [0]), None])
random.setstate(state)

for i in range(624):
	assert ress[i] == random.randint(0, 2**32 - 2)

for i in trange(100):
	io.sendline(b"y")
	ans = random.randint(0, 2**32 - 2)
	io.sendline(str(ans).encode())



io.interactive()