import random
from state2seed import state2seed
from recoverState import recoverState
from pwn import *

r = remote('janken-vs-kurenaif.seccon.games', 8080)
r.recvuntil(b'kurenaif: My spell is')
s = r.recvuntil(b'.')[:-1]
state = recoverState(s)
seed = state2seed(state)
r.recvuntil(b': ')
r.sendline(hex(seed)[2:])
r.interactive()    