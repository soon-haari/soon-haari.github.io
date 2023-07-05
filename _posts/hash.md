# Probability - Jack's Birthday Hash

https://crypto.stackexchange.com/questions/88748/calculating-minimum-number-of-messages-hashed-a-50-probability-of-a-collision

잘 설명되어 있다, 솔직히 맨땅으로 하라고 하면 억지다.

answer
```
1420
```

</br></br></br>

# Probability - Jack's Birthday Confusion

위에꺼를 통해 문제 의도를 분석하여 이번에는 직접 구할 수 있었다. 

ex.py
```python
import math

for i in range(2048):
	if math.perm(2048, i) / (2048 ** i) < 0.25:
		print(i)
		break
```

answer
```
76
```

---
</br></br></br>

# Collisions - Collider

md5 해시 충돌값을 아무거나 찾으면 된다.

무슨 lab에서 받은 md5 충돌 툴을 이용하였다.

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs
import os

r = remote('socket.cryptohack.org', 13389, level = 'debug')

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)


os.system("md5collgen -o col1 col2")

f = open("col1", "rb")

json_send({"document": bytes.hex(f.read())})

f = open("col2", "rb")

json_send({"document": bytes.hex(f.read())})

r.interactive()
```

flag
```
crypto{m0re_th4n_ju5t_p1g30nh0le_pr1nc1ple}
```

</br></br></br>

# Collisions - Hash Stuffing

그냥 리버싱처럼 블록 하나가 씹히도록 풀었다. 

다른 사람 풀이를 보니깐 63 * null, 64 * null을 넣어도 패딩 취약점 때문에 뚫린다고 한다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs
import os
from factordb.factordb import FactorDB

r = remote('socket.cryptohack.org', 13405)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

# 2^128 collision protection!
BLOCK_SIZE = 32

# Nothing up my sleeve numbers (ref: Dual_EC_DRBG P-256 coordinates)
W = [0x6b17d1f2, 0xe12c4247, 0xf8bce6e5, 0x63a440f2, 0x77037d81, 0x2deb33a0, 0xf4a13945, 0xd898c296]
X = [0x4fe342e2, 0xfe1a7f9b, 0x8ee7eb4a, 0x7c0f9e16, 0x2bce3357, 0x6b315ece, 0xcbb64068, 0x37bf51f5]
Y = [0xc97445f4, 0x5cdef9f0, 0xd3e05e1e, 0x585fc297, 0x235b82b5, 0xbe8ff3ef, 0xca67c598, 0x52018192]
Z = [0xb28ef557, 0xba31dfcb, 0xdd21ac46, 0xe2a91e3c, 0x304f44cb, 0x87058ada, 0x2cb81515, 0x1e610046]

# Lets work with bytes instead!
W_bytes = b''.join([x.to_bytes(4,'big') for x in W])
X_bytes = b''.join([x.to_bytes(4,'big') for x in X])
Y_bytes = b''.join([x.to_bytes(4,'big') for x in Y])
Z_bytes = b''.join([x.to_bytes(4,'big') for x in Z])

def pad(data):
    padding_len = (BLOCK_SIZE - len(data)) % BLOCK_SIZE
    return data + bytes([padding_len]*padding_len)

def blocks(data):
    return [data[i:(i+BLOCK_SIZE)] for i in range(0,len(data),BLOCK_SIZE)]

def xor(a,b):
    return bytes([x^y for x,y in zip(a,b)])

def rotate_left(data, x):
    x = x % BLOCK_SIZE
    return data[x:] + data[:x]

def rotate_right(data, x):
    x = x % BLOCK_SIZE
    return  data[-x:] + data[:-x]

def scramble_block(block):
    for _ in range(40):
        block = xor(W_bytes, block)
        block = rotate_left(block, 6)
        block = xor(X_bytes, block)
        block = rotate_right(block, 17)
    return block

def unscramble_block(block):
    for _ in range(40):
        block = rotate_left(block, 17)
        block = xor(X_bytes, block)
        block = rotate_right(block, 6)
        block = xor(W_bytes, block)
    return block

def cryptohash(msg):
    initial_state = xor(Y_bytes, Z_bytes)
    msg_padded = pad(msg)
    msg_blocks = blocks(msg_padded)
    for i,b in enumerate(msg_blocks):
        mix_in = scramble_block(b)
        for _ in range(i):
            mix_in = rotate_right(mix_in, i+11)
            mix_in = xor(mix_in, X_bytes)
            mix_in = rotate_left(mix_in, i+6)
        initial_state = xor(initial_state,mix_in)
    return initial_state.hex()

xor_hehe = b"\x00" * 32
mix_rev = rotate_right(xor_hehe, 1 + 6)
mix_rev = xor(mix_rev, X_bytes)
mix_rev = rotate_left(mix_rev, 1 + 11)
mix_rev = unscramble_block(mix_rev)

msg = pad(b"soon_haari")


json_send({"m1": bytes.hex(msg), "m2": bytes.hex(msg + mix_rev)})

r.interactive()
```

flag
```
crypto{Always_add_padding_even_if_its_a_whole_block!!!}
```

</br></br></br>

# Collisions - PriMeD5

소수 p1과 md5값이 같으면서 합성수인 p2를 찾아야 하는 문제이다. 

md5 해시는 해시가 같을 경우 앞뒤에 같은 값을 붙여줘도 해쉬값이 같다는 사실을 이용해서 풀 수 있는 문제이다. 

512비트 (single block) collision을 기준으로 시작해야 한다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs
import os
from factordb.factordb import FactorDB

r = remote('socket.cryptohack.org', 13392)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)


col1 = '4dc968ff0ee35c209572d4777b721587d36fa7b21bdc56b74a3dc0783e7b9518afbfa200a8284bf36e8e4b55b35f427593d849676da0d1555d8360fb5f07fea2'
col2 = '4dc968ff0ee35c209572d4777b721587d36fa7b21bdc56b74a3dc0783e7b9518afbfa202a8284bf36e8e4b55b35f427593d849676da0d1d55d8360fb5f07fea2'

added = 1

r.recvline()

while 1:
    print(added)
    p1 = bytes_to_long(bytes.fromhex(col1) + long_to_bytes(added))
    p2 = bytes_to_long(bytes.fromhex(col2) + long_to_bytes(added))

    if isPrime(p1) and not isPrime(p2):
        f = FactorDB(p2)
        f.connect()
        plist = f.get_factor_list()
        a = plist[0]

        json_send({"option": "sign", "prime": str(p1)})
        signature = json_recv()["signature"]
        json_send({"option": "check", "signature": signature, "prime": str(p2), "a": str(p2 // a)})
        break

    added += 2

r.interactive()
```

flag
```
crypto{MD5_5uck5_p4rt_tw0}
```

</br></br></br>

# Collisions - Twin Keys

hashclash를 이용해서 충돌 쌍을 찾는 게 정풀? 인 문제이다.

해킹 실력을 증가시켜주는 문제는 아닌 거 같다. 

collision1.bin, collision2.bin은 hashclash 툴을 이용한 결과이다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json

r = remote('socket.cryptohack.org', 13397)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

BLOCK_SIZE = 16

def pad(data):
    padding_len = (BLOCK_SIZE - len(data)) % BLOCK_SIZE
    return data + bytes([padding_len]*padding_len)

f1 = bytes.hex(open("collision1.bin", "rb").read())
f2 = bytes.hex(open("collision2.bin", "rb").read())

print(f1)
print(f2)

r.recvline()
json_send({
    "option": "insert_key",
    "key": f1
    })
json_recv()

json_send({
    "option": "insert_key",
    "key": f2
    })
json_recv()

json_send({
    "option": "unlock"
    })


r.interactive()
```

</br></br></br>

# Collisions - No Difference

permute 함수와 SBOX를 잘 관찰하면 쉽게 풀 수 있다. 

SBOX에서 운 좋게 0b01000000과 0b11110000이 둘다 0x79라는 값을 가져서 그냥 순삭이다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13395)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

xored = [80, 96, 112, 128]
xor1 = [0, 0, 1, 0]
xor2 = [1, 1, 1, 1]

key1 = 0
key2 = 0

for i in range(4):
    key1 *= 256
    key2 *= 256
    key1 += xored[i] ^ xor1[i]
    key2 += xored[i] ^ xor2[i]

key1 = long_to_bytes(key1)
key2 = long_to_bytes(key2)

r.recvline()
json_send({"a": bytes.hex(key1), "b": bytes.hex(key2)})

r.interactive()
```

flag
```
crypto{n0_d1ff_n0_pr0bl3m}
```

---
</br></br></br>

# Length Extension - MD0

hash 과정이 aes 암호화 과정이다. 해쉬 값을 충돌시키는 게 목적도 아니고 그냥 일치하는 값들을 조작해주기만 하면 된다. key가 랜덤으로 되어있기 때문에 해쉬를 알려주는 걸 이용해서 첫 블록 암호화 값을 알아낼 수 있다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

from Crypto.Util.Padding import pad
from Crypto.Cipher import AES

r = remote('socket.cryptohack.org', 13388)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

def bxor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

r.recvline()

json_send({
    "option": "sign",
    "message": ""
    })

enc = bytes.fromhex(json_recv()["signature"])

send = b"admin=True"
signature = bxor(AES.new(pad(send, 16), AES.MODE_ECB).encrypt(enc), enc)

json_send({
    "option": "get_flag",
    "signature": bytes.hex(signature),
    "message": bytes.hex(b"\x10" * 16 + send)
    })

r.interactive()
```

flag
```
crypto{l3ngth_3xT3nd3r}
```

</br></br></br>

# Length Extension - MDFlag

Length Extension Attack을 사용한다는 건 알았다. 그런데 hashpumpy 설치 실패 등의 문제로 생각보다 접근 자체가 빡셌다. 

하지만 length extension attack을 이해하고 나니 생각보다 막 어렵지는 않았다. 단지 툴을 구하기까지 조금 난관이 있었다.

https://github.com/cbornstein/python-length-extension 에 있는 모듈을 조금 수정해서 사용하였다. (digest 시, 자동으로 마지막에 패딩을 추가하는 것을 삭제하고 내가 원할 시 직접 추가할 수 있게 수정하였다. )

이 문제는 "}crypto{"라는 8바이트가 결정되어 있어 다행히도 쉽게 풀릴 수 있는 부분이 있었다. 

md5 해쉬는 특성상 바이트수를 64로 나눈 나머지가 56인 시점부터 현재 블록이 아닌 그 다음 블록까지 패딩을 하기 때문에, 64로 나눈 나머지가 55인 시점(가장 패딩을 적게 하는 시점)에 집중할 필요가 있다. 

위에서 설명한 "}crypto{"는 8바이트로 한 바이트가 부족하기 위해 머리를 잘 굴려서 256 브루트포스로 그 뒷 바이트를 구할 수 있다. "}crypto{i"이다. 

이 정보를 얻은 뒤에는 한 블록을 통째로 알기 때문에 md5의 state를 결정할 수 있고, 그 다음 블록에 한 바이트씩 추가하면서 256 브루트포스를 계속 돌리면 된다. 

어렵지는 않지만 논리적으로 세부적으로 들어가면 머리가 아픈 문제다. 풀이들을 보니 나는 굉장히 코드가 짧은 편이었다. 

그리고 위에서 다운받은 모듈이 python3에서 실행이 안 되길래 python2와 추가적인 import할 것들까지 다 다시 다운받아야 했다...... 다시 한번 byte string이 얼마나 혁신적인 기능인지 느낄 수 있었다. 

ex.py
```python
from itertools import cycle
import hashlib
from pwn import *
from Crypto.Util.number import *
import json
import pymd5

r = remote('socket.cryptohack.org', 13407)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

def bxor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

r.recvline()

json_send({
    "option": "message",
    "data": "00" * 183
    })

h183 = json_recv()["hash"]

FLAG = 'crypto{??????????????????????????????????????}'
l = len(FLAG) # 46

pad = pymd5.padding(8 * 183)
info = "}crypto{i"

flag = ""

for i in range(38):
    send_data = "00" * 183 + xor(info, pad).encode("hex") + "00" * len(flag) + "00"
    assert len(send_data) == (192 + i + 1) * 2

    json_send({
        "option": "message",
        "data": send_data
        })

    res = json_recv()["hash"]
    
    for j in range(256):
        h = pymd5.md5(state = h183.decode("hex"))
        upd = flag + chr(j) + pymd5.padding((192 + i + 1) * 8)

        h.update(upd)
        if h.hexdigest() == res:
            flag += chr(j)
            break
        if j == 255:
            print("RIP")
            quit()

    print("crypto{i" + flag)

r.interactive()
```
flag
```
crypto{i_Th1nk_mdFLAG_is_B3TTER_th4n_hmac!!!!}
```

---
</br></br></br>

# Pre-Image Attacks - Mixed Up

먼저 코드를 분석해보자

```python
mixed_xor = _xor(_xor(FLAG, data), os.urandom(len(FLAG)))
```

mixed_xor은 urandom이 들어간 순간부터 아무 의미가 없다, 완전 랜덤값이랑 동일하다.

mixed_and에서 힌트를 얻을 수 있는데, 플래그와 우리가 원하는 data와 and연산을 해서 조작을 한다. 

shuffle연산을 봐보면 39개의 and연산 후 바이트들이 다 같다면 sha256에 들어가는 값도 39개의 같은 문자임을 알 수 있다. 

한 비트씩 1을 집어넣어주면서 각 비트가 1인지 체크를 해주면 된다. 

ex.py
```python
from hashlib import sha256
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13402)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

FLAG = b"crypto{???????????????????????????????}"
l = len(FLAG)


def issamebit(h):
	for i in range(256):
		msg = bytes([i for _ in range(l)])
		if sha256(msg).hexdigest() == h:
			return 0
	return 1

flag = 0

for i in range(8 * l):
	msg = long_to_bytes(1 << i)
	msg = b"\x00" * (l - len(msg)) + msg
	bit = 0
	for _ in range(4):
		json_send({
			"option": "mix",
			"data": bytes.hex(msg)
			})
		if issamebit(json_recv()["mixed"]) == 1:
			bit = 1
	flag |= bit << i
	print(long_to_bytes(flag))

r.interactive()
```

중간에 for _ in range(4)는 낮은 확률로 shuffle연산에 들어가는 and연산값이 39개가 같은 문자가 아닌데도 같은 문자 39개를 반환하는 경우를 방지한다. 

실제로 저거를 한번만 하니까 crypto{y0u_c4n7_m1x^3v3ry7h1n6_1n_l1f3}라는 오류가 섞인 플래그가 나왔다. 

flag
```
crypto{y0u_c4n7_m1x_3v3ry7h1n6_1n_l1f3}
```

</br></br></br>

# Pre-Image Attacks - Invariant

먼저 해쉬를 생성하는 과정에 대해서 이해를 해보자.

메시지를 8바이트, 즉 64비트씩 쪼개서 블록 형태로 만든다. 

처음 상태의 메시지를 sha512에 넣어서 암호화 과정에 넣을 32 * 16 짜리 1비트 테이블을 만든다. 사실 여기서 힌트를 얻을 수 있다. sha512가 등장한 순간부터 그 테이블에 대한 정보는 얻을 수 있는 게 아무것도 없고, 랜덤값과 동일하게 생각해야 한다는 걸 말이다. 

그리고 16개의 값을 위에서 만든 테이블과 xor하고(중요: 즉 하위 4개 비트 중 최하위 비트밖에 안 바뀐다.), 순서를 좀 섞고 그 값을 __SB라는 배열에 넣는다. 

이 작업을 30번 반복한다. 

</br>

이제부터는 뭐 정형화된 방법을 떠나, 센스가 필요한 부분인데, 나의 경우 위에서 알게 되었듯이 최하위 비트를 모든 계산에서 제외하였다. 그래서 2k, 2k + 1은 앞 세 비트가 같은 쌍을 이룬다. 

그런데 __SB 배열을 보면 6, 7을 집어넣었을 때 7, 6이 나오는 것을 확인할 수 있다. 

즉 아무리 조작을 30번이고 많이 해도 조작되는 부분은 최하위 1비트이기 때문에 16개의 값을 다 6, 7로 설정하면, encrypt 후에도 6, 7로만 이루어진 상태가 유지되게 된다. 

Hash 과정에서도 우리가 목표하는 \x00 \* 16을 맞출 수 있게 착착 xor까지 해준다. 

</br>
이제는 이론상 1/65536의 확률로 6, 7 16개로만 이루어진 값을 넣었을 때 목표하는 값이 나온다는 것을 알 수 있다. 

단일 블록짜리 65536가지를 모두 돌렸을 때는 목표한 값이 나오지 않아서, 앞에 6 16개로만 이루어진 블록을 추가했더니 결과를 얻을 수 있었다. 

ex.py
```python
import itertools
import json
from hashlib import sha512

class MyCipher:
    __NR = 31
    __SB = [13, 14, 0, 1, 5, 10, 7, 6, 11, 3, 9, 12, 15, 8, 2, 4]
    __SR = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]

    def __init__(self, key):
        self.__RK = int(key.hex(), 16)
        self.__subkeys = [[(self.__RK >> (16 * j + i)) & 1 for i in range(16)]
                          for j in range(self.__NR + 1)]

    def __xorAll(self, v):
        res = 0
        for x in v:
            res ^= x
        return res

    def encrypt(self, plaintext):
        assert len(plaintext) == 8, "Error: the plaintext must contains 64 bits."

        S = [int(_, 16) for _ in list(plaintext.hex())]

        for r in range(self.__NR):
            S = [S[i] ^ self.__subkeys[r][i] for i in range(16)]
            S = [self.__SB[S[self.__SR[i]]] for i in range(16)]
            X = [self.__xorAll(S[i:i + 4]) for i in range(0, 16, 4)]
            S = [X[c] ^ S[4 * c + r]
                 for c, r in itertools.product(range(4), range(4))]

        S = [S[i] ^ self.__subkeys[self.__NR][i] for i in range(16)]
        return bytes.fromhex("".join("{:x}".format(_) for _ in S))


class MyHash:
    def __init__(self, content):
        self.cipher = MyCipher(sha512(content).digest())
        self.h = b"\x00" * 8
        self._update(content)

    def _update(self, content):
        while len(content) % 8:
            content += b"\x00"

        for i in range(0, len(content), 8):
            self.h = bytes(x ^ y for x, y in zip(self.h, content[i:i+8]))
            self.h = self.cipher.encrypt(self.h)
            self.h = bytes(x ^ y for x, y in zip(self.h, content[i:i+8]))

    def digest(self):
        return self.h

    def hexdigest(self):
        return self.h.hex()

yippie = 0

for i in range(1 << 16):
	content = "6" * 16
	for j in range(16):
		if i & (1 << j) > 0:
			content += '6'
		else:
			content += '7'
	content = bytes.fromhex(content)

	new = MyHash(content).digest()
	if new == bytes.fromhex("0000000000000000"):
		yippie = content
		break

assert MyHash(yippie).digest() == b"\x00" * 8

from pwn import *
import json

r = remote('socket.cryptohack.org', 13393)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()
json_send({
	"option": "hash",
	"data": bytes.hex(yippie)
	})

r.interactive()
```

flag
```
crypto{preimages_of_the_all_zero_output}
```

</br></br></br>