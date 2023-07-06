---
layout: post
category: example2
---

# Starter - Diffie-Hellman Starter 1

```python
print(pow(209, -1, 991))
```
```
569
```

</br></br></br>

# Starter - Diffie-Hellman Starter 2

앞에서부터 원시근을 찾으면 된다. 

```python
from Crypto.PublicKey import RSA
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import sympy

p = 28151

for i in range(2, p):
	chk = 1
	for j in range(1, p - 1):
		if pow(i, j, p) == 1:
			chk = 0
			break
	if chk == 1:
		print(i)
		break
```
```
7
```

</br></br></br>

# Starter - Diffie-Hellman Starter 3

```python
print(pow(g, a, p))
```

</br></br></br>

# Starter - Diffie-Hellman Starter 4

g ^ (a * b) = A ^ b = B ^ a가 shared secret이다. 

```python
print(pow(A, b, p))
```

</br></br></br>

# Starter - Diffie-Hellman Starter 5

만들어져 있는 decrypt 함수를 그대로 사용하면 된다. 

```python
shared_secret = pow(A, b, p)

print(decrypt_flag(shared_secret, iv, ciphertext))
```

flag
```
crypto{sh4r1ng_s3cret5_w1th_fr13nd5}
```
---
</br></br></br>

# Man In The Middle - Parameter Injection

Alice한테 보내는 B의 값을 1로 하면 B ^ a가 1이 되어 shared_secret이 1로 고정된다.

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import math
import sympy

r = remote('socket.cryptohack.org', 13371)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvuntil("Intercepted from Alice: ")
res = json_recv()

r.recvuntil("Send to Bob: ")
json_send({
    "p": hex(1),
    "g": hex(1),
    "A": hex(1)
    })


r.recvuntil("Intercepted from Bob: ")
res = json_recv()


r.recvuntil("Send to Alice: ")
json_send({
    "B": hex(1)
    })

r.recvuntil("Intercepted from Alice: ")
res = json_recv()

iv = res["iv"]
ciphertext = res["encrypted_flag"]



def is_pkcs7_padded(message):
    padding = message[-message[-1]:]
    return all(padding[i] == len(padding) for i in range(0, len(padding)))

def decrypt_flag(shared_secret: int, iv: str, ciphertext: str):
    # Derive AES key from shared secret
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]
    # Decrypt flag
    ciphertext = bytes.fromhex(ciphertext)
    iv = bytes.fromhex(iv)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)

    if is_pkcs7_padded(plaintext):
        return unpad(plaintext, 16).decode('ascii')
    else:
        return plaintext.decode('ascii')


shared_secret = 1

print(decrypt_flag(shared_secret, iv, ciphertext))

r.interactive()
```

flag
```
crypto{n1c3_0n3_m4ll0ry!!!!!!!!}
```
</br></br></br>

# Man In The Middle - Export-grade

64비트 내에서는 p를 법으로 한 로그를 discrete_log를 이용해서 뚫을 수 있다고 한다. 

그래서 DH64를 고르면 된다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import math
import sympy

r = remote('socket.cryptohack.org', 13379)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvuntil("Send to Bob: ")
json_send({
    "supported": ["DH64"]
    })

r.recvuntil("Send to Alice: ")
json_send({
    "chosen": "DH64"
    })

r.recvuntil("Intercepted from Alice: ")
res = json_recv()

p = int(res["p"], 16)
g = int(res["g"], 16)
A = int(res["A"], 16)

a = sympy.ntheory.residue_ntheory.discrete_log(p, A, g)

r.recvuntil("Intercepted from Bob: ")
res = json_recv()

B = int(res["B"], 16)

r.recvuntil("Intercepted from Alice: ")
res = json_recv()

iv = res["iv"]
ciphertext = res["encrypted_flag"]

def is_pkcs7_padded(message):
    padding = message[-message[-1]:]
    return all(padding[i] == len(padding) for i in range(0, len(padding)))

def decrypt_flag(shared_secret: int, iv: str, ciphertext: str):
    # Derive AES key from shared secret
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]
    # Decrypt flag
    ciphertext = bytes.fromhex(ciphertext)
    iv = bytes.fromhex(iv)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)

    if is_pkcs7_padded(plaintext):
        return unpad(plaintext, 16).decode('ascii')
    else:
        return plaintext.decode('ascii')


shared_secret = pow(B, a, p)

print(decrypt_flag(shared_secret, iv, ciphertext))

r.interactive()
```

flag
```
crypto{d0wn6r4d35_4r3_d4n63r0u5}
```

</br></br></br>

# Man In The Middle - Static Client

g를 A라고 뻥쳐서 보내면 Bob이 A ^ b를 계산해서 보내줘서 shared_secret을 바로 알 수 있다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import math
import sympy

r = remote('socket.cryptohack.org', 13373)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvuntil("Intercepted from Alice: ")
res = json_recv()
p = int(res["p"], 16)
g = int(res["g"], 16)
A = int(res["A"], 16)

r.recvuntil("Intercepted from Bob: ")
res = json_recv()

r.recvuntil("Intercepted from Alice: ")
res = json_recv()
iv = res["iv"]
ciphertext = res["encrypted"]

r.recvuntil("send him some parameters: ")
json_send({
    "p": hex(p),
    "g": hex(A),
    "A": hex(1)
    })

r.recvuntil("Bob says to you: ")
res = json_recv()
shared_secret = int(res["B"], 16)


def is_pkcs7_padded(message):
    padding = message[-message[-1]:]
    return all(padding[i] == len(padding) for i in range(0, len(padding)))

def decrypt_flag(shared_secret: int, iv: str, ciphertext: str):
    # Derive AES key from shared secret
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]
    # Decrypt flag
    ciphertext = bytes.fromhex(ciphertext)
    iv = bytes.fromhex(iv)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)

    if is_pkcs7_padded(plaintext):
        return unpad(plaintext, 16).decode('ascii')
    else:
        return plaintext.decode('ascii')

print(decrypt_flag(shared_secret, iv, ciphertext))

r.interactive()
```

flag
```
crypto{n07_3ph3m3r4l_3n0u6h}
```
---
</br></br></br>

# Group Theory - Additive

덧셈을 기준으로 한 군에서는 A -> a를 g로 나눠주기만 하면 되기 때문에 간단하다. 

당연히 p를 기준으로 해야 한다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import math
import sympy

r = remote('socket.cryptohack.org', 13380)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvuntil("Intercepted from Alice: ")
res = json_recv()
p = int(res["p"], 16)
g = int(res["g"], 16)
A = int(res["A"], 16)

r.recvuntil("Intercepted from Bob: ")
res = json_recv()
B = int(res["B"], 16)

r.recvuntil("Intercepted from Alice: ")
res = json_recv()
iv = res["iv"]
ciphertext = res["encrypted"]

a = A * pow(g, -1, p) % p

shared_secret = a * B % p


def is_pkcs7_padded(message):
    padding = message[-message[-1]:]
    return all(padding[i] == len(padding) for i in range(0, len(padding)))

def decrypt_flag(shared_secret: int, iv: str, ciphertext: str):
    # Derive AES key from shared secret
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]
    # Decrypt flag
    ciphertext = bytes.fromhex(ciphertext)
    iv = bytes.fromhex(iv)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)

    if is_pkcs7_padded(plaintext):
        return unpad(plaintext, 16).decode('ascii')
    else:
        return plaintext.decode('ascii')

print(decrypt_flag(shared_secret, iv, ciphertext))

r.interactive()
```

flag
```
crypto{cycl1c_6r0up_und3r_4dd1710n?}
```

</br></br></br>

# Group Theory - Static Client 2

g에다가 A를 넣는 방식으로는 안 된다. 

g에다가 들어갈 수 있는 대부분의 값을 막아놨는데 2 말고 5는 또 된다. 

그 두개가지고 할 수 있는 게 있나 해서 삽질을 많이 했다. 

</br>

또 간과하고 있던 사실이 p - 1이 소인수분해가 야무지게 되면은 p를 법으로 한 discrete_log가 잘 된다는 사실이었다. 

그리고 p를 검사하는 기준은 1536비트 이상인 소수인가여서 쉽게 뚫린다. 

이러한 p를 smooth p라고 하는 걸 알았다. 

나는 n! + 1 중에서 1536비트 이상이면서 소수인 값을 찾는 방식으로 했다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import math
from sympy.ntheory.residue_ntheory import discrete_log


r = remote('socket.cryptohack.org', 13378)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

def smooth_p():
    mul = 1
    i = 1
    while 1:
        mul *= i
        if (mul + 1).bit_length() >= 1536 and isPrime(mul + 1):
            return mul + 1
        i += 1

r.recvuntil("Intercepted from Alice: ")
res = json_recv()
p = int(res["p"], 16)
g = int(res["g"], 16)
A = int(res["A"], 16)

r.recvuntil("Intercepted from Bob: ")
res = json_recv()
B = int(res["B"], 16)

r.recvuntil("Intercepted from Alice: ")
res = json_recv()
iv = res["iv"]
ciphertext = res["encrypted"]

s_p = smooth_p()
print(s_p.bit_length())

r.recvuntil("send him some parameters: ")
json_send({
    "p": hex(s_p),
    "g": hex(2),
    "A": hex(A)
    })


r.recvuntil("Bob says to you: ")
res = json_recv()
B = int(res["B"], 16)
b = discrete_log(s_p, B, 2)

shared_secret = pow(A, b, p)


def is_pkcs7_padded(message):
    padding = message[-message[-1]:]
    return all(padding[i] == len(padding) for i in range(0, len(padding)))

def decrypt_flag(shared_secret: int, iv: str, ciphertext: str):
    # Derive AES key from shared secret
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]
    # Decrypt flag
    ciphertext = bytes.fromhex(ciphertext)
    iv = bytes.fromhex(iv)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)

    if is_pkcs7_padded(plaintext):
        return unpad(plaintext, 16).decode('ascii')
    else:
        return plaintext.decode('ascii')

print(decrypt_flag(shared_secret, iv, ciphertext))

r.interactive()
```

flag
```
crypto{uns4f3_pr1m3_sm4ll_oRd3r}
```

---
</br></br></br>

# Misc - Script Kiddie

^가 컴퓨터에서는 거듭제곱이 아니라 xor이다.

```python
shared_secret = g ^ A ^ B % p
```

flag
```
crypto{b3_c4r3ful_w1th_y0ur_n0tati0n}
```

</br></br></br>

# Misc - The Matrix

행렬의 거듭제곱을 한다. 

GF(2), 모든 항을 2를 법으로 했을 때 multiplicative_order()이라는 신기한 함수가 존재한다. 

몇 번 거듭제곱해야 E행렬이 되는지를 알려주기 때문에 

E * pow(E, -1, multiplicative_order) == 1 (mod multiplicative_order())이다. 

그거 말고도 sage 행렬관련 공부가 더 필요하다.

ex.sage
```python
import random

P = 2
N = 50
E = 31337

FLAG = b'crypto{??????????????????????????}'

bits = len(FLAG)

mat_bin = """00000001111101100001101010010001001011000110001001
생략
00110010110110011001001111110110000011001111010110"""

def bytes_to_binary(s):
    bin_str = ''.join(format(b, '08b') for b in s)
    bits = [int(c) for c in bin_str]
    return bits

def generate_mat():
    while True:
        msg = bytes_to_binary(FLAG)
        msg += [random.randint(0, 1) for _ in range(N*N - len(msg))]

        rows = [msg[i::N] for i in range(N)]
        mat = Matrix(GF(2), rows)

        if mat.determinant() != 0 and mat.multiplicative_order() > 10^12:
            return mat

def load_matrix():
    data = mat_bin.strip()
    rows = [list(map(int, row)) for row in data.splitlines()]
    return Matrix(GF(P), rows)


mat = load_matrix()

mat ^= pow(E, -1, mat.multiplicative_order())

bin_flag = ''.join([str(bit) for col in M.columns() for bit in col])
print(bytes.fromhex(hex(int(bin_flag[:34*8],2))[2:]).decode())

```

flag
```
crypto{there_is_no_spoon_66eff188}
```
</br></br></br>

# Misc - The Matrix Reload

jordan 대각화?라는 것을 사용하면 간단히 풀린다..... 대각화 까지는 생각했는데 그거는 막 p가 determinant가 0이고 문제가 많았다. 

jordan 대각화는 정말로 행렬에서 discrete log를 구할 때 많이 쓰일 것 같았다. 

jordan 행렬을 봐보니 맨 오른쪽, 아래서 두 번째에만 1이 있었고, 그게 k * lambda ^ (k - 1)이 되고, 아래 있는 값은 lambda ^ k가 되는 것을 이용해 잘 계산해서 k를 구하면 된다.

ex.sage
```python
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Util.number import *
from Crypto.Util.Padding import pad, unpad

P = 13322168333598193507807385110954579994440518298037390249219367653433362879385570348589112466639563190026187881314341273227495066439490025867330585397455471
N = 30

data = 생략
v = Matrix(GF(P), N, data["v"])
w = Matrix(GF(P), N, data["w"])

def load_matrix(fname):
    data = open(fname, 'r').read().strip()
    rows = [list(map(int, row.split(' '))) for row in data.splitlines()]
    return Matrix(GF(P), rows)

G = load_matrix("generator.txt")

j, p = G.jordan_form(transformation=True, subdivide=False)

assert p * j * p.inverse() == G

v_new = p.inverse() * v
w_new = p.inverse() * w

lamb = j[N - 1][N - 1]

a = v_new[N - 2][0]
b = v_new[N - 1][0]

lamb_k = w_new[N - 1] / b

k = int((w_new[N - 2] - a * lamb_k) * lamb / lamb_k / b)

assert (G ^ k) * v == w




aes_data = {"iv": "334b1ceb2ce0d1bef2af9937cf82aad6", "ciphertext": "543e29415bdb1f694a705b2532a5beb7ebd7009591503ef3c4fbcebf9e62fe91307e5d98efcd49f9f3b1985956cafc89"}
iv = bytes.fromhex(aes_data["iv"])
ct = bytes.fromhex(aes_data["ciphertext"])

KEY_LENGTH = 128
KEY = SHA256.new(data=str(k).encode()).digest()[:KEY_LENGTH]

cipher = AES.new(KEY, AES.MODE_CBC, iv)

flag = unpad(cipher.decrypt(ct), 16).decode()

print(flag)
```

flag
```
crypto{the_oracle_told_me_about_you_91e019ff}
```

</br></br></br>

# Misc - The Matrix Revolution

추천받은 논문에 설명되어 있다. 

https://theory.stanford.edu/~dfreeman/papers/discretelogs.pdf

그런데 읽고서도 사실 잘 이해가 안 된다.

ex.sage
```python
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Util.Padding import unpad

P = 2
N = 150
proof.arithmetic(False)

def load_matrix(fname):
    data = open(fname, 'r').read().strip()
    rows = [list(map(int, row)) for row in data.splitlines()]
    return Matrix(GF(P), rows)

G = load_matrix('generator.txt')
A_pub = load_matrix('alice.pub')
B_pub = load_matrix('bob.pub')

f = G.charpoly()
print(f.factor())

X = []
M = []
for g,e in f.factor():
    assert e == 1
    K = GF(2^g.degree(), x, modulus=g, impl='pari_ffelt')
    a = g.roots(K)[0][0]
    w = (G - a*1).right_kernel_matrix().rows()[0]
    V = [vector([0]*i + [1] + [0]*(N-1-i)) for i in range(150)]
    P = Matrix(K, [w] + V[:-1]).transpose()
    assert P.row_space().dimension() == N
    J_ = ~P * A_pub * P
    X.append(int(J_[0][0].log(a)))
    M.append(K.multiplicative_generator().multiplicative_order())

A_priv = crt(X, M)
shared_secret = B_pub^A_priv

iv = '43f14157442d75142d0d4993e99a9582'
ciphertext = '22abc3b347ffef55ec82488e5b4a338da5af7ef1918ac46f95029a4d94ace4cb2700fa9aeb31e6a4facee2601e99dabd6f9a81494c55f011e9227c9a6ae8d802'
KEY_LENGTH = 128
def derive_aes_key(M):
    mat_str = ''.join(str(x) for row in M for x in row)
    return SHA256.new(data=mat_str.encode()).digest()[:KEY_LENGTH]
key = derive_aes_key(shared_secret)
cipher = AES.new(key, AES.MODE_CBC, bytes.fromhex(iv))
flag = cipher.decrypt(bytes.fromhex(ciphertext))
print(unpad(flag, 16).decode())
```

풀이를 보고도 살짝은 이해가 잘 안 간다. 두고두고 되씹어야겠다. 

flag
```
crypto{we_are_looking_for_the_keymaker_478415c4}
```

---
</br></br></br>