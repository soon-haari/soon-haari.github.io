# How AES Works - Keyed Permutations

구글링해야된다.

flag
```
crypto{bijection}
```

</br></br></br>

# How AES Works - Resisting Bruteforce

또 구글링해야된다.

flag
```
crypto{biclique}
```

</br></br></br>

# How AES Works - Structure of AES

행렬을 스트링으로 만드는걸 구현하라고 하신다.

시키는데로 해야지.

matrix.py
```python
def bytes2matrix(text):
    """ Converts a 16-byte array into a 4x4 matrix.  """
    return [list(text[i:i+4]) for i in range(0, len(text), 4)]

def matrix2bytes(matrix):
    """ Converts a 4x4 matrix into a 16-byte array.  """
    ans = ""
    for i in range(4):
        for j in range(4):
            ans += chr(matrix[i][j])
    return ans

matrix = [
    [99, 114, 121, 112],
    [116, 111, 123, 105],
    [110, 109, 97, 116],
    [114, 105, 120, 125],
]

print(matrix2bytes(matrix))

```

flag
```
crypto{inmatrix}
```

</br></br></br>


# How AES Works - Round Keys

xor이 암호화 알고리즘이다.

add_round_key.py
```python
state = [
    [206, 243, 61, 34],
    [171, 11, 93, 31],
    [16, 200, 91, 108],
    [150, 3, 194, 51],
]

round_key = [
    [173, 129, 68, 82],
    [223, 100, 38, 109],
    [32, 189, 53, 8],
    [253, 48, 187, 78],
]


def add_round_key(s, k):
    ans = ""
    for i in range(4):
        for j in range(4):
            ans += chr(s[i][j] ^ k[i][j])
    return ans


print(add_round_key(state, round_key))
```

flag
```
crypto{r0undk3y}
```

</br></br></br>

# How AES Works - Confusion through Substitution

키 테이블을 통해서 state를 역변환한다. 

sbox.py
```python
def sub_bytes(s, sbox=s_box):
    ans = ""
    for i in range(4):
        for j in range(4):
            ans += chr(sbox[state[i][j]])
    return ans

```

flag
```
crypto{l1n34rly}
```

</br></br></br>

# How AES Works - Diffusion through Permutation

shift 역연산을 구현한다.

diffusion.py
```python
def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]


def inv_shift_rows(s):
    s[1][1], s[2][1], s[3][1], s[0][1] = s[0][1], s[1][1], s[2][1], s[3][1]
    s[2][2], s[3][2], s[0][2], s[1][2] = s[0][2], s[1][2], s[2][2], s[3][2]
    s[3][3], s[0][3], s[1][3], s[2][3] = s[0][3], s[1][3], s[2][3], s[3][3]


# learned from http://cs.ucsb.edu/~koc/cs178/projects/JT/aes.c
xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):
    # see Sec 4.1.2 in The Design of Rijndael
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)


def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])


def inv_mix_columns(s):
    # see Sec 4.1.3 in The Design of Rijndael
    for i in range(4):
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        s[i][0] ^= u
        s[i][1] ^= v
        s[i][2] ^= u
        s[i][3] ^= v

    mix_columns(s)

def matrix2bytes(matrix):
    """ Converts a 4x4 matrix into a 16-byte array.  """
    ans = ""
    for i in range(4):
        for j in range(4):
            ans += chr(matrix[i][j])
    return ans

state = [
    [108, 106, 71, 86],
    [96, 62, 38, 72],
    [42, 184, 92, 209],
    [94, 79, 8, 54],
]


inv_mix_columns(state)
inv_shift_rows(state)

print(matrix2bytes(state))
```

flag
```
crypto{d1ffUs3R}
```

</br></br></br>

# How AES Works - Bringing It All Together

지금까지 나온 것들을 다 합치면 된다. 

xtime과 expand_key는 이해가 더 필요해 보인다. 

ex.py
```python
N_ROUNDS = 10

key        = b'\xc3,\\\xa6\xb5\x80^\x0c\xdb\x8d\xa5z*\xb6\xfe\\'
ciphertext = b'\xd1O\x14j\xa4+O\xb6\xa1\xc4\x08B)\x8f\x12\xdd'

s_box = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

inv_s_box = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

def add_round_key(s, k):
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]

def bytes2matrix(master_key):
    key = []
    for i in range(4):
        subkey = []
        for j in range(4):
            subkey.append(master_key[4 * i + j])
        key.append(subkey)
    return key

def matrix2bytes(matrix):
    ans = ""
    for i in range(4):
        for j in range(4):
            ans += chr(matrix[i][j])
    return ans

def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]


def inv_shift_rows(s):
    s[1][1], s[2][1], s[3][1], s[0][1] = s[0][1], s[1][1], s[2][1], s[3][1]
    s[2][2], s[3][2], s[0][2], s[1][2] = s[0][2], s[1][2], s[2][2], s[3][2]
    s[3][3], s[0][3], s[1][3], s[2][3] = s[0][3], s[1][3], s[2][3], s[3][3]


xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):
    # see Sec 4.1.2 in The Design of Rijndael
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)


def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])


def inv_mix_columns(s):
    # see Sec 4.1.3 in The Design of Rijndael
    for i in range(4):
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        s[i][0] ^= u
        s[i][1] ^= v
        s[i][2] ^= u
        s[i][3] ^= v

    mix_columns(s)

def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = inv_s_box[s[i][j]]

def expand_key(master_key):
    """
    Expands and returns a list of key matrices for the given master_key.
    """

    # Round constants https://en.wikipedia.org/wiki/AES_key_schedule#Round_constants
    r_con = (
        0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
        0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
        0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
    )

    # Initialize round keys with raw key material.
    key_columns = bytes2matrix(master_key)
    iteration_size = len(master_key) // 4
    # Each iteration has exactly as many columns as the key material.
    i = 1
    while len(key_columns) < (N_ROUNDS + 1) * 4:
        # Copy previous word.
        word = list(key_columns[-1])

        # Perform schedule_core once every "row".
        if len(key_columns) % iteration_size == 0:
            # Circular shift.
            word.append(word.pop(0))
            # Map to S-BOX.
            word = [s_box[b] for b in word]
            # XOR with first byte of R-CON, since the others bytes of R-CON are 0.
            word[0] ^= r_con[i]
            i += 1
        elif len(master_key) == 32 and len(key_columns) % iteration_size == 4:
            # Run word through S-box in the fourth iteration when using a
            # 256-bit key.
            word = [s_box[b] for b in word]

        # XOR with equivalent word from previous iteration.
        word = bytes(i^j for i, j in zip(word, key_columns[-iteration_size]))
        key_columns.append(word)

    # Group key words in 4x4 byte matrices.
    return [key_columns[4*i : 4*(i+1)] for i in range(len(key_columns) // 4)]

def decrypt(key, ciphertext):
    round_keys = expand_key(key)
    
    text = bytes2matrix(ciphertext)

    # Initial add round key step
    add_round_key(text, round_keys[10])

    for i in range(N_ROUNDS - 1, 0, -1):
        inv_shift_rows(text)
        inv_sub_bytes(text)
        add_round_key(text, round_keys[i])
        inv_mix_columns(text)

    inv_shift_rows(text)
    inv_sub_bytes(text)
    add_round_key(text, round_keys[0])

    plaintext = matrix2bytes(text)

    return plaintext


print(decrypt(key, ciphertext))

```
flag
```
crypto{MYAES128}
```
---
</br></br></br>

# Symmetric Starter - Modes of Operation Starter

encrypt_flag를 decrypt하고 hex to bytes만 실행해주면 된다. 

flag
```
crypto{bl0ck_c1ph3r5_4r3_f457_!}
```

</br></br></br>

# Symmetric Starter - Passwords as Keys

words를 다운받아서 십만개의 가능한 키들을 모두 돌려준다.

그중에서 decrypt했을 때 앞글자가 cr로 시작하는 것으로 필터링하면 정답이 하나밖에 나오지 않는다. 

ex.py
```python
from Crypto.Cipher import AES
import hashlib
import random

def decrypt(ciphertext, password_hash):
    ciphertext = bytes.fromhex(ciphertext)
    key = password_hash

    cipher = AES.new(key, AES.MODE_ECB)
    try:
        decrypted = cipher.decrypt(ciphertext)
    except ValueError as e:
        return {"error": str(e)}

    return decrypted

with open("words.txt") as f:
    words = [w.strip() for w in f.readlines()]
keyword = random.choice(words)

KEY = hashlib.md5(keyword.encode()).digest()

l = len(words)

for i in range(l):
    KEY = hashlib.md5(words[i].encode()).digest()
    decrypted = decrypt("c92b7734070205bdf6c0087a751466ec13ae15e6f1bcdd3f3a535ec0f4bbae66", KEY)
    
    if decrypted[0] == ord('c') and decrypted[1] == ord('r'):
        print(decrypted.decode())
```

flag
```
crypto{k3y5__r__n07__p455w0rdz?}
```
---
</br></br></br>

# Block Ciphers - ECB CBC WTF

ECB와 CBC의 구조 차이를 알고 있으면 풀 수 있는 문제이다.

CBC는 ECB에 비해 추가적인 xor연산을 하며, 그것만 해결해 주면 블록들별로 정보를 얻을 수 있다. 

ex.py
```python
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import long_to_bytes, bytes_to_long

def response(byte_string):
	url = "http://aes.cryptohack.org/ecbcbcwtf/decrypt/"
	url += byte_string.hex()
	url += "/"
	r = requests.get(url)
	js = r.json()
	return bytes.fromhex(js["plaintext"])

def encrypt_flag():
	url = "http://aes.cryptohack.org/ecbcbcwtf/encrypt_flag/"
	r = requests.get(url)
	js = r.json()
	return bytes.fromhex(js["ciphertext"])

def xor(a, b):
	return long_to_bytes(bytes_to_long(a) ^ bytes_to_long(b))

enc = encrypt_flag()

iv = enc[:16]
block1 = enc[16:32]
block2 = enc[32:]

decrypt_block1 = xor(response(block1), iv)
decrypt_block2 = xor(response(block2), block1)
print(decrypt_block1 + decrypt_block2)
```

flag
```
crypto{3cb_5uck5_4v01d_17_!!!!!}
```

</br></br></br>

# Block Ciphers - ECB Oracle

재미있는 문제였다. 플래그를 앞으로 한 칸씩 당기면서 256바이트 브루트 포싱을 하면 된다. 

출력 가능한 캐릭터만 고려하면 33 ~ 128만 해도 된다고 한다. 

그런데 이제 resquest get 뭐 그런걸 써야되다 보니깐 좀 신기하고 생소했다.

답이 나오기까지 시간이 오래걸렸다. 

ex.py
```python
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

flag = b"crypto{"

# flag length is 26

def response(byte_string):
	url = "http://aes.cryptohack.org/ecb_oracle/encrypt/"
	url += byte_string.hex()
	url += "/"
	r = requests.get(url)
	js = r.json()
	return bytes.fromhex(js["ciphertext"])

for i in range(7, 26):
	byte_string = b""
	byte_string += b"\x00" * (31 - i)
	
	res = response(byte_string)[:32]

	byte_string += flag
	for j in range(33, 128):
		byte_string = byte_string[:31]
		byte_string += j.to_bytes(1, byteorder = "big")
		print(j)

		res2 = response(byte_string)[:32]
		if res == res2:
			flag += j.to_bytes(1, byteorder = "big")
			print(flag)
			break
```

flag
```
crypto{p3n6u1n5_h473_3cb}
```

</br></br></br>

# Block Ciphers - Flipping Cookie

CBC의 decryption 과정을 잘 알면 iv만 조작을 하면 복호화문도 쉽게 조작을 할 수 있음을 알 수 있다. 

코드만 봐도 이해가 되는 풀이이다. 

ex.py
```python
import requests
from Crypto.Cipher import AES
import os
from Crypto.Util.Padding import pad, unpad
from datetime import datetime, timedelta
from Crypto.Util.number import long_to_bytes, bytes_to_long

def get_cookie():
    url = "http://aes.cryptohack.org/flipping_cookie/get_cookie/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["cookie"])

def response(cookie, iv):
    url = "http://aes.cryptohack.org/flipping_cookie/check_admin/"
    url += cookie.hex()
    url += "/"
    url += iv.hex()
    url += "/"
    r = requests.get(url)
    js = r.json()
    print(js)

def xor(a, b):
    return long_to_bytes(bytes_to_long(a) ^ bytes_to_long(b))

cookie = get_cookie()

origin = b'admin=False;expi'
goal = b'admin=True;\x05\x05\x05\x05\x05'

iv = cookie[:16]
block1 = cookie[16:32]
block2 = cookie[32:]

send_iv = xor(xor(origin, goal), iv)

response(block1, send_iv)
```

flag
```
crypto{4u7h3n71c4710n_15_3553n714l}
```

</br></br></br>

# Block Ciphers - Lazy CBC

머리를 굴리면 보인다. 

AB (A, B는 모두 16글자)를 decrypt했을 때의 결과를 CD라고 하자.

A를 key로 decrypt한거 ^ key = C

B를 key로 decrypt한거 ^ A = D 이다. 

즉 A = B이면 

key ^ C = A를 key로 decrypt한거 = A ^ D가 되어

key = A ^ C ^ D이다. 

A = B = 0 * 16으로 놓으면 key는 C ^ D가 될 것이다. 

ex.py
```python
import requests
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import long_to_bytes, bytes_to_long

def get_flag(key):
    url = "http://aes.cryptohack.org/lazy_cbc/get_flag/"
    url += key.hex()
    url += "/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["plaintext"])

def response(ciphertext):
    url = "http://aes.cryptohack.org/lazy_cbc/receive/"
    url += ciphertext.hex()
    url += "/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["error"][len("Invalid plaintext: "):])

def xor(a, b):
    return long_to_bytes(bytes_to_long(a) ^ bytes_to_long(b))

ciphertext = b"\x00" * 32

CD = response(ciphertext)
C = CD[:16]
D = CD[16:]

print(get_flag(xor(C, D)))
```

flag
```
crypto{50m3_p30pl3_d0n7_7h1nk_IV_15_1mp0r74n7_?}
```

</br></br></br>

# Block Ciphers - Triple DES

DES에 weak key라는 것이 있다는 것을 구글링하면서 알았다 우와....

weak key의 경우 encryption과 decryption이 같은 동작을 해서 encrypt를 두 번 하면 평문이 된다고 한다. 

ex.py
```python
import requests
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import long_to_bytes, bytes_to_long

def encrypt_flag(key):
    url = "http://aes.cryptohack.org/triple_des/encrypt_flag/"
    url += key.hex()
    url += "/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["ciphertext"])

def encrypt(key, plaintext):
    url = "http://aes.cryptohack.org/triple_des/encrypt/"
    url += key.hex()
    url += "/"
    url += plaintext.hex()
    url += "/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["ciphertext"])

def xor(a, b):
    return long_to_bytes(bytes_to_long(a) ^ bytes_to_long(b))

ciphertext = b"\x00" * 32

weak_key = [b"\x00" * 8, b"\xff" * 8]

key = weak_key[0] + weak_key[1]

step1 = encrypt_flag(key)
step2 = encrypt(key, step1)

print(unpad(step2, 8))
```

저 두 byte string이 weak key의 예시이다. 

flag
```
crypto{n0t_4ll_k3ys_4r3_g00d_k3ys}
```
---
</br></br></br>

# Stream Ciphers - Symmetry

OFB의 구조를 친절하게 그려준다. 

plaintext ^ ciphertext의 값이 plaintext의 값과 상관없이 key, iv에만 의존하는 취약점을 이용할 수 있다. 

00.....을 암호화한 값 = flag ^ encrypt_flag임을 알 수 있다. 

ex.py
```python
import requests
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import long_to_bytes, bytes_to_long

def encrypt_flag():
    url = "http://aes.cryptohack.org/symmetry/encrypt_flag/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["ciphertext"])

def encrypt(plaintext, iv):
    url = "http://aes.cryptohack.org/symmetry/encrypt/"
    url += plaintext.hex()
    url += "/"
    url += iv.hex()
    url += "/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["ciphertext"])

def xor(a, b):
    return long_to_bytes(bytes_to_long(a) ^ bytes_to_long(b))

res = encrypt_flag()
iv = res[:16]
flag_enc = res[16:]
l = len(flag_enc)

plaintext = b'\x00' * l

flag = xor(encrypt(plaintext, iv), flag_enc)

print(flag)
```

flag
```
crypto{0fb_15_5ymm37r1c4l_!!!11!}
```

</br></br></br>

# Stream Ciphers - Bean Counter

푸는 방법은 쉽게 알았는데 내 xor 함수가 잘못 구현되어 있었고 지금까지 256/1의 낮은 확률의 오류를 모두 뚫어냈다는 것을 알게 되었다. 

PNG의 첫 16바이트는 동일함을 이용해서 xor해주어야 하는 키를 찾는다. 

ex.py
```python
import requests
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import long_to_bytes, bytes_to_long


def encrypt():
    url = "http://aes.cryptohack.org/bean_counter/encrypt/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["encrypted"])

def xor(a, b):
    xored = b""
    for i in range(len(a)):
        xored += (a[i] ^ b[i]).to_bytes(1, byteorder = "big")

    return xored


f = open("29072.png", 'rb')
start = f.read(16)

encrypted = encrypt()

key = xor(start, encrypted[:16])

f.close()

f1 = open("bean_flag.png", 'wb')

rnd = len(encrypted) // 16

for i in range(rnd):
    f1.write(xor(key, encrypted[i * 16: (i + 1) * 16]))

f1.close()
```

flag
```
crypto{hex_bytes_beans}
```

</br></br></br>

# Stream Ciphers - CTRIME

zlib compress에 함정이 있겠구나 하고 생각했지만 솔직히 검색해보다가 힌트를 얻어서 풀었다. 

"crypto{" 로 시작해서 플래그 앞에 스트링을 붙이면 반복되면 압축 후의 길이가 더 짧은 것을 이용한 문제였다. 

중간에 CRIM까지 찾은 이후에는 융통성으로 E를 직접 추가해주어야 한다. 

ex.py
```python
import requests
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import long_to_bytes, bytes_to_long


def encrypt(plaintext):
    url = "http://aes.cryptohack.org/ctrime/encrypt/"
    url += plaintext.hex()
    url += "/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["ciphertext"])

def xor(a, b):
    xored = b""
    for i in range(len(a)):
        xored += (a[i] ^ b[i]).to_bytes(1, byteorder = "big")

    return xored

# flag = "crypto{"
flag = "crypto{CRIME"

while(1):
    base = len(encrypt((flag + chr(0)).encode()))
    for i in range(33, 128):
        print(i)
        if len(encrypt((flag + chr(i)).encode())) < base:
            flag += chr(i)
            break

    print(flag)
    if flag[-1] == "}":
        break
```

flag
```
crypto{CRIME_571ll_p4y5}
```

</br></br></br>

# Stream Ciphers - Logon Zero

어려워 보이지만, token을 가장 간단하게 주었을 때 결과를 생각해보면 된다. 

token = b"\x00" * 28 이라면 decrypt 과정에서 state가 항상 동일해서 decrypt한 결과는 같은 바이트만 12개가 넘어가게 된다. 

그 바이트가 \x00이라면 password = ""

아니라면 password = chr(i) * 8이 될 것이다. 

256가지를 모두 돌려주면 된다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13399)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

to_send = {
    "option": "reset_password", 
    "token": "00" * 28
}

json_send(to_send)
json_recv()

for i in range(256):
    password = ""
    if i > 0:
        password = 8 * chr(i)
    to_send = {
        "option": "authenticate", 
        "password": password
    }
    json_send(to_send)

    msg = json_recv()["msg"]

    print(i)
    if msg != "Wrong password.":
        print(msg)
        break

r.interactive()
```

flag
```
crypto{Zerologon_Windows_CVE-2020-1472}
```
</br></br></br>

# Stream Ciphers - Stream of Consciousness

모든 단어를 긁어온 다음에 공통된 xor 키가 무엇일지 알아내면 된다. 

"crypto{" 로부터 시작하는 것은 당연하다. 

ex.py
```python
import requests
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import long_to_bytes, bytes_to_long


def encrypt():
    url = "http://aes.cryptohack.org/stream_consciousness/encrypt/"
    r = requests.get(url)
    js = r.json()
    return bytes.fromhex(js["ciphertext"])

def xor(a, b):
    xored = b""
    for i in range(len(a)):
        xored += (a[i] ^ b[i]).to_bytes(1, byteorder = "big")

    return xored

words = []

cnt = 22

'''
while(1):
    ciphertext = encrypt()

    if not ciphertext in words:
        words.append(ciphertext)
        print(len(words))
    
    if len(words) == cnt:
        break

print(words)
'''

words = 생략, 위 주석 부분의 출력을 복붙하면 된다. 


key = ""
plain = "crypto{"
flag_idx = 4

def flag_dig(idx, c):
    global key
    key += chr(ord(c) ^ words[idx][len(key)])
    show()

def show():
    for i in range(cnt):
        plaintext = ""
        for j in range(min(len(key), len(words[i]))):
            plaintext += chr(ord(key[j]) ^ words[i][j])
        print(f"{i:02}: " + plaintext)
    print()

for i in range(7):
    key += chr(ord(plain[i]) ^ words[flag_idx][i])

show()

flag_dig(20, "l")
flag_dig(7, " ")
flag_dig(9, "l")
flag_dig(9, "l")
flag_dig(5, "b")
flag_dig(5, "l")
flag_dig(5, "y")
flag_dig(9, "n")
flag_dig(9, "o")
flag_dig(9, "r")
flag_dig(9, "e")
flag_dig(14, "v")
flag_dig(14, "e")
flag_dig(2, "i")
flag_dig(2, "n")
flag_dig(2, "g")
flag_dig(7, "y")
flag_dig(7, "t")
flag_dig(7, "h")
flag_dig(7, "i")
flag_dig(7, "n")
flag_dig(7, "g")
flag_dig(3, "v")
flag_dig(5, "w")
flag_dig(3, "n")
flag_dig(3, "g")
```

아래의 flag_dig 함수들은 출력 결과를 보면서 수작업으로 추측을 진행해야 한다. 

출력 결과
```
00: Our? Why our?
01: It can't be torn out, but it can
02: I shall lose everything and not g
03: Dolly will think that I'm leaving
04: crypto{k3y57r34m_r3u53_15_f474l}
05: Love, probably? They don't know h
06: The terrible thing is that the pa
07: I shall, I'll lose everything if
08: As if I had any wish to be in the
09: And I shall ignore it.
10: But I will show him.
11: Dress-making and Millinery
12: What a nasty smell this paint had
13: What a lot of things that then se
14: Would I have believed then that I
15: I'm unhappy, I deserve it, the fa
16: How proud and happy he'll be when
17: These horses, this carriage - how
18: Three boys running, playing at ho
19: Why do they go on painting and bu
20: No, I'll go in to Dolly and tell
21: Perhaps he has missed the train a
```

flag
```
crypto{k3y57r34m_r3u53_15_f474l}
```

</br></br></br>

# Stream Ciphers - Dancing Queen

리버싱이다.

암호화하는 함수를 거꾸로 뒤집어서 만들어주면 키를 바로 구할 수 있다. 

class ChaCha20:
```python
def _inner_block(self, state):
    self._quarter_round(state, 0, 4, 8, 12)
    self._quarter_round(state, 1, 5, 9, 13)
    self._quarter_round(state, 2, 6, 10, 14)
    self._quarter_round(state, 3, 7, 11, 15)
    self._quarter_round(state, 0, 5, 10, 15)
    self._quarter_round(state, 1, 6, 11, 12)
    self._quarter_round(state, 2, 7, 8, 13)
    self._quarter_round(state, 3, 4, 9, 14)

def _inner_reverse(self, state):
    self._quarter_reverse(state, 3, 4, 9, 14)
    self._quarter_reverse(state, 2, 7, 8, 13)
    self._quarter_reverse(state, 1, 6, 11, 12)
    self._quarter_reverse(state, 0, 5, 10, 15)
    self._quarter_reverse(state, 3, 7, 11, 15)
    self._quarter_reverse(state, 2, 6, 10, 14)
    self._quarter_reverse(state, 1, 5, 9, 13)
    self._quarter_reverse(state, 0, 4, 8, 12)

def _quarter_round(self, x, a, b, c, d):
    x[a] = word(x[a] + x[b]); x[d] ^= x[a]; x[d] = rotate(x[d], 16)
    x[c] = word(x[c] + x[d]); x[b] ^= x[c]; x[b] = rotate(x[b], 12)
    x[a] = word(x[a] + x[b]); x[d] ^= x[a]; x[d] = rotate(x[d], 8)
    x[c] = word(x[c] + x[d]); x[b] ^= x[c]; x[b] = rotate(x[b], 7)

def _quarter_reverse(self, x, a, b, c, d):
    x[b] = rotate(x[b], 25); x[b] ^= x[c]; x[c] = word(x[c] - x[d])
    x[d] = rotate(x[d], 24); x[d] ^= x[a]; x[a] = word(x[a] - x[b])
    x[b] = rotate(x[b], 20); x[b] ^= x[c]; x[c] = word(x[c] - x[d])
    x[d] = rotate(x[d], 16); x[d] ^= x[a]; x[a] = word(x[a] - x[b])
```

ex.py(뒷부분만)
```python
xored = bytes_to_words(xor(msg, msg_enc)[:64])

C = ChaCha20()
C._state = xored

for _ in range(10):
	C._inner_reverse(C._state)

key_word = []
for i in range(4, 12):
	key_word.append(C._state[i])

key = words_to_bytes(key_word)

C = ChaCha20()

flag = C.decrypt(flag_enc, key, iv2)

print(flag)
```

flag
```
crypto{M1x1n6_r0und5_4r3_1nv3r71bl3!}
```

</br></br></br>

# Stream Ciphers - Oh SNAP

RC4 key scheduling algorithm을 알고, 안으로 깊숙히 들여다봐야 볼 수 있는 문제이다. 

https://github.com/dj311/rc4-key-recovery-attacks

여기에 있는 모듈은 사용이 조금 어렵지만, md 파일에 굉장히 상세하게 잘 나와 있어서 이 공격을 이해할 수 있었다. 

보아하니 한 바이트를 얻는 데 약 50000, 안전하려면 100000개 이상의 샘플이 필요해 보였다. 그리고 모든 바이트당 그 샘플을 얻으려면 3000000개의 샘플이라는 말도 안되는 수치가 나온다. 

하지만 조금 생각을 달리해서 모든 데이터를 파일에 저장해 놓고 난 후에, 로컬에서 그 정보를 읽으면서 바이트를 찾으면 효율적일 것이라는 생각이 들었다. 

get_data.py
```python
import requests
from Crypto.Util.number import *
import os

def send_cmd(ciphertext, nonce):
    url = f"http://aes.cryptohack.org/oh_snap/send_cmd/{bytes.hex(ciphertext)}/{bytes.hex(nonce)}/"
    return requests.get(url).json()

rnd = 0
while 1:
	rnd += 1
	filename = "data_" + bytes.hex(os.urandom(4)) + ".txt"

	f = open(f"datas/{filename}", "w")

	try:
		cnt = 0
		while 1:
			cnt += 1
			send = os.urandom(222)
			recv = bytes.fromhex(send_cmd(b"\x00" * 256, send)["error"][17:])

			assert len(recv) == 256
			f.write(f"{cnt}\n")
			f.write(bytes.hex(send))
			f.write("\n")
			f.write(bytes.hex(recv))
			f.write("\n\n")
			print(f"round {rnd}: {cnt}")

	except:
		f.close()
		continue
```

이걸로 생긴 파일들을 모두 읽어서 리스트에 저장한 후, 그 리스트의 데이터를 바탕으로 플래그를 구한다. 
50000으로는 부족했고, 100000개 정도의 데이터로는 충분했다. 

ex.py
```python
import requests
from Crypto.Util.number import *
import os

def read_f(f):
    try:
        f.readline()
        send = bytes.fromhex(f.readline())
        recv = bytes.fromhex(f.readline())
        f.readline()

        assert len(send) == 222 and len(recv) == 256
        return (send, recv)
    except:
        return None


fdir = "datas/"
files = os.listdir(fdir)

datas = []

for file in files:
    f = open(fdir + file, "r")
    
    while 1:
        data = read_f(f)
        if data == None:
            break

        datas.append(data)




def KSA_t(key):
    s = [i for i in range(256)]
    
    j = 0
    for i in range(len(key)):
        j = (j + s[i] + key[i]) % 256
        s[i], s[j] = s[j], s[i]

    return s, j

flag_len = 34

flag = b""

for idx in range(34):
    cnt = 0
    lst = [0] * 256

    new_c = 0

    for i in range(len(datas)):
        cnt += 1

        key, res = datas[i]

        assert len(key) == 222
        assert len(res) == 256

        s, j = KSA_t(key + flag[:idx])

        val = (256 - flag_len + idx) - res[(256 - flag_len - 1 + idx)]
        
        new_j = 0
        for i in range(256):
            if s[i] == val:
                new_j = i
                break

        key_byte = (new_j - s[256 - flag_len + idx] - j) % 256

        lst[key_byte] += 1

        maxx = 0
        maxc = 0
        for i in range(32, 127):
            if lst[i] > maxx:
                maxx = lst[i]
                maxc = i
        new_c = maxc

        print(f"flag: {flag},  count: {cnt},  {bytes([maxc])}: {maxx / cnt * 256}")

    flag += bytes([new_c])

print(flag)
```

flag
```
crypto{w1R3d_equ1v4l3nt_pr1v4cy?!}
```

풀이들을 보니 다른 풀이도 많아서 신기했다. 더 쉬운 방법 또한 존재했다. 

---
</br></br></br>

# Authenticated Encryption - Paper Plane

패딩 정보가 맞는지의 여부를 검사하는 기능만이 존재한다. 

블록 암호의 과정을 잘 봐보면 plaintext ^ (우리가 원하는 값) 의 패딩 일치 여부를 알 수 있다. 

한 바이트당 최대 96번의 request를 통해 브루트포스로 플래그를 알아낼 수 있다. 

ex.py
```python
import requests
from Crypto.Cipher import AES
from Crypto.Util.number import *
from Crypto.Util.Padding import pad, unpad

def encrypt_flag():
	url = f"http://aes.cryptohack.org/paper_plane/encrypt_flag/"
	r = requests.get(url)
	js = r.json()
	return js

def send_msg(ct, m, c):
	url = f"http://aes.cryptohack.org/paper_plane/send_msg/{bytes.hex(ct)}/{bytes.hex(m)}/{bytes.hex(c)}/"
	r = requests.get(url)
	js = r.json()
	return js

def xor_blocks(a, b):
	return bytes([x ^ y for x, y in zip(a, b)])

flag_data = encrypt_flag()
m0 = bytes.fromhex(flag_data["m0"])
c0 = bytes.fromhex(flag_data["c0"])
enc_flag = bytes.fromhex(flag_data["ciphertext"])
b1 = enc_flag[:16]
b2 = enc_flag[16:]

print(len(b2))

p1 = b"\x00" * 16
p2 = b"\xff" * 16

for i in range(16):
	for j in range(32, 128):
		check = b"\x00" * (15 - i) + long_to_bytes(i + 1) * (i + 1)
		p1 = p1[:15 - i] + long_to_bytes(j) + p1[16 - i:]
		assert len(check) == 16
		if "msg" in send_msg(b1, m0, xor_blocks(c0, xor_blocks(check, p1))):
			print(f"yaaaaaay!: {p1[15 - i:]}")
			break
		else:
			print(f"f*ck: {p1[15 - i:]}")


for i in range(16):
	for j in range(0, 256):
		check = b"\x00" * (15 - i) + long_to_bytes(i + 1) * (i + 1)
		p2 = p2[:15 - i] + long_to_bytes(j) + p2[16 - i:]
		assert len(check) == 16
		if "msg" in send_msg(b2, p1, xor_blocks(b1, xor_blocks(check, p2))):
			print(f"yaaaaaay!: {p2[15 - i:]}")
			break
		else:
			print(f"f*ck: {p2[15 - i:]}")

print(unpad(p1 + p2, 16))
```

flag
```
crypto{h3ll0_t3l3gr4m}
```

</br></br></br>

# Authenticated Encryption - Forbidden Fruit

GCM모드는 결국 Modulo를 가진 GF(2^128)에서의 polynomial 연산임을 이해하고 나면 어렵지 않다. 

sage 내에서 구현하는 방법은 rbtree님의 블로그를 참고하였다. 

https://rbtree.blog/posts/2022-03-27-sage-script-for-aes-gcm/#lazy-stek-solution

이 문제의 경우 ct 블록만 다르고, len 블록과 ad블록이 모두 같기 때문에 y값을 알아낸 후에 그저 ct블록을 바꿔치기하는 것만으로 해결 가능하다. 

ex.sage
```python
from Crypto.Cipher import AES
import os
import requests

F.<a> = GF(2^128, modulus=x^128 + x^7 + x^2 + x + 1)
P.<x> = PolynomialRing(F)

def bytes_to_poly(b):
    v = int.from_bytes(b, 'big')
    v = int(f"{v:0128b}"[::-1], 2)
    return F.fetch_int(v)

def poly_to_bytes(p):
    v = p.integer_representation()
    v = int(f"{v:0128b}"[::-1], 2)
    return v.to_bytes(16, 'big')

def encrypt(pt):
	url = f"http://aes.cryptohack.org/forbidden_fruit/encrypt/{bytes.hex(pt)}/"
	return requests.get(url).json()

def decrypt(nonce, ct, tag, ad):
	url = f"http://aes.cryptohack.org/forbidden_fruit/decrypt/{bytes.hex(nonce)}/{bytes.hex(ct)}/{bytes.hex(tag)}/{bytes.hex(ad)}/"
	return requests.get(url).json()

pt1 = b"soonharisoonhari"
pt2 = b"harisoonharisoon"
pt3 = b"give me the flag"

ad = bytes.fromhex('43727970746f4861636b') # useless really except for verifying

data1 = encrypt(pt1)
ct1 = bytes.fromhex(data1["ciphertext"])
tag1 = bytes.fromhex(data1["tag"])

nonce = bytes.fromhex(data1["nonce"])

data2 = encrypt(pt2)
ct2 = bytes.fromhex(data2["ciphertext"])
tag2 = bytes.fromhex(data2["tag"])

data3 = encrypt(pt3)
ct3 = bytes.fromhex(data3["ciphertext"])

f = (bytes_to_poly(ct1) + bytes_to_poly(ct2)) * x^2 + (bytes_to_poly(tag1) + bytes_to_poly(tag2))

root, _ = f.roots()[0]

tag3_poly = bytes_to_poly(tag1) + root^2 * (bytes_to_poly(ct1) + bytes_to_poly(ct3))
tag3 = poly_to_bytes(tag3_poly)

print(bytes.fromhex(decrypt(nonce, ct3, tag3, ad)["plaintext"]))
```

flag
```
crypto{https://github.com/attr-encrypted/encryptor/pull/22}
```

---
</br></br></br>

# Linear Cryptanalysis - Beatboxer

AES 시스템을 구현해놓았다. 

sbox만 다르고 나머지는 실제 AES와 정확히 일치하였다. (실제 sbox로 바뀌주니 같은 결과값은 나타냄을 확인할 수 있었다.)

고민하다가 5unkn0wn님의 sbox 취약점에 관련된 글을 추천받아 보게 되었다.

http://blog.0wn.kr/codegate2022-final-aesmaster/

결과적으로는 sbox가 선형으로 구현되어 있고, AES는 sbox 치환을 제외하고는 항상 선형적으로 구현되어 있기 때문에, 이 AES 시스템은 선형이라는 것을 알 수 있다. 

`b"\x00" * 16` 블록과 임의의 블록을 xor하면 key의 값에 상관없이 항상 같은 결과를 가지고,

평문을 xor한 것의 암호문은 암호문을 xor한 것과 일치한다는 특징을 사용할 수 있다. 

</br>

128개 비트별로 128개의 암호문을 저장해서 어떤 것들을 xor해야 해당하는 암호문이 나오는지,

즉 128*128 GF(2) 행렬의 역행렬을 이용해서 복호화가 가능한다. 

ex.sage
```python
from os import urandom

class AES2:
    {생략}

from Crypto.Util.number import *
import os

key = os.urandom(16)
aes = AES2(key)

def xor(a, b):
    res = []
    for i in range(16):
        res.append(a[i] ^^ b[i])
    return bytes(res)

def encrypt(pt):
    return xor(aes.encrypt(b"\x00" * 16), aes.encrypt(pt))

mat = Matrix(GF(2), 128, 128)

for i in range(128):
    send_block = long_to_bytes(1 << i)
    send_block = b"\x00" * (16 - len(send_block)) + send_block
    assert len(send_block) == 16
    assert bytes_to_long(send_block) == (1 << i)

    enc = bytes_to_long(encrypt(send_block))

    for j in range(128):
        mat[j, i] = enc % 2
        enc //= 2

matinv = mat.inverse()

def decrypt(ct):
    ct_v = Matrix(GF(2), 128, 1)
    enc = bytes_to_long(ct)

    for i in range(128):
        ct_v[i, 0] = enc % 2
        enc //= 2

    pt_v = matinv * ct_v
    pt = 0

    for i in range(128):
        pt += int(pt_v[i, 0]) << i

    return long_to_bytes(pt)



test_block_enc = encrypt(b"soon_haari_idiot")
assert decrypt(test_block_enc) == b"soon_haari_idiot"






from pwn import *
import json
from Crypto.Util.Padding import pad, unpad

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r = remote('socket.cryptohack.org', 13406)

r.recvline()

json_send({
    "option": "encrypt_message",
    "message": bytes.hex(b"\x00" * 16)
    })
enc_0 = bytes.fromhex(json_recv()["encrypted_message"])
assert len(enc_0) == 16

json_send({
    "option": "encrypt_flag",
    })
enc_flag = bytes.fromhex(json_recv()["encrypted_flag"])
assert len(enc_flag) == 48

flag = b""

for i in range(3):
    flag += decrypt(xor(enc_0, enc_flag[16 * i:16 * (i + 1)]))

print(unpad(flag, 16))



r.interactive()
```

flag
```
crypto{5b0x_l1n34r17y_15_d35457r0u5}
```

---
</br></br></br>