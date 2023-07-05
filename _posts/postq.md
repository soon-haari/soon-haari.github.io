# LWE - LWE Background

oded regev

</br></br></br>

# LWE - LWE Intro

Gaussian elimination

</br></br></br>

# LWE - LWE High Bits Message

ex.sage
```python
S =  (55542, 19411, 34770, 6739, 63198, 63821, 5900, 32164, 51223, 38979, 24459, 10936, 17256, 20215, 35814, 42905, 53656, 17000, 1834, 51682, 43780, 22391, 33012, 61667, 37447, 16404, 58991, 61772, 44888, 43199, 32039, 26885, 17206, 62186, 58387, 57048, 38393, 29306, 58001, 57199, 33472, 56572, 53429, 62593, 14134, 40522, 25106, 34325, 37646, 43688, 14259, 24197, 33427, 43977, 18322, 38877, 55093, 12466, 16869, 25413, 54773, 59532, 62694, 13948) 
A =  (13759, 12750, 38163, 63722, 39130, 22935, 58866, 48803, 15933, 64995, 60517, 64302, 42432, 32000, 22058, 58123, 53993, 33790, 35783, 61333, 53431, 43016, 60795, 25781, 28091, 11212, 64592, 11385, 24690, 40658, 35307, 63583, 60365, 60359, 32568, 35417, 22078, 38207, 16331, 53636, 28734, 30436, 18170, 15939, 966, 48519, 41621, 36371, 41836, 4026, 33536, 57062, 52428, 59850, 476, 43354, 61614, 32243, 42518, 19733, 63464, 29357, 56039, 15013)
b =  44007

n = 64
p = 257
q = 0x10001

delta = int(round(q/p))

S = vector(GF(q), S)
A = vector(GF(q), A)

m = int(round(int((b - S * A)) / delta))

print(m)
```

```
201
```

</br></br></br>

# LWE - LWE Low Bits Message

ex.sage
```python
S =  (10082, 48747, 17960, 55638, 37012, 51876, 10128, 37750, 7608, 58952, 33296, 25463, 38900, 85, 65248, 42153, 44966, 31594, 40676, 56828, 30325, 38502, 65083, 7497, 2667, 54022, 24029, 32162, 57267, 12253, 6668, 5181, 14906, 51655, 61347, 4722, 22227, 23606, 63183, 52860, 1670, 31085, 42633, 47197, 7255, 16150, 9574, 62956, 26742, 57998, 49467, 31224, 60073, 12730, 41419, 41042, 53032, 16339, 32913, 16351, 34283, 47845, 3617, 35718) 
A =  (53751, 21252, 55954, 16345, 60990, 2822, 56279, 37048, 36153, 52141, 2121, 56565, 48112, 43755, 12951, 22539, 29478, 28421, 62175, 10265, 36378, 21305, 42402, 26359, 939, 60690, 1161, 65097, 34505, 19777, 29652, 42868, 49148, 38296, 31916, 25606, 18700, 12655, 35631, 64674, 29018, 21021, 14865, 40196, 14036, 40278, 37209, 35585, 34344, 33030, 285, 58536, 56121, 40899, 24262, 62326, 57433, 5765, 24456, 28859, 45170, 14799, 21567, 55484)
b =  11507


n = 64
p = 257
q = 0x10001

delta = int(round(q/p))

error_bound = int(floor((q/p)/2))

S = vector(GF(q), S)
A = vector(GF(q), A)

val = int(b - S * A)

if val > error_bound * p:
	val -= q

m = val % p

print(m)
```

```
147
```

</br></br></br>

# LWE - From Private to Public Key LWE

https://openquantumsafe.org/liboqs/algorithms/kem/kyber.html

```
1568
```

</br></br></br>

# LWE - Noise Free

그냥 역행렬 쓰는 선형대수학이다. 

ex.sage
```python
from pwn import *
from Crypto.Util.number import *
import json

r = remote('socket.cryptohack.org', 13411)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.readline()

n = 64
# plaintext modulus
p = 257
# ciphertext modulus
q = 0x10001

m = []
res = []

for i in range(n):
    json_send({
        "option": "encrypt",
        "message": "0"
        })
    data = json_recv()

    print(data)
    A = json.loads(data["A"])
    b = int(data["b"])
    m.append(A)
    res.append(b)

m = Matrix(GF(q), m)
res = Matrix(GF(q), n, res)

S = m.inverse() * res

flag = ""

for i in range(len(b"crypto{????????????????????????}")):
    json_send({
        "option": "get_flag",
        "index": str(i)
        })

    data = json_recv()

    print(data)
    A = json.loads(data["A"])
    b = int(data["b"])

    A = Matrix(GF(q), A)
    m = b - (A * S)[0][0]
    flag += chr(m)
    print(flag)

r.interactive()
```

flag
```
crypto{linear_algebra_is_useful}
```

</br></br></br>

# LWE - Nativity

뒤에 나올 noise cheap에서 사용하는 방법을 사용하면 된다....

GF(65536)을 사용하니깐 GF(2)와 같이 행동하는 걸 알 수 있었다.

IntegerModRing(65536)을 사용해야 우리가 원하는 결과를 얻을 수 있다. 

이 문제의 경우는 비트 연산을 사용하기 때문에 GF(2)를 사용해도 지장은 없다. 

ex.sage
```python
from Crypto.Util.number import *

A_origin = []
b_origin = []

n = 64
m = 128
q = 65536

f = open("public_key.txt", "r")

for i in range(n):
	sp = f.readline().split()
	line = []
	for val in sp:
		line.append(int(val))
	A_origin.append(vector(IntegerModRing(q), line))



sp = f.readline().split()
for val in sp:
	b_origin.append(int(val))

b_origin = vector(IntegerModRing(q), b_origin)


from sage.modules.free_module_integer import IntegerLattice

def Babai_closest_vector(M, G, target):
    small = target
    for _ in range(1):
        for i in reversed(range(M.nrows())):
            c = ((small * G[i]) / (G[i] * G[i])).round()
            small -= M[i] * c
    return target - small


'''
A = matrix(ZZ, m + n, m)
for i in range(m):
    A[i, i] = q

for i in range(n):
	for j in range(m):
		A[i + m, j] = A_origin[i][j]

print("LLL start")
lattice = IntegerLattice(A, lll_reduce=True)
print("LLL done")
gram = lattice.reduced_basis.gram_schmidt()[0]
print("gram-schmidt done")

print()
target = vector(ZZ, b_origin[:m])
res = Babai_closest_vector(lattice.reduced_basis, gram, target)
print("Closest Vector: {}".format(res))
print("Difference: {}".format(target - res))
'''

res = [13513, 51163, 36034, 13140, 15769, 16315, 43051, 61223, 29294, 62882, 42254, 22852, 15974, 7335, 45646, 64417, 62137, 31792, 46704, 59371, 55434, 21765, 27327, 59847, 34792, 56479, 53425, 37386, 7574, 18854, 29038, 44261, 33545, 12415, 54621, 65444, 5291, 1068, 27598, 15692, 32382, 57173, 34349, 59374, 46486, 37806, 33155, 18186, 34238, 13668, 9516, 39120, 10925, 38781, 54928, 21909, 4290, 58226, 20153, 30326, 1648, 9201, 21653, 31210, 6753, 34208, 19604, 34063, 38621, 19417, 16587, 19268, 8841, 52330, 58160, 45229, 51996, 36584, 550, 31341, 5175, 53996, 9199, 16169, 61901, 53727, 65102, 29750, 50333, 35023, 169, 59404, 5906, 29998, 18473, 31117, 8693, 61451, 19755, 30407, 38520, 2780, 36860, 32132, 4213, 55990, 24321, 58347, 41884, 2704, 45529, 27007, 40699, 14206, 57575, 61180, 22156, 16225, 22883, 57647, 35986, 7533, 55140, 40350, 36165, 48567, 15322, 26366]
res = vector(IntegerModRing(q), res)


A_mat = Matrix(IntegerModRing(q), m, n)

for i in range(m):
	for j in range(n):
		A_mat[i, j] = A_origin[j][i]

S = A_mat.solve_right(res)

print(S)

assert res == A_mat * S

sk = []
for i in range(64):
	sk.append(int(-S[i]) % q)

sk.append(1)

f = open("ciphertexts.txt", "r")

flag = 0

for i in range(392):
	sp = f.readline().split()
	line = []
	for val in sp:
		line.append(int(val))
	
	flag_bit = 0
	for j in range(65):
		flag_bit += int(line[j] * sk[j])

	flag_bit &= 1
	flag |= flag_bit << (391 - i)


print(long_to_bytes(flag))
```

flag
```
crypto{flavortext-flag-coprime-regev-yadda-yadda}
```

</br></br></br>

# LWE - Missing Modulus

복잡하게 생각하지 않고, 처음에 의심해 봤던 걸 실행만 하면 됐었다...

a^-1 * b를 계산하고 round하면 S가 나온다. 

ex.sage
```python
from pwn import *
from Crypto.Util.number import *
import json

r = remote('socket.cryptohack.org', 13412)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

n = 512

def Babai_closest_vector(M, G, target):
    small = target
    for _ in range(1):
        for i in reversed(range(M.nrows())):
            c = ((small * G[i]) / (G[i] * G[i])).round()
            small -= M[i] * c
    return target - small


r.recvline()

A_mat = matrix(ZZ, n, n)
b_mat = matrix(ZZ, n, 1)

for i in range(n):
    print(i)
    json_send({
        "option": "encrypt",
        "message": "0"
        })
    data = json_recv()
    A_recv = json.loads(data["A"])
    b_recv = int(data["b"])
    
    for j in range(n):
        A_mat[i, j] = A_recv[j]

    b_mat[i, 0] = b_recv

S = A_mat.inverse() * b_mat

for i in range(n):
    S[i, 0] = int(round(S[i, 0]))

p = 257
# ciphertext modulus
q = 6007
# message scaling factor
delta = int(round(q/p))

l = len(b'crypto{??????????????????????????????????????}')

flag = ""

for i in range(l):
    json_send({
        "option": "get_flag",
        "index": str(i)
        })
    data = json_recv()
    A_recv = json.loads(data["A"])
    b_recv = int(data["b"])
    A_recv = matrix(ZZ, A_recv)
    val = b_recv - (A_recv * S)[0, 0]

    flag += chr(int(round(val / delta)))

    print(flag)



r.interactive()
```

flag
```
crypto{learning-is-easy-over-the-real-numbers}
```

</br></br></br>

# LWE - Noise Cheap

CVP, Babai 알고리즘에 대해서 알게 되었다.

https://hackmd.io/@hakatashi/B1OM7HFVI

이 링크를 참고하였다. 

LLL과 gram-schmidt 과정을 이용한 기법으로

특정 벡터와 가장 거리가 가까운 벡터를 찾을 수 있다. 

그런데 에러를 고려하면 벡터와의 거리가 257, 0, -257이기 때문에 q에 대한 p의 inverse를 곱해서 에러를 작게 만들어준다. 

ex.sage
```python
from pwn import *
from Crypto.Util.number import *
import json
from sage.modules.free_module_integer import IntegerLattice

r = remote('socket.cryptohack.org', 13413)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

q = 1048583
n = 64
m = 128
p = 257

def Babai_closest_vector(M, G, target):
    small = target
    for _ in range(1):
        for i in reversed(range(M.nrows())):
            c = ((small * G[i]) / (G[i] * G[i])).round()
            small -= M[i] * c
    return target - small


r.recvline()

A_mat = matrix(GF(q), n, n)
b_mat = matrix(GF(q), n, 1)

b_values = []

A = matrix(ZZ, m + n, m)
for i in range(m):
    A[i, i] = q

for i in range(m):
    print(i)
    json_send({
        "option": "encrypt",
        "message": "0"
        })
    data = json_recv()
    A_recv = json.loads(data["A"])
    b_recv = int(data["b"])
    
    if i < n:
        for j in range(n):
            A_mat[i, j] = (A_recv[j] * pow(p, -1, q)) % q

    for j in range(n):
        A[m + j, i] = (A_recv[j] * pow(p, -1, q)) % q

    b_values.append((b_recv * pow(p, -1, q)) % q)

lattice = IntegerLattice(A, lll_reduce=True)
print("LLL done")
gram = lattice.reduced_basis.gram_schmidt()[0]
target = vector(ZZ, b_values)
res = Babai_closest_vector(lattice.reduced_basis, gram, target)
print("Closest Vector: {}".format(res))
print("Difference: {}".format(target - res))

for i in range(n):
    b_mat[i, 0] = res[i]

S = A_mat.inverse() * b_mat



l = len(b"crypto{????????????????????????}")

flag = ""

for i in range(l):
    json_send({
        "option": "get_flag",
        "index": str(i)
        })
    data = json_recv()
    A_recv = json.loads(data["A"])
    b_recv = int(data["b"])
    A_recv = matrix(GF(q), A_recv)
    val = int(b_recv - (A_recv * S)[0, 0])
    if val > q // 2:
        val -= q

    flag += chr(val % p)

    print(flag)



r.interactive()
```

flag
```
crypto{LLL_is_also_very_useful!}
```

</br></br></br>

# LWE - Too Many Errors

get_sample -> reset을 반복하면

조작마다 절반 확률로 한 바이트가 변조된 a배열이 나오고 절반 확률로 올바른 a배열이 나온다. 

즉 조작을 두 번 했을 때 나오는 두 a배열이 같다면 거의 100%의 확률로 올바른 a배열이라고 할 수 있다. 이 확률은 1/4이므로 4번만 connect, close를 반복하면 될 것으로 기대할 수 있다. 

나머지는 변조된 바이트별로 b값차이을 이용해 찾아주면 된다. 

ex.py
```python
from Crypto.Util.number import *
from pwn import *
import json

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

while 1:
    r = remote('socket.cryptohack.org', 13390)

    r.recvline()
    l = len(b'crypto{????????????????????}')


    json_send({
        "option": "get_sample"
        })
    res = json_recv()
    json_send({
        "option": "reset"
        })
    json_recv()

    json_send({
        "option": "get_sample"
        })
    test_res = json_recv()
    json_send({
        "option": "reset"
        })
    json_recv()

    a = res['a']
    b = res['b']
    test_a = test_res['a']
    test_b = test_res['b']

    if a != test_a or b != test_b:
        r.close()
        continue

    flag = [-1] * l
    done_cnt = 0

    while 1:
        if done_cnt == l:
            break

        json_send({
            "option": "get_sample"
            })
        new_res = json_recv()
        json_send({
            "option": "reset"
            })
        json_recv()
        new_a = new_res['a']
        new_b = new_res['b']

        cnt = 0
        idx = -1
        for i in range(l):
            if a[i] != new_a[i]:
                idx = i
                break
        if idx == -1:
            continue

        if flag[idx] != -1:
            continue

        flag[idx] = ((b - new_b) * pow(a[idx] - new_a[idx], -1, 127)) % 127
        done_cnt += 1

        flag_str = ""
        for i in range(l):
            if flag[i] == -1:
                flag_str += '?'
            else:
                flag_str += chr(flag[i])
        print(flag_str)

    r.interactive()
```

flag
```
crypto{f4ult_4ttack5_0n_lw3}
```
---
</br></br></br>