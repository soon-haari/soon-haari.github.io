# Modular - Quadratic Residues

제곱 항의 나머지에 대해 설명한다. 

ex.py
```python
p = 29
ints = [14, 6, 11]

for i in range(p):
	if i * i % p in ints:
		print(i)
```

answer
```
8
```

</br></br></br>

# Modular - Legendre Symbol

```
a ^ ((p - 1) / 2) = 1 (mod p)
```
가 제곱근을 가질 조건이다.

제곱근은 p를 4로 나눈 나머지에 따라 다르게 표현된다. 

3인 경우는 굉장히 간단히 표현된다:
```
a ^ ((p + 1) / 4)
```

ex.py
```python
p = 101524035174539890485408575671085261788758965189060164484385690801466167356667036677932998889725476582421738788500738738503134356158197247473850273565349249573867251280253564698939768700489401960767007716413932851838937641880157263936985954881657889497583485535527613578457628399173971810541670838543309159139

ints = [생략]

for i in range(len(ints)):
	if pow(ints[i], (p - 1) // 2, p) == 1:
		print(pow(ints[i], (p + 1) // 4, p))
```

answer
```
93291799125366706806545638475797430512104976066103610269938025709952247020061090804870186195285998727680200979853848718589126765742550855954805290253592144209552123062161458584575060939481368210688629862036958857604707468372384278049741369153506182660264876115428251983455344219194133033177700490981696141526
```
</br></br></br>

# Modular - Modular Square Root

sage를 이용해서 바로 답을 구해주는 문제이다.

토넬리 샹크스 알고리즘을 이용한다. 

ex.py
```python
from sage.all import *

a = 8479994658316772151941616510097127087554541274812435112009425778595495359700244470400642403747058566807127814165396640215844192327900454116257979487432016769329970767046735091249898678088061634796559556704959846424131820416048436501387617211770124292793308079214153179977624440438616958575058361193975686620046439877308339989295604537867493683872778843921771307305602776398786978353866231661453376056771972069776398999013769588936194859344941268223184197231368887060609212875507518936172060702209557124430477137421847130682601666968691651447236917018634902407704797328509461854842432015009878011354022108661461024768
p = 30531851861994333252675935111487950694414332763909083514133769861350960895076504687261369815735742549428789138300843082086550059082835141454526618160634109969195486322015775943030060449557090064811940139431735209185996454739163555910726493597222646855506445602953689527405362207926990442391705014604777038685880527537489845359101552442292804398472642356609304810680731556542002301547846635101455995732584071355903010856718680732337369128498655255277003643669031694516851390505923416710601212618443109844041514942401969629158975457079026906304328749039997262960301209158175920051890620947063936347307238412281568760161

print(Mod(a, p).sqrt())
```

answer
```
2362339307683048638327773298580489298932137505520500388338271052053734747862351779647314176817953359071871560041125289919247146074907151612762640868199621186559522068338032600991311882224016021222672243139362180461232646732465848840425458257930887856583379600967761738596782877851318489355679822813155123045705285112099448146426755110160002515592418850432103641815811071548456284263507805589445073657565381850521367969675699760755310784623577076440037747681760302434924932113640061738777601194622244192758024180853916244427254065441962557282572849162772740798989647948645207349737457445440405057156897508368531939120
```

</br></br></br>

# Modular - Chinese Remainder Theorem

중국인의 나머지 정리

작은 수들이라서 그냥 가장 쉬운 방법을 사용했다.

```python
for i in range(935):
	if i % 5 == 2 and i % 11 == 3 and i % 17 == 5:
		print(i)
```
---
</br></br></br>

# Lattices - Vectors

백터의 정의와 내적을 다룬다.

뭔가 sage에 이미 벡터 자료형이 정의되어 있을 것 같다. 

ex.py
```python
v = [2, 6, 3]
w = [1, 0, 0]
u = [7, 7, 2]

ans = 0
for i in range(3):
	ans += 3 * (2 * v[i] - w[i]) * 2 * u[i]

print(ans)
```

answer
```
702
```

</br></br></br>

# Lattices - Size and Basis

벡터의 크기를 구한다. 

ex.py
```python
import math

v = [4, 6, 2, 5]

size_2 = 0
for i in range(4):
	size_2 += v[i] ** 2

print(math.sqrt(size_2))
```

answer
```
9
```

</br></br></br>

# Lattices - Gram Schmidt

orthonormal basis를 만드는 과정이다. 

선형대수학의 과정을 그대로 따라간다. 

ex.py
```python
import math

def inner_product(a, b):
	ans = 0.0
	for i in range(4):
		ans += a[i] * b[i]

	return ans

v = [0] * 4
v[0] = [4.0, 1.0, 3.0, -1.0]
v[1] = [2.0, 1.0, -3.0, 4.0]
v[2] = [1.0, 0.0, -2.0, 7.0]
v[3] = [6.0, 2.0, 9.0, -5.0]

for i in range(4):
	for j in range(i):
		size = inner_product(v[j], v[j])
		inner = inner_product(v[i], v[j])

		for k in range(4):
			v[i][k] -= v[j][k] * inner / size

print(v[3][1])
```

answer
```
0.91611
```

</br></br></br>

# Lattices - What's a Lattice?

세 백터로 이루어진 단위 직육면체? 의 부피를 구하는 문제이다. 

3 * 3 행렬에서 determinant를 구하면 된다는 건 널리 알려져 있다. 

ex.py
```python
from sage.all import *

v = matrix(3, [6, 2, -3, 5, 1, 4, 2, 7, 1])

print(abs(v.det()))
```

answer
```
255
```

</br></br></br>

# Lattices - Gaussian Reduction

가우스 소거법을 배운다. 

최대공약수 구하는거랑 비슷한 느낌이었다. 

ex.py
```python
def inner_product(a, b):
	ans = 0
	for i in range(2):
		ans += a[i] * b[i]

	return ans

def gauss(a, b):
	if inner_product(a, a) > inner_product(b, b):
		return gauss(b, a)

	m = inner_product(a, b) // inner_product(a, a)

	if m == 0:
		return a, b

	for i in range(2):
		b[i] -= m * a[i]

	return gauss(a, b)

v = [846835985, 9834798552]
u = [87502093, 123094980]

(u, v) = gauss(u, v)
print(inner_product(u, v))
```

answer
```
7410790865146821
```

</br></br></br>

# Lattices - Find the Lattice

f, g가 비트수가 q, h 등에 비해서 매우 작아야 하는 것이 핵심이었다.

256비트 이내여야 한 조건을 생각하면 가우스 소거법을 사용해야 함을 알 수가 있다. 

ex.py
```python
from Crypto.Util.number import getPrime, inverse, long_to_bytes

q, h = 7638232120454925879231554234011842347641017888219021175304217358715878636183252433454896490677496516149889316745664606749499241420160898019203925115292257, 2163268902194560093843693572170199707501787797497998463462129592239973581462651622978282637513865274199374452805292639586264791317439029535926401109074800
e = 5605696495253720664142881956908624307570671858477482119657436163663663844731169035682344974286379049123733356009125671924280312532755241162267269123486523

def inner_product(a, b):
	ans = 0
	for i in range(2):
		ans += a[i] * b[i]

	return ans

def gauss(a, b):
	if inner_product(a, a) > inner_product(b, b):
		return gauss(b, a)

	m = inner_product(a, b) // inner_product(a, a)

	if m == 0:
		return [a, b]

	for i in range(2):
		b[i] -= m * a[i]

	return gauss(a, b)

def decrypt(q, h, f, g, e):
    a = (f*e) % q
    m = (a*inverse(f, g)) % g
    return m

v = [0, q]
u = [1, h]

ans = gauss(u, v)

f = ans[0][0]
g = ans[0][1]

m = decrypt(q, h, f, g, e)

print(long_to_bytes(m).decode())
```

answer
```
crypto{Gauss_lattice_attack!}
```

</br></br></br>

# Lattices - Backpack Cryptography

LLL을 사용하면 쉽게 풀릴 것으로 예상되지만, 실제로 일반 LLL을 사용하면 문제에 마주하게 된다. 

만들어진 public_key들의 값이 굉장히 커 보이지만, 만들어진 과정을 보면 작은 값에서 시작했기 때문에 적절히 합해서 0이 되는 벡터들이 존재하는 것을 확인할 수 있다.

그래서 1/2을 넣어주는 트릭을 쓸 수 있다. 모든 계수의 값이 0, 1임을 이용해 올바른 값의 경우 1/2, -1/2으로만 이루어진 벡터가 생성되고, 위에서 말한 다른 합해서 0이 되는 벡터들의 크기들은 1.5, 2.5, 3.5 이런식으로 돼서 크기가 훨씬 커진다. 물론 더 작은 경우도 있어서 이 방법이 안 통할 수도 있지만, 생성된 272개의 벡터들을 봐보니 다행히 정답이 나온다. 

ex.sage
```python
from Crypto.Util.number import *

pubkey = 생략
enc = 45690752833299626276860565848930183308016946786375859806294346622745082512511847698896914843023558560509878243217521

l = len(pubkey)

matrix = []

for i in range(l):
	line = [0] * (l + 1)
	line[l] = pubkey[i]
	line[i] = 1
	matrix.append(line)

line = [1 / 2] * l + [enc]
matrix.append(line)

M = Matrix(matrix)
M = M.LLL()
for line in M:
	if line[l] == 0:
		fail = 0
		for i in range(l):
			if line[i] != 1 / 2 and line[i] != -1 / 2:
				fail = 1
				break

		if fail == 0:
			val = 0
			for i in range(l):
				if line[i] == -1 / 2:
					val |= (1 << i)

			print(long_to_bytes(val))
```

flag
```
crypto{my_kn4ps4ck_1s_l1ghtw31ght}
```

---
</br></br></br>

# Brainteasers part 1 - Successive powers

588, 665, 216, 113, 642, 4, 836, 114, 851, 492, 819, 237
이 연속적인 거듭제곱이라고 한다.

그럼 무엇을 생각할 수 있느냐, 665 * 665 와 588 * 216은 p를 법으로 같아야 함을 알 수 있다. 

```
665 * 665 - 588 * 216 = 7 ^ 3 * 919
```

p = 919임을 알 수 있다. 
x를 구하는 것 또한 간단하다. 

ex.py
```python
print(pow(588, -1, 919) * 665 % 919)
```

answer
```
crypto{919,209}
```

</br></br></br>

# Brainteasers part 1 - Adrien's Signs

legendre symbol만 쓰면 풀린다. 

ex.py
```python
from Crypto.Util.number import long_to_bytes

output = 생략

a = 288260533169915
p = 1007621497415251

ans = 0

for i in range(224):
	if pow(output[i], (p - 1) // 2, p) == 1:
		ans += 1 << (224 - i - 1)

print(long_to_bytes(ans).decode())
```

flag
```
crypto{p4tterns_1n_re5idu3s}
```

</br></br></br>

# Brainteasers part 1 - Modular Binomials

머리를 좀만 굴리고 행렬을 좀만 쓰면 p, q를 구할 수 있다. 

e1, e2가 서로소인게 핵심 포인트이다. 

ex.py
```python
N, e1, e2, c1, c2 생략


# a * e1 - b * e2 = 1
a = pow(e1, -1, e2)
b = (a * e1 - 1) // e2

eq1 = []
eq1.append(pow(2, a * e1, N))
eq1.append(pow(3, a * e1, N))
eq1.append(pow(c1, a, N))

eq2 = []
eq2.append(pow(5, b * e2, N))
eq2.append(pow(7, b * e2, N))
eq2.append(pow(c2, b, N))

f1 = []
f1.append(pow(eq2[0], -1, N) * eq1[0] % N)
f1.append(pow(eq2[1], -1, N) * eq1[1] % N)
f1.append(pow(eq2[2], -1, N) * eq1[2] % N)

a += e2
b += e1

eq1 = []
eq1.append(pow(2, a * e1, N))
eq1.append(pow(3, a * e1, N))
eq1.append(pow(c1, a, N))

eq2 = []
eq2.append(pow(5, b * e2, N))
eq2.append(pow(7, b * e2, N))
eq2.append(pow(c2, b, N))

f2 = []
f2.append(pow(eq2[0], -1, N) * eq1[0] % N)
f2.append(pow(eq2[1], -1, N) * eq1[1] % N)
f2.append(pow(eq2[2], -1, N) * eq1[2] % N)

m = Matrix(IntegerModRing(N), 2, [f1[0], f1[1], f2[0], f2[1]])
res = Matrix(IntegerModRing(N), 2, [f1[2], f2[2]])

pq = (m.inverse() * res).list()

assert pq[0] * pq[1] == N
print(f"crypto{{{pq[0]},{pq[1]}}}")
```

더 깔끔하게 풀 수 있을 것 같지만 귀찮다.

flag
```
crypto{112274000169258486390262064441991200608556376127408952701514962644340921899196091557519382763356534106376906489445103255177593594898966250176773605432765983897105047795619470659157057093771407309168345670541418772427807148039207489900810013783673957984006269120652134007689272484517805398390277308001719431273,132760587806365301971479157072031448380135765794466787456948786731168095877956875295282661565488242190731593282663694728914945967253173047324353981530949360031535707374701705328450856944598803228299967009004598984671293494375599408764139743217465012770376728876547958852025425539298410751132782632817947101601}
```

</br></br></br>

# Brainteasers part 1 - Broken RSA

하하하 익숙한 문제다.

cykor freshman ctf에서 본 친구이다.

ex.py
```python
from Crypto.Util.number import *

p = 생략
c = 생략

def modular_sqrt(a, p):
    if legendre_symbol(a, p) != 1:
        return 0
    elif a == 0:
        return 0
    elif p == 2:
        return 0
    elif p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    s = p - 1
    e = 0
    while s % 2 == 0:
        s //= 2
        e += 1

    n = 2
    while legendre_symbol(n, p) != -1:
        n += 1

    x = pow(a, (s + 1) // 2, p)
    b = pow(a, s, p)
    g = pow(n, s, p)
    r = e

    while True:
        t = b
        m = 0
        for m in range(r):
            if t == 1:
                break
            t = pow(t, 2, p)

        if m == 0:
            return x

        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m


def legendre_symbol(a, p):
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


for i in range(16):
    f = c
    chk = i
    for j in range(4):
        f = modular_sqrt(f, p)
        if f == 0:
            break
        if chk%2 == 1:
            f = p - f
        chk //= 2
    if f > 0:
        print(long_to_bytes(f))
```

말이 되는 스트링이 나올때까지 돌리면 된다. 

answer
```
b"Hey, if you are reading this maybe I didn't mess up my code too much. Phew. I really should play more CryptoHack before rushing to code stuff from scratch again. Here's the flag: crypto{m0dul4r_squ4r3_r00t}"
```

flag
```
crypto{m0dul4r_squ4r3_r00t}
```
</br></br></br>

# Brainteasers part 1 - No Way Back Home

푼 사람 수를 보니 꽤 어려운 문제인가 본데 나한테는 왠지 모르게 굉장히 쉬웠다.

Modular binomials가 훨씬 어려웠다 난.

```
v = vka * vkb * vkakb^-1
```
을 생각하고 inverse를 돌리면 inverse가 안 된다고 하는데 생각해보면 v가 p의 배수이다. 

근데 그래도 문제가 될 건 없다. 

v를 q로 나눈 나머지는 위 식과 같이 구하면 되고, v를 p로 나눈 나머지는 0이므로
중국인의 나머지 정리를 쓰면 끝난다. 

ex.py
```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from hashlib import sha256
from Crypto.Util.number import getPrime, GCD, bytes_to_long, long_to_bytes, inverse
from random import randint

p, q, vka, vkb, vkakb, c 생략

v_q = vka * vkb * pow(vkakb, -1, q) % q

# v = p * x
# p * x =v_q (mod q)

x = pow(p, -1, q) * v_q % q
v = x * p


key = sha256(long_to_bytes(v)).digest()
cipher = AES.new(key, AES.MODE_ECB)
m = unpad(cipher.decrypt(bytes.fromhex(c)), 16)

print(m.decode())

```

flag
```
crypto{1nv3rt1bl3_k3y_3xch4ng3_pr0t0c0l}
```

---
</br></br></br>

# Brainteasers part 2 - Ellipse Curve Cryptography

처음에는 타원곡선 문제인 줄 알고 시도도 하지 않았었고, 조금의 공부를 끝낸 뒤에 보았는데, 그냥 아예 다른 문제였다. 

하지만 타원곡선의 방식을 아니 더 이해가 쉽기는 했다. 

(x1, y1) + (x2, y2) = (x1 * x2 + d * y1 + y2, x1 * y2 + x2 * y1)으로 덧셈이 정의되어 있다. 

G를 (x0, y0)로 놓고서 계속 더하면서 규칙을 찾아 나가면 쉽게 규칙성이 보인다. 

G * n = P라고 하고, sqrt(d) = D라고 할 때,

Px + D\*Py = (Gx + D\*Gy) ^ n (mod p)이다. 

p가 크지 않기 때문에 discrete_log를 사용해서 풀 수 있다. 

ex.sage
```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Util.number import *
from hashlib import sha1
import random
from collections import namedtuple

Point = namedtuple("Point", "x y")

def point_addition(P, Q):
    Rx = (P.x*Q.x + D*P.y*Q.y) % p
    Ry = (P.x*Q.y + P.y*Q.x) % p
    return Point(Rx, Ry)


def scalar_multiplication(P, n):
    Q = Point(1, 0)
    while n > 0:
        if n % 2 == 1:
            Q = point_addition(Q, P)
        P = point_addition(P, P)
        n = n//2
    return Q


def gen_keypair():
    private = random.randint(1, p-1)
    public = scalar_multiplication(G, private)
    return (public, private)


def gen_shared_secret(P, d):
    return scalar_multiplication(P, d).x


def decrypt_flag(shared_secret: int, iv: bytes, ct: bytes):
    key = sha1(str(shared_secret).encode('ascii')).digest()[:16]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), 16)

p = 173754216895752892448109692432341061254596347285717132408796456167143559
D = 529
G = Point(29394812077144852405795385333766317269085018265469771684226884125940148, 94108086667844986046802106544375316173742538919949485639896613738390948)
A = Point(155781055760279718382374741001148850818103179141959728567110540865590463, 73794785561346677848810778233901832813072697504335306937799336126503714)
B = Point(171226959585314864221294077932510094779925634276949970785138593200069419, 54353971839516652938533335476115503436865545966356461292708042305317630)
data = {'iv': '64bc75c8b38017e1397c46f85d4e332b', 'encrypted_flag': '13e4d200708b786d8f7c3bd2dc5de0201f0d7879192e6603d7c5d6b963e1df2943e3ff75f7fda9c30a92171bbbc5acbf'}

sqrt_D = 23

F = IntegerModRing(p)
base = F(G.x + sqrt_D * G.y)
num = F(A.x + sqrt_D * A.y)
n = num.log(base)

assert scalar_multiplication(G, n) == A

shared_secret = gen_shared_secret(B, n)

print(decrypt_flag(shared_secret, bytes.fromhex(data["iv"]), bytes.fromhex(data["encrypted_flag"])))
```

flag
```
crypto{c0n1c_s3ct10n5_4r3_f1n1t3_gr0up5}
```

</br></br></br>

# Brainteasers part 2 - Roll your Own

이런 생각은 어떻게 하는지 모르겠다. 그냥 너무 유명한건가..?

나는 g = 2, n = 2 ^ q - 1로 하면 되겠다는 생각까지는 했다. 수학적으로는 가능하다, 왜 불가능한지는 말하지 않겠다.

</br>

g = q + 1, n = q ^ 2로 놓으면 풀린다. 

g ^ x == x * q + 1 (mod N)이 되어 x의 값도 알 수 있다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13403)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvuntil("Prime generated: \"0x")

q = bytes_to_long(bytes.fromhex(r.recvline().decode()[:-2]))

r.recvuntil("Send integers (g,n) such that pow(g,q,n) = 1: ")

to_send = {
	"g": hex(q + 1),
	"n": hex(q ** 2)
}

json_send(to_send)

r.recvuntil("Generated my public key: \"0x")

key = bytes_to_long(bytes.fromhex(r.recvline().decode()[:-2]))

r.recvuntil("What is my private key: ")

x = (key - 1) // q

to_send = {
	"x": hex(x)
}

json_send(to_send)

r.interactive()
```

flag
```
crypto{Grabbing_Flags_with_Pascal_Paillier}
```

</br></br></br>

# Brainteasers part 2 - Unencryptable
```
D ^ 65537 == D (mod N)
```

이라는 정보를 알려주었다. 

```
D ^ 65536 == 1 (mod N)
```
이라서 다음에는 무엇을 할까 생각하다가

D ^ 1 ~ D ^ 65536을 다 N을 법으로 계산해보기로 했다. 

중간중간에 1이 섞여 있는 신기한 결과가 나왔고

D ^ 512는 1, D ^ 256은 1이 아니라는 것을 알 수 있었다. 

여기서 D ^ 256에 주목하였다.

D ^ 256 = sq라고 하자.

</br>

N = p * q일때 sq * sq == 1 (mod p), sq * sq == 1 (mod q)인데, 

sq = 1 또는 -1이라는 값밖에 가질 수 없다. (p, q를 법으로)

둘 다 1이면 sq = 1, 둘 다 -1이면 sq = N - 1인데 둘 다 아닌 걸로 보아

하나는 1, 하나는 -1이라고 추측할 수 있다. 

</br>

여기서 sq - 1은 p 또는 q의 배수임을 알 수 있다. 

gcd(N, sq - 1)을 구하면 끝난다. 

ex.py
```python
from Crypto.Util.number import getPrime, inverse, bytes_to_long, long_to_bytes
import math

N = 생략
e = 0x10001
c = 생략
DATA = 생략

sqrt = pow(DATA, 256, N)

p = math.gcd(sqrt - 1, N)
q = N // p
phi = (p - 1) * (q - 1)

d = pow(e, -1, phi)
flag = long_to_bytes(pow(c, d, N))

print(flag)
```

flag
```
crypto{R3m3mb3r!_F1x3d_P0iNts_aR3_s3crE7s_t00}
```


</br></br></br>

# Brainteasers part 2 - Cofactor Cofantasy

처음에는 어렵게 생각했는데 quadratic residue를 생각하면 간단하다. 

일단 factorDB로 N을 소인수분해 하면 몇 개의 소수들의 곱이 나온다. 

비트가 1이라면 1/2의 확률로 거듭제곱하는 수가 짝수일 텐데, 이 경우는 모든 소수들에 대한 quadratic residue가 1이어야 함을 알 수 있다. 

운이 나쁘게 홀수가 나올 수 있으니 10번 시험을 해서 1023/1024의 정확도를 얻게 설정한다.

근데 비트 수가 344개인거를 고려하면 20번정도는 시험해야 더 안전할 것 같긴 하다.

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

import math
from factordb.factordb import FactorDB
from Crypto.Random.random import randint


r = remote('socket.cryptohack.org', 13398)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

N = 56135841374488684373258694423292882709478511628224823806418810596720294684253418942704418179091997825551647866062286502441190115027708222460662070779175994701788428003909010382045613207284532791741873673703066633119446610400693458529100429608337219231960657953091738271259191554117313396642763210860060639141073846574854063639566514714132858435468712515314075072939175199679898398182825994936320483610198366472677612791756619011108922142762239138617449089169337289850195216113264566855267751924532728815955224322883877527042705441652709430700299472818705784229370198468215837020914928178388248878021890768324401897370624585349884198333555859109919450686780542004499282760223378846810870449633398616669951505955844529109916358388422428604135236531474213891506793466625402941248015834590154103947822771207939622459156386080305634677080506350249632630514863938445888806223951124355094468682539815309458151531117637927820629042605402188751144912274644498695897277
phi = 56135841374488684373258694423292882709478511628224823806413974550086974518248002462797814062141189227167574137989180030483816863197632033192968896065500768938801786598807509315219962138010136188406833851300860971268861927441791178122071599752664078796430411769850033154303492519678490546174370674967628006608839214466433919286766123091889446305984360469651656535210598491300297553925477655348454404698555949086705347702081589881912691966015661120478477658546912972227759596328813124229023736041312940514530600515818452405627696302497023443025538858283667214796256764291946208723335591637425256171690058543567732003198060253836008672492455078544449442472712365127628629283773126365094146350156810594082935996208856669620333251443999075757034938614748482073575647862178964169142739719302502938881912008485968506720505975584527371889195388169228947911184166286132699532715673539451471005969465570624431658644322366653686517908000327238974943675848531974674382848
g = 986762276114520220801525811758560961667498483061127810099097

f = FactorDB(N)
f.connect()
primes = f.get_factor_list()
n = len(primes)

r.recvline()

def quad(a, p):
	return pow(a, (p - 1) // 2, p)

def check(x):
	for p in primes:
		if quad(x, p) != 1:
			return 0
	return 1

def check_idx(idx):
	for _ in range(10):
		json_send({
			"option": "get_bit",
			"i": str(idx)
			})
		if check(int(json_recv()["bit"], 16)) == 1:
			return 1
	return 0

FLAG = b"crypto{???????????????????????????????????}"
l = len(FLAG)

flag = ""

for i in range(l):
	c = 0
	for j in range(8):
		c |= check_idx(8 * i + j) << j
	flag += chr(c)
	print(flag)

r.interactive()
```

flag
```
crypto{0ver3ng1neering_ch4lleng3_s0lution$}
```

</br></br></br>

# Brainteasers part 2 - Real Eisenstein

LLL 알고리즘에 대해서 공부하면 답을 얻을 수 있다. 
sage는 보면 볼수록 레전드다.

ex.py
```python
import math
from decimal import *

getcontext().prec = 100

ciphertext = 1350995397927355657956786955603012410260017344805998076702828160316695004588429433

FLAG = "crypto{???????????????}"
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]

l = len(FLAG)

basis = []

for i in range(l + 1):
	line = [0] * (l + 1)
	line[i] = 1
	basis.append(line)

for i in range(l):
	basis[i][l] = math.floor(Decimal(PRIMES[i]).sqrt() * 16**64)

basis[l][l] = -ciphertext

M = Matrix(basis)
M = M.LLL()
flag = M[0].list()

flag_s = ""
for i in range(l):
	flag_s += chr(flag[i])
print(flag_s)

```

flag
```
crypto{r34l_t0_23D_m4p}
```

---
</br></br></br>

# Primes - Prime and Prejudice

Miller - Rabin 소수 판별법으로 소수 여부를 결정한다. 

저 판별법을 뚫는 pseudo-prime을 찾는 것이 이 문제의 목적이다. 

조금만 검색을 해 보면 방법을 찾을 수 있다. 

https://github.com/loluwot/StrongPseudoPrimeGeneratorMkII
https://eprint.iacr.org/2018/749.pdf

방법 자체가 어려운 메서드는 아니었지만, 확실히 논문 없이는 쉽지 않을 것 같다. 

flag
```
crypto{Forging_Primes_with_Francois_Arnault}
```

---
</br></br></br>