# Background - Background Reading

```
crypto{abelian}
```
---
</br></br></br>

# Starter - Point Negation

P = (8045,6936) 이기 때문에 Q = (8045,-6936)

즉 p = 9739를 법으로 (8045, 2803)이다.

```
crypto{8045,2803}
```

</br></br></br>

# Starter - Point Addition

덧셈의 과정을 이해하고 직접 구현해본다. 

ex.py
```python
from Crypto.Util.number import *

a = 497
b = 1768
p = 9739

def check(P):
	if P == None:
		return True

	return (P[0] ** 3 + a * P[0] + b) % p == (P[1] ** 2) % p

def addition(P1, P2):
	if P1 == None:
		return P2
	if P2 == None:
		return P1

	x1, y1 = P1[0], P1[1]
	x2, y2 = P2[0], P2[1]

	if x1 == x2 and y1 != y2:
		return None

	slope = 0
	if x1 != x2:
		slope = ((y1 - y2) * pow(x1 - x2, -1, p)) % p

	else:
		slope = ((3 * x1 ** 2 + a) * pow(2 * y1, -1, p)) % p

	x3 = (slope ** 2 - x1 - x2) % p
	y3 = (slope * (x1 - x3) - y1) % p

	R_prime = (x3, y3)
	assert check(R_prime)
	return R_prime

P = (493, 5564)
Q = (1539, 4742)
R = (4403, 5202)

assert check(P)
assert check(Q)
assert check(R)


addition(P, addition(P, addition(Q, R)))
print(addition(P, addition(P, addition(Q, R))))
```

flag
```
crypto{4215, 2162}
```
</br></br></br>

# Starter - Scalar Multiplication

정수 배 곱셈을 O(logn)의 시작복잡도로 정의한다. 

ex.py
```python
from Crypto.Util.number import *

a = 497
b = 1768
p = 9739

def check(P):
	if P == None:
		return True

	return (P[0] ** 3 + a * P[0] + b) % p == (P[1] ** 2) % p

def addition(P1, P2):
	if P1 == None:
		return P2
	if P2 == None:
		return P1

	x1, y1 = P1[0], P1[1]
	x2, y2 = P2[0], P2[1]

	if x1 == x2 and y1 != y2:
		return None

	slope = 0
	if x1 != x2:
		slope = ((y1 - y2) * pow(x1 - x2, -1, p)) % p

	else:
		slope = ((3 * x1 ** 2 + a) * pow(2 * y1, -1, p)) % p

	x3 = (slope ** 2 - x1 - x2) % p
	y3 = (slope * (x1 - x3) - y1) % p

	R_prime = (x3, y3)
	assert check(R_prime)
	return R_prime


def multiplication(P, n):
	if n == 0:
		return None

	Q = multiplication(P, n // 2)
	P_neven = addition(Q, Q)

	if n % 2 == 1:
		return addition(P_neven, P)

	return P_neven

P = (2339, 2213)

assert check(P)

print(multiplication(P, 7863))
```

flag
```
crypto{9467, 2742}
```

</br></br></br>

# Starter - Curves and Logs

Diffie-Hellman과 비슷한 방식으로 작동한다. 

n_B * Q_A가 shared secret이다. 

ex.py
```python
from Crypto.Util.number import *
import hashlib 

a = 497
b = 1768
p = 9739

def check(P):
	if P == None:
		return True

	return (P[0] ** 3 + a * P[0] + b) % p == (P[1] ** 2) % p

def addition(P1, P2):
	if P1 == None:
		return P2
	if P2 == None:
		return P1

	x1, y1 = P1[0], P1[1]
	x2, y2 = P2[0], P2[1]

	if x1 == x2 and y1 != y2:
		return None

	slope = 0
	if x1 != x2:
		slope = ((y1 - y2) * pow(x1 - x2, -1, p)) % p

	else:
		slope = ((3 * x1 ** 2 + a) * pow(2 * y1, -1, p)) % p

	x3 = (slope ** 2 - x1 - x2) % p
	y3 = (slope * (x1 - x3) - y1) % p

	R_prime = (x3, y3)
	assert check(R_prime)
	return R_prime


def multiplication(P, n):
	if n == 0:
		return None

	Q = multiplication(P, n // 2)
	P_neven = addition(Q, Q)

	if n % 2 == 1:
		return addition(P_neven, P)

	return P_neven

qa = (815, 3190)

assert check(qa)

x = multiplication(qa, 1829)[0]

sha1 = hashlib.sha1()
sha1.update(str(x).encode())

print(f"crypto{{{sha1.hexdigest()}}}")
```

flag
```
crypto{80e5212754a824d3a4aed185ace4f9cac0f908bf}
```

</br></br></br>

# Starter - Efficient Exchange

x만 알면 y는 2개로 결정되기 때문에 사실 크게 중요하지 않다는 것을 말하고자 한다. 

p == 3 (mod 4)이기 때문에 a ^ ((p + 1) / 4)로 간단히 루트를 구할 수 있다. 

ex.py
```python
from Crypto.Util.number import *
import hashlib 

a = 497
b = 1768
p = 9739

def check(P):
	if P == None:
		return True

	return (P[0] ** 3 + a * P[0] + b) % p == (P[1] ** 2) % p

def addition(P1, P2):
	if P1 == None:
		return P2
	if P2 == None:
		return P1

	x1, y1 = P1[0], P1[1]
	x2, y2 = P2[0], P2[1]

	if x1 == x2 and y1 != y2:
		return None

	slope = 0
	if x1 != x2:
		slope = ((y1 - y2) * pow(x1 - x2, -1, p)) % p

	else:
		slope = ((3 * x1 ** 2 + a) * pow(2 * y1, -1, p)) % p

	x3 = (slope ** 2 - x1 - x2) % p
	y3 = (slope * (x1 - x3) - y1) % p

	R_prime = (x3, y3)
	assert check(R_prime)
	return R_prime


def multiplication(P, n):
	if n == 0:
		return None

	Q = multiplication(P, n // 2)
	P_neven = addition(Q, Q)

	if n % 2 == 1:
		return addition(P_neven, P)

	return P_neven

q_x = 4726
q_y = pow(q_x ** 3 + a * q_x + b, (p + 1) // 4, p)
q = (q_x, q_y)

assert check(q)

x = multiplication(q, 6534)[0]

print(x)
```

flag
```
crypto{3ff1c1ent_k3y_3xch4ng3}
```

---
</br></br></br>

# Parameter Choice - Smooth Criminal

sagemath의 discrete_log를 사용하면 n을 구해준다. 

이게 되는 이유가 E의 order이 소수가 아니라 소인수분해가 굉장히 잘 돼서라고 한다. 

ex.sage
```python
from Crypto.Cipher import AES
from Crypto.Util.number import inverse
from Crypto.Util.Padding import pad, unpad
from collections import namedtuple
from random import randint
import hashlib
import os

def decrypt_flag(shared_secret: int, iv, ciphertext):
    # Derive AES key from shared secret
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), 16)


p = 310717010502520989590157367261876774703
a = 2
b = 3

F = GF(p)
E = EllipticCurve(F,[a,b])

g_x = 179210853392303317793440285562762725654
g_y = 105268671499942631758568591033409611165
G = E(g_x, g_y)

p_x = 280810182131414898730378982766101210916
p_y = 291506490768054478159835604632710368904
P = E(p_x, p_y)

b_x = 272640099140026426377756188075937988094
b_y = 51062462309521034358726608268084433317
B = E(b_x, b_y)

n = discrete_log(P, G, operation="+")

shared_secret = (B * n).xy()[0]

print(decrypt_flag(shared_secret, bytes.fromhex('07e2628b590095a5e332d397b8a59aa7'), bytes.fromhex('8220b7c47b36777a737f5ef9caa2814cf20c1c1ef496ec21a9b4833da24a008d0870d3ac3a6ad80065c138a2ed6136af')))
```

flag
```
crypto{n07_4ll_curv3s_4r3_s4f3_curv3s}
```

</br></br></br>

# Parameter Choice - Exceptional Curves

타원곡선에서 가장 기본적이고 유명한 취약점이 p와 E의 order이 같은 경우인 것 같다. 

Dreamhack의 not so smart 문제에서도 이를 다루었다. 

Smart attack이라는 것을 사용하면 해결할 수 있다. 

ex.sage
```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from random import randint
import hashlib

p = 0xa15c4fb663a578d8b2496d3151a946119ee42695e18e13e90600192b1d0abdbb6f787f90c8d102ff88e284dd4526f5f6b6c980bf88f1d0490714b67e8a2a2b77
a = 0x5e009506fcc7eff573bc960d88638fe25e76a9b6c7caeea072a27dcd1fa46abb15b7b6210cf90caba982893ee2779669bac06e267013486b22ff3e24abae2d42
b = 0x2ce7d1ca4493b0977f088f6d30d9241f8048fdea112cc385b793bce953998caae680864a7d3aa437ea3ffd1441ca3fb352b0b710bb3f053e980e503be9a7fece

E = EllipticCurve(GF(p), [a, b])

assert E.order() == p

def _lift(E, P, gf):
    x, y = map(ZZ, P.xy())
    for point_ in E.lift_x(x, all=True):
        _, y_ = map(gf, point_.xy())
        if y == y_:
            return point_


def attack(G, P):
    E = G.curve()
    gf = E.base_ring()
    p = gf.order()
    assert E.trace_of_frobenius() == 1, f"Curve should have trace of Frobenius = 1."

    E = EllipticCurve(Qp(p), [int(a) + p * ZZ.random_element(1, p) for a in E.a_invariants()])
    G = p * _lift(E, G, gf)
    P = p * _lift(E, P, gf)
    Gx, Gy = G.xy()
    Px, Py = P.xy()
    return int(gf((Px / Py) / (Gx / Gy)))

G_x = 3034712809375537908102988750113382444008758539448972750581525810900634243392172703684905257490982543775233630011707375189041302436945106395617312498769005
G_y = 4986645098582616415690074082237817624424333339074969364527548107042876175480894132576399611027847402879885574130125050842710052291870268101817275410204850
G = E(G_x, G_y)

pub_x = 4748198372895404866752111766626421927481971519483471383813044005699388317650395315193922226704604937454742608233124831870493636003725200307683939875286865
pub_y = 2421873309002279841021791369884483308051497215798017509805302041102468310636822060707350789776065212606890489706597369526562336256272258544226688832663757
pub = E(pub_x, pub_y)

b_x = 0x7f0489e4efe6905f039476db54f9b6eac654c780342169155344abc5ac90167adc6b8dabacec643cbe420abffe9760cbc3e8a2b508d24779461c19b20e242a38
b_y = 0xdd04134e747354e5b9618d8cb3f60e03a74a709d4956641b234daa8a65d43df34e18d00a59c070801178d198e8905ef670118c15b0906d3a00a662d3a2736bf
B = E(b_x, b_y)

shared_secret = (attack(G, pub) * B).xy()[0]
sha1 = hashlib.sha1()
sha1.update(str(shared_secret).encode('ascii'))
key = sha1.digest()[:16]
iv = bytes.fromhex('719700b2470525781cc844db1febd994')
enc = bytes.fromhex('335470f413c225b705db2e930b9d460d3947b3836059fb890b044e46cbb343f0')

cipher = AES.new(key, AES.MODE_CBC, iv)
print(unpad(cipher.decrypt(enc), 16))
```

flag
```
crypto{H3ns3l_lift3d_my_fl4g!}
```

</br></br></br>

# Parameter Choice - Micro Transmissions

이 문제 또한 제목과 코드를 잘 봤어야 했다. 

n의 비트는 64, p의 비트는 256으로 n의 비트가 상대적으로 굉장히 작다. 

Pohlig-Hellman Attack을 사용할 때, 가장 큰 소수의 경우는 discrete_log를 구하는데 굉장히 오래 걸리는데, n의 범위가 그 소수보다 작아서 나머지만으로 CRT를 진행해도 올바른 n의 값을 구할 수 있다. 

ex.sage
```python
from Crypto.Util.number import *
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

def gen_shared_secret(P, n):
    S = P*n
    return S.xy()[0]

def decrypt_flag(shared_secret: int, iv, ct):
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), 16)

p = 99061670249353652702595159229088680425828208953931838069069584252923270946291
a = 1
b = 4
E = EllipticCurve(GF(p), [a,b])

G = E(43190960452218023575787899214023014938926631792651638044680168600989609069200, 20971936269255296908588589778128791635639992476076894152303569022736123671173)
ax = 87360200456784002948566700858113190957688355783112995047798140117594305287669
bx = 6082896373499126624029343293750138460137531774473450341235217699497602895121
A = E.lift_x(ax)
B = E.lift_x(bx)
data = {'iv': 'ceb34a8c174d77136455971f08641cc5', 'encrypted_flag': 'b503bf04df71cfbd3f464aec2083e9b79c825803a4d4a43697889ad29eb75453'}

n = G.order()

fac = list(factor(n))

print(fac)

moduli = []
remainder = []

for i, j in fac:
    if i == 210071842937040101:
        break

    mod = i**j
    _g_ = G * ZZ(n / mod)
    _q_ = A * ZZ(n / mod)

    dl = discrete_log(_q_, _g_, operation = "+")
    moduli.append(mod)
    remainder.append(dl)
    print(dl, mod)

#Alice's secret integer
nA = crt(remainder,moduli)

assert nA * G == A

shared_secret = gen_shared_secret(B, nA)

print(decrypt_flag(shared_secret, bytes.fromhex(data["iv"]), bytes.fromhex(data["encrypted_flag"])))
```

flag
```
crypto{d0nt_l3t_n_b3_t00_sm4ll}
```

</br></br></br>

# Parameter Choice - Elliptic Nodes

알고 있는 점의 정보가 2개 있기 때문에 a, b의 값을 알아낼 수 있다. 

처음부터 강조한 4a^3 + 27b^2가 0이 아닌 중요성을 다루는 문제이다. 

https://l0z1k.com/singular_curves 에서 singular curve을 뚫는 과정을 잘 설명해주셨다. 

ex.sage
```python
from Crypto.Util.number import *

p = 4368590184733545720227961182704359358435747188309319510520316493183539079703

gx = 8742397231329873984594235438374590234800923467289367269837473862487362482
gy = 225987949353410341392975247044711665782695329311463646299187580326445253608

qx = 2582928974243465355371953056699793745022552378548418288211138499777818633265
qy = 2421683573446497972507172385881793260176370025964652384676141384239699096612

a = (((gy**2 - gx**3) - (qy**2 - qx**3)) * pow(gx - qx, -1, p)) % p
b = ((gy**2 - gx**3) - a * gx) % p

assert (gy**2 - (gx**3 + a * gx + b)) % p == 0
assert (qy**2 - (qx**3 + a * qx + b)) % p == 0

assert (4 * a**3 + 27 * b**2) % p == 0 # singular

P.<x,y> = PolynomialRing(GF(p))
f = y^2 - (x^3 + a*x + b)
g = x^3 + a*x + b

diff = 2810666857764293539402767964015657133595357252060455687347132657823581321982

f = f(x - diff, y)

t = GF(p)(-4063410388559334897980342709342612042350324567872047551521081480287204886243).square_root()

gx += diff
qx += diff

g = (gy + t * gx)/(gy - t * gx) % p
q = (qy + t * qx)/(qy - t * qx) % p

flag = int(q.log(g))

print(long_to_bytes(flag))
```

flag
```
crypto{s1ngul4r_s1mplif1c4t1on}
```

</br></br></br>

# Parameter Choice - Moving Problems

이름에서 알 수 있듯이, 그리고 description에서도 힌트가 있는데, MOV attack을 사용할 수 있는 조건을 충족시킨다. 

https://github.com/jvdsn/crypto-attacks

굉장히 쓸만한 툴이 많은 깃헙이다.

ex.sage
```python
from Crypto.Util.number import *
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

from attacks.ecc.mov_attack import attack

def gen_shared_secret(P, n):
    S = P*n
    return S.xy()[0]

def decrypt_flag(shared_secret: int, iv, ct):
    sha1 = hashlib.sha1()
    sha1.update(str(shared_secret).encode('ascii'))
    key = sha1.digest()[:16]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), 16)

p = 1331169830894825846283645180581
a = -35
b = 98
E = EllipticCurve(GF(p), [a,b])

G = E(479691812266187139164535778017, 568535594075310466177352868412)
A = E(1110072782478160369250829345256, 800079550745409318906383650948)
B = E(1290982289093010194550717223760, 762857612860564354370535420319)
data = {'iv': 'eac58c26203c04f68d63dc2c58d79aca', 'encrypted_flag': 'bb9ecbd3662d0671fd222ccb07e27b5500f304e3621a6f8e9c815bc8e4e6ee6ebc718ce9ca115cb4e41acb90dbcabb0d'}

n = attack(G, A)

shared_secret = gen_shared_secret(B, n)

print(decrypt_flag(shared_secret, bytes.fromhex(data["iv"]), bytes.fromhex(data["encrypted_flag"])))
```

flag
```
crypto{MOV_attack_on_non_supersingular_curves}
```

아는 공격 +1!

</br></br></br>

# Parameter Choice - Real Curve Crypto

머리아픈 문제다..........

풀이 구걸을 얼마나 해서 풀었는지 모르겠다.

결과적으로는 weierstrass-p 함수라는 것이 존재해 복소수 범위에서의 타원곡선은 2차원 격자로 이동시키는 것이 가능하다. 

글고 sagemath에서 찾은 굉장한 함수들 덕에 풀 수 있었다. 나의 경우는 e_log_RC()

https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/period_lattice.html
이 링크는 두고두고 봐야될 것 같다는 생각이 든다. 

쨌든 a * flag = b (mod e) 와 같은 식(a, b, e는 실수, 사실은 복소수긴 하다)을 통해서

LLL을 머리굴려서 잘 쓰면 flag를 구할 수 있다. 

ex.sage
```python
from Crypto.Cipher import AES
from Crypto.Util.number import *
from Crypto.Util.Padding import pad, unpad

from mpmath import mp, mpf
from os import urandom

import json
import random

px_test = 5.7936422841648631603801488271326840133366257361747844958466724707375149663891529401615095812310596840098915446734828057824002036270361374672987740490726230497045237157628507834062758774401974035616144
gx =      1.1593952488083258955953169788699597120285034179687500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
px =      1052.1869486109503324827555468817188804055933729601321435932864694301534931492427433020783168479195188024409373571681097603398390379320742186401833284576176214641603772370675124838606986281131453644941

E = EllipticCurve('32a2')
L = E.period_lattice()
G = E.lift_x(gx)
mod = L.basis(prec=1000)[0]

key = 119530811605533856381092150719308346379

P_test = E.lift_x(px_test)
P = E.lift_x(px)

def log_point(pt):
    xP, yP = pt.xy()
    return L.e_log_RC(xP, yP, prec=1000)


#assert abs(mul(key, log_point(G), mod) - log_point(P_test)) < 10^(-150)

base = (log_point(G) / mod).real()
goal = (log_point(P) / mod).real()
goal_test = (log_point(P_test) / mod).real()

to_int = 10^100

def mul(val, m):
    res = (val * m) % 1
    while res < 0:
        res += 1
    return res

print(mul(base, key) - goal_test)

goal_test = round(goal_test * to_int)
goal = round(goal * to_int)



bases = []
for i in range(8):
    bases.append(round(mul(base, 65536^i) * to_int))

base = base * 10^150

mat = Matrix(ZZ, 10, 11)



for i in range(8):
    mat[i, i] = 1
    mat[i, 10] = bases[i]

mat[8, 8] = 1
mat[8, 10] = to_int
mat[9, 9] = 1
mat[9, 10] = goal

mat = mat.LLL()

print(mat[0])



key_ = mat[0]

key = 0

for i in range(8):
    key += 65536^i * (-key_[i])

key //= -2

print(key)

key = long_to_bytes(key)



ciphertext = bytes.fromhex("a104b68d30a207eabf293324fbde64f8d628fb07068058c1e76e670e7e805fc567f739185bbe6cbb44f09013173ee653")
iv = bytes.fromhex("485f9a1e4a3b19348367280df13f9e77")

cipher = AES.new(key, AES.MODE_CBC, iv)
pt = cipher.decrypt(ciphertext)

print(pt)


```

flag
```
crypto{real_fields_arent_finite}
```

---
</br></br></br>

# Signatures - Digestive

ECDSA는 hash function을 거친 뒤에, 앞의 n비트(n비트는 타원곡선의 order의 비트수와 동일하다)만을 따오기 때문에, hash function이 아무 조작도 거치지 않는 이 digestive 함수에서는 n비트 이후의 값을 아무것으로나 조작해도 상관없다는 의미이다. 

즉 긴 이름으로 signature을 요청한 후, 뒤에 "admin"을 true로 변조시키는 값을 딕셔너리에 넣어주면 된다. 

```
{"msg":"{"admin": false, "username": "soonhaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaari"}","signature":"ac9281edc29c22994f9b7b33a640c7beeca2bfdea71ab686c2ab57bad55367dfb775a66d5fc418bce0c195cf48ba617d"}
```

```
{"admin": false, "username": "soonhaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaari", “admin": true}
```

flag
```
crypto{thanx_for_ctf_inspiration_https://mastodon.social/@filippo/109360453402691894}
```

</br></br></br>

# Signatures - Curveball

P256의 order을 구한 다음에(홀수이다.) 2의 order에 대한 inverse를 계산한 값을 bing의 G값에 곱하면, d를 2로 설정했을 때 bing의 G가 나오게 되어 문제가 풀린다. 100점이나 할 문제는 아닌 것 같다. 

ex.sage
```python
from pwn import *
import json

r = remote("socket.cryptohack.org", 13382)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

p = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
a = 0xffffffff00000001000000000000000000000000fffffffffffffffffffffffc
b = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
G = (0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296, 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5)

bing = (0x3B827FF5E8EA151E6E51F8D0ABF08D90F571914A595891F9998A5BD49DFA3531, 0xAB61705C502CA0F7AA127DEC096B2BBDC9BD3B4281808B3740C320810888592A)
E = EllipticCurve(GF(p), [a, b])

G = E(G[0], G[1])
bing = E(bing[0], bing[1])

order = E.order()

send = bing * ((order + 1) // 2)
assert 2 * send == bing
send = send.xy()
# print(send)


r.recvline()
r.sendline(f"{{\"private_key\": {2}, \"host\": \"eh\", \"curve\": \"eh\", \"generator\": {[send[0], send[1]]}}}")

r.interactive()
```

flag
```
crypto{Curveballing_Microsoft_CVE-2020-0601}
```

</br></br></br>

# Signatures - ProSign 3

코드를 잘 봐보면 n의 값이 G의 order이 아닌 시간을 나타낼 때 쓰는 값으로 바뀌는 것을 볼 수 있다. 즉 범위가 커봤자 60이고, k의 값을 알아낼 수 있다.

ex.sage
```python
import hashlib
from Crypto.Util.number import *
from ecdsa.ecdsa import Public_key, Private_key, Signature, generator_192
import json
from pwn import *

r = remote("socket.cryptohack.org", 13381)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

def sha1(data):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(data)
    return sha1_hash.digest()

def sha1(data):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(data.encode())
    
    return bytes_to_long(sha1_hash.digest())

p = 6277101735386680763835789423207666416083908700390324961279
a = 6277101735386680763835789423207666416083908700390324961276
b = 2455155546008943817740293915197451784769108058161191238065
G_x = generator_192.x()
G_y = generator_192.y()

E = EllipticCurve(GF(p), [a, b])
G = E(G_x, G_y)

n = int(generator_192.order())
assert n == G.order()

Ln = n.bit_length()

r.recvline()

json_send({
    "option": "sign_time"
    })
data = json_recv()

msg = data["msg"]
sig_r = int(data["r"], 16)
sig_s = int(data["s"], 16)

k = 0

for i in range(1, 60):
    if int((G * i).xy()[0]) == sig_r:
        k = i
        break

assert int((G * k).xy()[0]) == sig_r

z = sha1(msg)

dA = ((k * sig_s - z) * pow(sig_r, -1, n)) % n


msg = "unlock"
send_r = int(G.xy()[0]) % n
send_s = (sha1(msg) + send_r * dA) % n

json_send({
    "option": "verify",
    "msg": msg,
    "r": hex(send_r),
    "s": hex(send_s)
    })


r.interactive()
```

flag
```
crypto{ECDSA_700_345y_70_5cr3wup}
```

</br></br></br>

# Signatures - No Random, No Bias

뚫어지게 쳐다보다 z값과 k값이 모두 sha1의 결과값인 160비트로 n에 비해서 굉장히 작다는 것을 알았다. 이름에 있는 bias만 검색을 잘해봐도 바로 쓸모있는 논문이 나온다. 

https://eprint.iacr.org/2019/023.pdf

LLL을 이용해서 k값들을 구할 수 있다. 

ex.sage
```python
from hashlib import sha1
from Crypto.Util.number import bytes_to_long, long_to_bytes
from ecdsa import ellipticcurve
from ecdsa.ecdsa import curve_256, generator_256, Public_key, Private_key
from random import randint

hidden_flag = (16807196250009982482930925323199249441776811719221084165690521045921016398804, 72892323560996016030675756815328265928288098939353836408589138718802282948311)

pubkey = (48780765048182146279105449292746800142985733726316629478905429239240156048277, 74172919609718191102228451394074168154654001177799772446328904575002795731796)

datas = []

datas.append({'msg': 'I have hidden the secret flag as a point of an elliptic curve using my private key.', 'r': '0x91f66ac7557233b41b3044ab9daf0ad891a8ffcaf99820c3cd8a44fc709ed3ae', 's': '0x1dd0a378454692eb4ad68c86732404af3e73c6bf23a8ecc5449500fcab05208d'})
datas.append({'msg': 'The discrete logarithm problem is very hard to solve, so it will remain a secret forever.', 'r': '0xe8875e56b79956d446d24f06604b7705905edac466d5469f815547dea7a3171c', 's': '0x582ecf967e0e3acf5e3853dbe65a84ba59c3ec8a43951bcff08c64cb614023f8'})
datas.append({'msg': 'Good luck!', 'r': '0x566ce1db407edae4f32a20defc381f7efb63f712493c3106cf8e85f464351ca6', 's': '0x9e4304a36d2c83ef94e19a60fb98f659fa874bfb999712ceb58382e2ccda26ba'})

n = Integer(generator_256.order())

r = []
s = []
z = []
t = []
a = []

for data in datas:
	sig_r = Integer(int(data["r"], 16))
	sig_s = Integer(int(data["s"], 16))
	sig_z = Integer(bytes_to_long(sha1(data["msg"].encode()).digest()))

	t_i = (sig_r / sig_s) % n
	a_i = (-sig_z / sig_s) % n
	r.append(sig_r)
	s.append(sig_s)
	z.append(sig_z)
	t.append(t_i)
	a.append(a_i)

B = 2^165

M = []

for i in range(3):
	line = [0] * 5
	line[i] = n
	M.append(line)

line = []
line.extend(t)
line.append(B / n)
line.append(0)
M.append(line)

line = []
line.extend(a)
line.append(0)
line.append(B)
M.append(line)

M = Matrix(M).LLL()
res = M[1]
k0 = -res[0]
k1 = -res[1]
k2 = -res[2]

print(res)

sig_d = (k0 * s[0] - z[0]) / r[0] % n
print(sig_d)

sig_d = (k1 * s[1] - z[1]) / r[1] % n
print(sig_d)

sig_d = (k2 * s[2] - z[2]) / r[2] % n
print(sig_d)
# same

hidden_flag = ellipticcurve.Point(curve_256, hidden_flag[0], hidden_flag[1])

d_inv = pow(sig_d, -1, n)

flag = hidden_flag * d_inv

print(long_to_bytes(int(flag.x())))
```
flag
```
crypto{3mbrac3_r4nd0mn3ss}
```

---
</br></br></br>

# Edwards Curves - Edwards Goes Degenerate

타원곡선의 edward, montgomery, weierstrass form에 대해 공부했다. 

신기하게 다 우리가 가장 잘 알고 있는 weierstrass(y^2 = x^3 + ax + b)로 변환될 수 있다는 걸 알았다. 

그런데 그걸 이용해서 풀려니깐 정작 이 문제는 잘못 구현된 함수의 약점을 이용해서 푸는 거였다. 

먼저 x recovery 과정에서 똑바로 진행되지가 않아서 항상 x가 0으로 고정되는 걸 볼 수 있다. 

addition과정에 x에 0을 넣어서 어떤 동작을 하는지 보면 y값이 곱해지는 과정만이 진행된다. 즉 정수에서의 discrete_log 문제와 동일하다. 

ex.sage
```python
from Crypto.Util.number import inverse, bytes_to_long
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from random import randint
from hashlib import sha1
import os

class TwistedEdwards():
    # Elliptic curve in Edwards form:
    # -x**2 + y**2 = 1 + d*x**2*y**2
    # birationally equivalent to the Montgomery curve:
    # y**2 = x**3 + 2*(1-d)/(1+d)*x**2 + x

    def __init__(self, p, d, order, x0bit, y0):
        self.p = p
        self.d = d
        self.order = order
        self.base_point = (x0bit, y0)

    def recover_x(self, xbit, y):
        xsqr = (y**2 - 1)*inverse(1 + self.d*y**2, self.p) % self.p
        x = pow(xsqr, (self.p + 1)//4, self.p)
        if x**2 == xsqr :
            if x & 1 != xbit:
                return p - x
            return x
        return 0

    def decompress(self, compressed_point):
        xbit, y = compressed_point
        x = self.recover_x(xbit, y)
        return (x, y)

    # complete point addition formulas
    def add(self, P1, P2):
        x1, y1 = P1
        x2, y2 = P2
        
        C = x1*x2 % self.p
        D = y1*y2 % self.p
        E = self.d*C*D
        x3 = (1 - E)*((x1 + y1)*(x2 + y2) - C - D) % self.p
        y3 = (1 + E)*(D + C) % self.p
        z3 = 1 - E**2 % self.p
        z3inv = inverse(z3, self.p)
        return (x3*z3inv % self.p, y3*z3inv % self.p)

    # left-to-right double-and-add
    def single_mul(self, n, compressed_point):
        P = self.decompress(compressed_point)        
        t = n.bit_length()
        if n == 0:
            return (0,1)
        R = P
        for i in range(t-2,-1,-1):
            bit = (n >> i) & 1
            R = self.add(R, R)
            if bit == 1:
                R = self.add(R, P)
        return (R[0] & 1, R[1])


def gen_key_pair(curve):
    n = randint(1, curve.order-1)
    P = curve.single_mul(n, curve.base_point)
    return n, P
    
def gen_shared_secret(curve, n, P):
    xbit, y = curve.single_mul(n, P)
    return y
    

def decrypt_flag(shared_secret: int, iv, ct):
	# Derive AES key from shared secret
	key = sha1(str(shared_secret).encode('ascii')).digest()[:16]
	# Encrypt flag
	cipher = AES.new(key, AES.MODE_CBC, iv)
	pt = unpad(cipher.decrypt(ct), 16)

	return pt



# curve parameters
# birationally equivalent to the Montgomery curve y**2 = x**3 + 337*x**2 + x mod p
p = 110791754886372871786646216601736686131457908663834453133932404548926481065303
order = 27697938721593217946661554150434171532902064063497989437820057596877054011573
d = 14053231445764110580607042223819107680391416143200240368020924470807783733946
x0bit = 1
y0 = 11
curve = TwistedEdwards(p, d, order, x0bit, y0)

P_alice = (0, 109790246752332785586117900442206937983841168568097606235725839233151034058387)
P_bob = (0, 45290526009220141417047094490842138744068991614521518736097631206718264930032)

data = {'iv': '31068e75b880bece9686243fa4dc67d0', 'encrypted_flag': 'e2ef82f2cde7d44e9f9810b34acc885891dad8118c1d9a07801639be0629b186dc8a192529703b2c947c20c4fe5ff2c8'}

Zp = GF(p)

y_alice = Zp(P_alice[1])
y_bob = Zp(P_bob[1])
y_g = Zp(y0)

n_a = y_alice.log(y_g)

shared_secret = gen_shared_secret(curve, n_a, P_bob)

print(decrypt_flag(shared_secret, bytes.fromhex(data["iv"]), bytes.fromhex(data["encrypted_flag"])))
```

flag
```
crypto{degenerates_will_never_keep_a_secret}
```

---
</br></br></br>

# Side Channels - Montgomery's Ladder

montgomery form을 weierstrass form으로 변경해서 그냥 기존 타원곡선 곱셈을 하였다. 또 덧셈과 곱셈을 구현하기 귀찮았다...

ex.sage
```python
A = 486662
B = 1
p = 2^255 - 19

a = (3 - A^2) / (3 * B^2)
a %= p
b = (2 * A^3 - 9 * A) / (27 * B^3)
b %= p

P_x = 9 / B + A / (3 * B)
P_x %= p

E = EllipticCurve(GF(p), [a, b])

P = E.lift_x(P_x)

new_x = int((P * 0x1337c0decafe).xy()[0])

original_x = (new_x - A / (3 * B)) * B % p

print(f"crypto{{{original_x}}}")
```

flag
```
crypto{49231350462786016064336756977412654793383964726771892982507420921563002378152}
```

</br></br></br>

# Side Channels - Double and Broken

50개의 지지직거리는 데이터를 다 합해보면 플래그의 비트가 1일때는 큰 값(약 7천 얼마 ~ 8천 얼마), 비트가 0일때는 작은 값(약 4천 얼마 ~ 5천 얼마)의 값을 가진다. 

https://circuitcellar.com/research-design-hub/design-solutions/power-analysis-of-ecc-hardware-implementations/

ex.sage
```python
import json
from Crypto.Util.number import *

data = 생략

FLAG = b'crypto{?????????????????????????????????????}'
l = bytes_to_long(FLAG).bit_length()

som = []

for i in range(l):
	v = 0
	for j in range(50):
		v += data[j][i]
	som.append(v)

flag = 0

for i in range(l):
	if som[i] > 6700:
		flag += 1 << i

print(long_to_bytes(flag))
```

flag
```
crypto{Sid3_ch4nn3ls_c4n_br34k_s3cur3_curv3s}
```

---
</br></br></br>