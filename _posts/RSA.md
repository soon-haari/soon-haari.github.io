# Starter - RSA Starter 1

```python
print(pow(101, 17, 22663))
```
```
19906
```

</br></br></br>

# Starter - RSA Starter 2

```python
print(pow(12, 65537, 17 * 23))
```
```
301
```

</br></br></br>

# Starter - RSA Starter 3

```python
p = 857504083339712752489993810777
q = 1029224947942998075080348647219
print((p -1) * (q - 1))
```
```
882564595536224140639625987657529300394956519977044270821168
```

</br></br></br>

# Starter - RSA Starter 4

```python
p = 857504083339712752489993810777
q = 1029224947942998075080348647219

e = 65537

phi = (p - 1) * (q - 1)
N = p * q

d = pow(e, -1, phi)

print(d)
```
```
121832886702415731577073962957377780195510499965398469843281
```

</br></br></br>

# Starter - RSA Starter 5

```python
d = 121832886702415731577073962957377780195510499965398469843281
c = 77578995801157823671636298847186723593814843845525223303932
N = 882564595536224140639625987659416029426239230804614613279163

print(pow(c, d, N))
```
```
13371337
```
</br></br></br>

# Starter - RSA Starter 6

```python
from Crypto.Util.number import *
import hashlib

N = 15216583654836731327639981224133918855895948374072384050848479908982286890731769486609085918857664046075375253168955058743185664390273058074450390236774324903305663479046566232967297765731625328029814055635316002591227570271271445226094919864475407884459980489638001092788574811554149774028950310695112688723853763743238753349782508121985338746755237819373178699343135091783992299561827389745132880022259873387524273298850340648779897909381979714026837172003953221052431217940632552930880000919436507245150726543040714721553361063311954285289857582079880295199632757829525723874753306371990452491305564061051059885803
d = 11175901210643014262548222473449533091378848269490518850474399681690547281665059317155831692300453197335735728459259392366823302405685389586883670043744683993709123180805154631088513521456979317628012721881537154107239389466063136007337120599915456659758559300673444689263854921332185562706707573660658164991098457874495054854491474065039621922972671588299315846306069845169959451250821044417886630346229021305410340100401530146135418806544340908355106582089082980533651095594192031411679866134256418292249592135441145384466261279428795408721990564658703903787956958168449841491667690491585550160457893350536334242689

H = hashlib.sha256("crypto{Immut4ble_m3ssag1ng}".encode()).digest()

val = int.from_bytes(H, 'big')
S = pow(val, d, N)

print(hex(S))
```

```
6ac9bb8f110b318a40ad8d7e57defdcce2652f5928b5f9b97c1504d7096d7af1d34e477b30f1a08014e8d525b14458b709a77a5fa67d4711bd19da1446f9fb0ffd9fdedc4101bdc9a4b26dd036f11d02f6b56f4926170c643f302d59c4fe8ea678b3ca91b4bb9b2024f2a839bec1514c0242b57e1f5e77999ee67c450982730252bc2c3c35acb4ac06a6ce8b9dbf84e29df0baa7369e0fd26f6dfcfb22a464e05c5b72baba8f78dc742e96542169710918ee2947749477869cb3567180ccbdfe6fdbe85bcaca4bf6da77c8f382bb4c8cd56dee43d1290ca856318c97f1756b789e3cac0c9738f5e9f797314d39a2ededb92583d97124ec6b313c4ea3464037d3
```
---
</br></br></br>

# Primes Part 1 - Factoring

신기하다.

ex.py
```python
from factordb.factordb import FactorDB

f = FactorDB(510143758735509025530880200653196460532653147)
f.connect()
print(min(f.get_factor_list()))
```

answer
```
19704762736204164635843
```

</br></br></br>

# Primes Part 1 - Inferius Prime

ex.py
```python
from factordb.factordb import FactorDB
from Crypto.Util.number import *

n = 742449129124467073921545687640895127535705902454369756401331
e = 3
ct = 39207274348578481322317340648475596807303160111338236677373

f = FactorDB(n)
f.connect()
f.get_factor_list()

phi = (f.get_factor_list()[0] - 1) * (f.get_factor_list()[1] - 1)
d = pow(e, -1, phi)
m = pow(ct, d, n)

flag = long_to_bytes(m)

print(flag)
```

flag
```
crypto{N33d_b1g_pR1m35}
```

</br></br></br>

# Primes Part 1 - Monoprime

n이 소수이다. 

```python
from Crypto.Util.number import *

n = 171731371218065444125482536302245915415603318380280392385291836472299752747934607246477508507827284075763910264995326010251268493630501989810855418416643352631102434317900028697993224868629935657273062472544675693365930943308086634291936846505861203914449338007760990051788980485462592823446469606824421932591                                                                  
e = 65537
ct = 161367550346730604451454756189028938964941280347662098798775466019463375610700074840105776873791605070092554650190486030367121011578171525759600774739890458414593857709994072516290998135846956596662071379067305011746842247628316996977338024343628757374524136260758515864509435302781735938531030576289086798942  

d = pow(e, -1, n - 1)
m = pow(ct, d, n)
print(long_to_bytes(m))
```

flag
```
crypto{0n3_pr1m3_41n7_pr1m3_l0l}
```

</br></br></br>

# Primes Part 1 - Square Eyes

n이 제곱수다(p = q이다).

루트로 p를 구하고 phi = p * (p - 1)이게 설정하면 된다.

ex.py
```python
from Crypto.Util.number import *
import math

N = 생략
e = 65537
c = 생략

p = math.isqrt(N)

phi = p * (p - 1)
d = pow(e, -1, phi)
m = pow(c, d, N)

print(long_to_bytes(m))
```

flag
```
crypto{squar3_r00t_i5_f4st3r_th4n_f4ct0r1ng!}
```

</br></br></br>

# Primes Part 1 - Manyprime

ex.py
```python
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math

n = 생략
e = 65537
ct = 생략

f = FactorDB(n)
f.connect()
plist = f.get_factor_list()

phi = 1
for p in plist:
	phi *= p - 1

d = pow(e, -1, phi)
m = pow(ct, d, n)

print(long_to_bytes(m))
```

flag
```
crypto{700_m4ny_5m4ll_f4c70r5}
```
---
</br></br></br>

# Public Exponent - Salty

e가 1이다. ㅋㅋㅋㅋ

ex.py
```python
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math

n = 110581795715958566206600392161360212579669637391437097703685154237017351570464767725324182051199901920318211290404777259728923614917211291562555864753005179326101890427669819834642007924406862482343614488768256951616086287044725034412802176312273081322195866046098595306261781788276570920467840172004530873767                                                                  
e = 1
ct = 44981230718212183604274785925793145442655465025264554046028251311164494127485

print(long_to_bytes(ct))
```

flag
```
crypto{saltstack_fell_for_this!}
```

</br></br></br>

# Public Exponent - Modulus Inutilis

신기한 cbrt라는 함수와 sympy 모듈에 대해 알게 되었다. 

ct가 n보다 굉장히 작기 때문에 mod n을 무시해도 된다. 

ex.py
```python
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import sympy

n = 17258212916191948536348548470938004244269544560039009244721959293554822498047075403658429865201816363311805874117705688359853941515579440852166618074161313773416434156467811969628473425365608002907061241714688204565170146117869742910273064909154666642642308154422770994836108669814632309362483307560217924183202838588431342622551598499747369771295105890359290073146330677383341121242366368309126850094371525078749496850520075015636716490087482193603562501577348571256210991732071282478547626856068209192987351212490642903450263288650415552403935705444809043563866466823492258216747445926536608548665086042098252335883
e = 3
ct = 243251053617903760309941844835411292373350655973075480264001352919865180151222189820473358411037759381328642957324889519192337152355302808400638052620580409813222660643570085177957

print(long_to_bytes(sympy.cbrt(ct)))
```

flag
```
crypto{N33d_m04R_p4dd1ng}
```
</br></br></br>

# Public Exponent - Everything is Big

Wiener Attack이라고 해서 d가 N에 비해 굉장히 작을 때 d를 찾을 수 있는 방법이 존재한다.

ex.py
```python
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import owiener

N = 생략
e = 생략
c = 생략

d = owiener.attack(e, N)
m = pow(c, d, N)

print(long_to_bytes(m))
```

flag
```
crypto{s0m3th1ng5_c4n_b3_t00_b1g}
```

</br></br></br>

# Public Exponent - Crossed Wires

n과 phi(n)이 크기가 거의 비슷(나누었을 때 1과 유사)함을 이용할 수 있다. 

먼저 e, d를 둘 다 알고 있으므로 phi는 e * d - 1의 약수임을 알 수 있는데, 그 값을 n으로 나누어서 phi가 몇 번 더해졌는지 알 수 있다. 그렇게 phi를 구하면 문제가 끝난다. 

ex.py
```python
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import owiener

prikey = 생략
pubkeys = 생략
flag_enc = 생략

n = prikey[0]
phi = prikey[1] * 0x10001 - 1

mul = round(phi / n)

assert phi % mul == 0
phi //= mul

e = 1
for i in range(5):
	e *= pubkeys[i][1]

d = pow(e, -1, phi)
m = pow(flag_enc, d, n)

print(long_to_bytes(m))
```

flag
```
crypto{3ncrypt_y0ur_s3cr3t_w1th_y0ur_fr1end5_publ1c_k3y}
```

</br></br></br>

# Public Exponent - Everything is Still Big

https://github.com/mimoo/RSA-and-LLL-attacks/blob/master/boneh_durfee.sage

어렵다.....

flag
```
crypto{bon3h5_4tt4ck_i5_sr0ng3r_th4n_w13n3r5}
```

</br></br></br>

# Public Exponent - Endless Emails

7개의 메시지가 다 같은 줄 알고 꽤나 고생했다. 

3개의 정보만 있어도 이론상 M을 구할 수 있다. 

7개중 3개의 쌍을 다 돌려야 한다.

아마 7개 중 5개 정도가 같은 메시지인 것 같다. 

CRT를 기반으로 한 Hastad의 공격을 이용하면 풀 수 있다.

ex.py
```python
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import sympy

n = [0] * 7
c = [0] * 7
n, c의 입력은 생략한다.

def find_invpow(x,n):
    high = 1
    while high ** n <= x:
        high *= 2
    low = high//2
    while low < high:
        mid = (low + high) // 2
        if low < mid and mid**n < x:
            low = mid
        elif high > mid and mid**n > x:
            high = mid
        else:
            return mid
    return mid + 1



l = [0] * 3
for i in range(7 ** 3):
	l[0] = i % 7
	l[1] = (i // 7) % 7
	l[2] = i // 49

	if l[0] == l[1] or l[0] == l[2] or l[1] == l[2]:
		continue

	C = 0
	N = 1
	for j in l:
		t = (c[j] - C) * pow(N, -1, n[j])
		C += t * N
		N *= n[j]
		C %= N

	m = find_invpow(C, 3)
	if m ** 3 == C:
		print(long_to_bytes(m).decode())
		break

```

flag
```
yes

---

Johan Hastad
Professor in Computer Science in the Theoretical Computer Science
Group at the School of Computer Science and Communication at KTH Royal Institute of Technology in Stockholm, Sweden.

crypto{1f_y0u_d0nt_p4d_y0u_4r3_Vuln3rabl3}
```

---
</br></br></br>

# Primes Part 2 - Infinite Descent

만들어져 있는 함수를 돌려보면 앞 2/3가량의 비트가 p, q가 항상 같음을 알 수 있다. 

p, q = m + a, m - a로 놓을 수 있는데

m ^ 2가 가장 n과 가까운 값을 m으로 놓을 수 있음을 알 수 있다. 

m을 알아내면 p + q가 2m이라서 phi도 알 수 있다. 

ex.py
```python
import random
from Crypto.Util.number import *

n = 생략
e = 65537
c = 생략

def find_invpow(x, n):
    """Finds the integer component of the n'th root of x,
    an integer such that y ** n <= x < (y + 1) ** n.
    """
    high = 1
    while high ** n <= x:
        high *= 2
    low = high // 2
    while low < high:
        mid = (low + high) // 2
        if low < mid and mid**n < x:
            low = mid
        elif high > mid and mid**n > x:
            high = mid
        else:
            return mid
    return mid + 1

def getPrimes(bitsize):
    r = random.getrandbits(bitsize)
    p, q = r, r
    while not isPrime(p):
        p += random.getrandbits(bitsize//4)
    while not isPrime(q):
        q += random.getrandbits(bitsize//8)
    return p, q

sq = find_invpow(n, 2)

'''
l = []
for i in range(4):
	k = sq - 1 + i
	diff = abs(n - k ** 2)
	l.append(diff.bit_length())

print(l)

# [2050, 2049, 1036, 2049]
sq + 1 is the value we want
'''

plus = 2 * (sq + 1)
phi = n - plus + 1
d = pow(e, -1, phi)
m = pow(c, d, n)

print(long_to_bytes(m))
```

flag
```
crypto{f3rm47_w45_4_g3n1u5}
```
</br></br></br>

# Primes Part 2 - Marin's Secrets

2 ^ p - 1꼴의 소수들을 사용했다. 

생각해보니 그냥 작은 값부터 나눠가면서 나누어떨어지는지 검사해도 됐을 것 같다. 

하여튼 나는 비트별로 1이 나타나는 순간을 찾아 p를 구했다. 

ex.py
```python
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import sympy

n = 생략
e = 65537
c = 생략

N = n - 1
smaller = 0

while(1):
	if N % 2 == 0:
		N //= 2
		smaller += 1
	else:
		break

p = (1 << smaller) - 1
assert n % p == 0
q = n // p

assert isPrime(p)
assert isPrime(q)

phi = p * q - p - q + 1
d = pow(e, -1, phi)
m = pow(c, d, n)

print(long_to_bytes(m))

```

flag
```
crypto{Th3se_Pr1m3s_4r3_t00_r4r3}
```

</br></br></br>

# Primes Part 2 - Fast Primes

FactorDB에서 바로 소인수분해를 해준다. 

대부분의 유명한 취약점들을 대입해 보나보다.

ROCA라는 공격 방법을 이용해서 푸는 것이라고 한다.

공부를 더 해야겠다. 

flag
```
crypto{p00R_3570n14}
```

</br></br></br>

# Primes Part 2 - Ron was Wrong, Whit is Right

힌트로 서로 다른 n의 최대공약수가 존재할 경우 p, q를 뚫을 수 있다는 정보를 알면 쉬운 문제이다.

ex.py
```python
from Crypto.PublicKey import RSA
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import sympy
from Crypto.Cipher import PKCS1_OAEP

rsa_key = RSA.importKey(open('keys_and_messages/1.pem', "rb").read())
n = rsa_key.n
e = rsa_key.e

n = []
e = []
c = []

for i in range(50):
	rsa_key = RSA.importKey(open(f'keys_and_messages/{i + 1}.pem', "rb").read())
	n.append(rsa_key.n)
	e.append(rsa_key.e)
	ciphertext = open(f'keys_and_messages/{i + 1}.ciphertext', "r").read()
	c.append(int(ciphertext, 16))


for i in range(50):
	for j in range(50):
		if i == j:
			continue

		if math.gcd(n[i], n[j]) > 1:
			p = math.gcd(n[i], n[j])
			q = n[i] // p

			assert p * q == n[i]

			phi = n[i] - p - q + 1
			d = pow(e[i], -1, phi)
			
			key = RSA.construct((n[i], e[i], d))

			cipher = PKCS1_OAEP.new(key)
			plaintext = cipher.decrypt(long_to_bytes(c[i]))
			print(plaintext)
			exit()

```

flag
```
crypto{3ucl1d_w0uld_b3_pr0ud} If you haven't already, check out https://eprint.iacr.org/2012/064.pdf
```
</br></br></br>

# Primes Part 2 - RSA Backdoor Viability

FactorDB로 뚫리긴 하는데, 풀이를 읽어보니깐 eliptic curve를 배워야 하는 것 같다. 

나중에 공부해서 다시 적도록 하겠다. 

---
</br></br></br>

# Padding - Bespoke Padding

다른 좋은 풀이들도 있는 것 같은데, 나는 행렬연산을 사용했다. 

e가 11밖에 안 되기 때문에 11 * 11 행렬에서 역행렬을 구하는 방법을 사용해서 

m, m ^ 2, m ^ 3 ..... m ^ 11의 행렬을 구할 수 있다. 

sage를 못 써서 구현해서 썼다.

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13386)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

def C(n, r):
	res = 1
	for i in range(n):
		res *= i + 1
	for i in range(r):
		res //= i + 1
	for i in range(n - r):
		res //= i + 1
	
	return res

def inverse(mat, n):
	inv = [[0] * e for i in range(e)]
	for i in range(e):
		inv[i][i] = 1

	for i in range(e):
		j = i
		while mat[j][i] == 0:
			j += 1

		tmp = mat[i]
		mat[i] = mat[j]
		mat[j] = tmp
		tmp = inv[i]
		inv[i] = inv[j]
		inv[j] = tmp

		mul = pow(mat[i][i], -1, n)
		for j in range(e):
			mat[i][j] *= mul
			mat[i][j] %= n
			inv[i][j] *= mul
			inv[i][j] %= n

		for j in range(e):
			if j == i:
				continue
			mul = mat[j][i]
			for k in range(e):
				mat[j][k] -= mul * mat[i][k]
				mat[j][k] %= n
				inv[j][k] -= mul * inv[i][k]
				inv[j][k] %= n
	return inv

r.recvline()

json_send({"option": "get_flag"})
res = json_recv()
n = res["modulus"]
e = 11


mat = []
enc = []

for _ in range(e):
	json_send({"option": "get_flag"})
	res = json_recv()
	v = res["encrypted_flag"]
	a = res["padding"][0]
	b = res["padding"][1]

	line = []
	for i in range(11):
		exp = i + 1
		line.append(pow(a, exp, n) * pow(b, e - exp, n) * C(e, exp) % n)
	enc.append((v - pow(b, e, n)) % n)
	mat.append(line)

inv = inverse(mat, n)

m = 0

for i in range(e):
	m += enc[i] * inv[0][i]
	m %= n

print(long_to_bytes(m))

r.interactive()
```

flag
```
crypto{linear_padding_isnt_padding}
```

</br></br></br>

# Padding - Null or Never

뒤에 \x00을 붙인다는 것은 256을 곱해준다는 것이다. 

즉 flag_pad = flag * 256 ^ 57임을 알 수 있다. 

flag ^ 3 = flag_pad ^ 3 * 256 ^ (-57 * 3) = c * 256 ^ (-57 * 3) 이다. 

그런데 flag ^ 3의 비트는 약 1028비트, n의 비트는 1024비트라서 

c * 256 ^ (-57 * 3)를 구한 후 n을 계속 더해주면서 세제곱근이 되는지를 검사해야 한다. 

4비트 차이밖에 안 나서 거의 얼마 안 걸린다. 

ex.py
```python
from Crypto.PublicKey import RSA
from Crypto.Util.number import *
from factordb.factordb import FactorDB
import math
import sympy
from Crypto.Cipher import PKCS1_OAEP

def find_invpow(x,n):
    """Finds the integer component of the n'th root of x,
    an integer such that y ** n <= x < (y + 1) ** n.
    """
    high = 1
    while high ** n <= x:
        high *= 2
    low = high // 2
    while low < high:
        mid = (low + high) // 2
        if low < mid and mid**n < x:
            low = mid
        elif high > mid and mid**n > x:
            high = mid
        else:
            return mid
    return mid + 1

n = 95341235345618011251857577682324351171197688101180707030749869409235726634345899397258784261937590128088284421816891826202978052640992678267974129629670862991769812330793126662251062120518795878693122854189330426777286315442926939843468730196970939951374889986320771714519309125434348512571864406646232154103
e = 3
c = 63476139027102349822147098087901756023488558030079225358836870725611623045683759473454129221778690683914555720975250395929721681009556415292257804239149809875424000027362678341633901036035522299395660255954384685936351041718040558055860508481512479599089561391846007771856837130233678763953257086620228436828

# flag is 43 characters long

# pad_flag = flag * 256 ** 57

c = c * pow(256, -57 * 3, n) % n

#FLAG = b"crypto{???????????????????????????????????}"
# f = bytes_to_long(FLAG)

# print((f ** 3).bit_length())
# 1028
# print(n.bit_length())
# 1024

while 1:
	m = find_invpow(c, e)
	if m ** 3 == c:
		print(long_to_bytes(m))
		break
		
	c += n
```

flag
```
crypto{n0n_574nd4rd_p4d_c0n51d3r3d_h4rmful}
```

---
</br></br></br>

# Signatures Part 1 - Signing Server

암호화된 플래그를 받아서 복호화해주기만 하면 된다.

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13374, level = 'debug')

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

json_send({"option": "get_secret"})

json_send({"option":"sign", "msg": json_recv()["secret"][2:]})

m = json_recv()["signature"][2:]

print(m)

print(bytes.fromhex(m))

r.interactive()
```

flag
```
TODO: audit signing server to make sure that meddling hacker doesn't get hold of my secret flag: crypto{d0n7_516n_ju57_4ny7h1n6}
```

</br></br></br>

# Signatures Part 1 - Let's Decrypt

이상한 re 스트링이라는게 등장해서 조금 애먹었지만, 수학 방면에서는 너무 간단한 문제였다. 

ex.py
```python
from pkcs1 import emsa_pkcs1_v15
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13391, level = "debug")

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

json_send({"option": "get_signature"})
signature = json_recv()["signature"]

msg = 'I am Mallory.own CryptoHack.org'
digest = bytes_to_long(emsa_pkcs1_v15.encode(msg.encode(), 256))

m = bytes_to_long(msg.encode())
s = int(signature, 16)

n = s - digest

json_send({"option": "verify", "N": hex(n), "e": hex(1), "msg": msg})

r.interactive()
```

flag
```
Congratulations, here's a secret: crypto{dupl1c4t3_s1gn4tur3_k3y_s3l3ct10n}
```

</br></br></br>

# Signatures Part 1 - Blinding Light

token * 2를 보내고 2를 보내서 나누어서 토큰을 뚫는다.

이 방법을 예전에 본 적이 있어서 쉽게 생각해낸 것 같은데 어딘지 기억이 안 난다. 

ex.py
```python
from pkcs1 import emsa_pkcs1_v15
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13376, level = "debug")

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

json_send({"option": "get_pubkey"})
res = json_recv()

n = int(res["N"], 16)
e = int(res["e"], 16)

ADMIN_TOKEN = bytes_to_long(b"admin=True")

json_send({"option": "sign", "msg": hex(ADMIN_TOKEN * 2)[2:]})
res = json_recv()
sig_2token = int(res["signature"], 16)

json_send({"option": "sign", "msg": "02"})
res = json_recv()
sig_2 = int(res["signature"], 16)

sig_token = sig_2token * pow(sig_2, -1, n) % n

json_send({"option": "verify", "signature": hex(sig_token), "msg": hex(ADMIN_TOKEN)[2:]})

r.interactive()
```

flag
```
crypto{m4ll34b1l17y_c4n_b3_d4n63r0u5}
```

---
</br></br></br>

# Signatures Part 2 - Vote for Pedro

Alice n은 다 함정이다. 

스트링의 마지막 15바이트가 같게 하는게 목표이므로 mod는 2 ^ 120이 된다. 

m ^ 3 % (2 ^ 120)이 보트포페드로가 되게 하면 되는데 처음에는 어렵게 생각했다가

n = 2 ^ 120인 RSA와 동일하다는 걸 깨달았다. 

ex.py
```python
from pkcs1 import emsa_pkcs1_v15
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13375, level = "debug")

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

pedro = b'\x00VOTE FOR PEDRO'

pedro_long = bytes_to_long(pedro)

n = 2 ** 120
phi = 2 ** 119
e = 3
d = pow(e, -1, phi)
m = pow(pedro_long, d, n)

assert m ** 3 % n == pedro_long

json_send({"option": "vote", "vote": hex(m)})

r.interactive()
```

flag
```
crypto{y0ur_v0t3_i5_my_v0t3}
```

</br></br></br>

# Signatures Part 2 - Let's Decrypt Again

코드 분석에 한참 걸렸지만 문제 자체는 수학적으로 굉장히 어렵지는 않았다. 

discrete_log를 사용하면 문제에서 요구한 조건들을 모두 뚫을 수 있다. 

n은 768비트보다 커야 하면서 discrete_log를 뚫어야 하므로 prime ** k의 꼴로 나타내면 된다. 

대충 n이 primt ** k의 꼴일 때 discrete_log를 구하는 과정이 구상이 간다. 나중에 실제로 구현해봐야겠다. 

그런데 sage에서 pwntools가 작동을 안 한다. magic number이 일치하지 않는다며 기분나쁜 에러를 내뿜는다. 이 오류도 해결하고싶다. 

ex.py
```python
from pkcs1 import emsa_pkcs1_v15
from pwn import *
from Crypto.Util.number import *
import json
import codecs
from sympy.ntheory import discrete_log

r = remote('socket.cryptohack.org', 13394)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

def xor(a, b):
    assert len(a) == len(b)
    return bytes(x ^ y for x, y in zip(a, b))

strs = ["This is a test for a fake signature.", "My name is Jerry and I own CryptoHack.org", "Please send all my money to 1"]

alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
res = 256 ** 19
raw = b"\x00" + long_to_bytes(res)
raw += hashlib.sha256(hashlib.sha256(raw).digest()).digest()[:4]
res = bytes_to_long(raw)

addr = ""
while res > 0:
    addr = alpha[res % 58] + addr
    res //= 58
strs[2] += addr


n = getPrime(20) ** 41



def solve(msg, idx):
    digest = bytes_to_long(emsa_pkcs1_v15.encode((msg + suffix).encode(), 768 // 8))
    e = discrete_log(n, digest, s)
    json_send({'option':'claim', 'msg': msg + suffix, 'e': hex(e), 'index': idx})
    return bytes.fromhex(json_recv()['secret'])

r.recvline()

json_send({"option": "get_signature"})
res = json_recv()
s = int(res["signature"], 16)

json_send({'option': 'set_pubkey', 'pubkey': hex(n)})
suffix = json_recv()['suffix']

secret0 = solve(strs[0], 0)
secret1 = solve(strs[1], 1)
secret2 = solve(strs[2], 2)

print(xor(secret0, xor(secret1, secret2)))


```

flag
```
crypto{let's_decrypt_w4s_t0o_ez_do_1t_ag41n}
```

---
</br></br></br>