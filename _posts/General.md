# Encoding - ASCII

아스키 코드의 개념을 이해한다. 

ex.py
```python
s = [99, 114, 121, 112, 116, 111, 123, 65, 83, 67, 73, 73, 95, 112, 114, 49, 110, 116, 52, 98, 108, 51, 125]

flag = ""

for i in range(len(s)):
    flag += chr(s[i])

print(flag)
```
flag

    crypto{ASCII_pr1nt4bl3}

</br></br></br>

# Encoding - Hex

hex 스트링을 자유자재로 바꿀 줄 안다.

ex.py
```python
val = "63727970746f7b596f755f77696c6c5f62655f776f726b696e675f776974685f6865785f737472696e67735f615f6c6f747d"

print(bytes.fromhex(val).decode())
```
flag

    crypto{You_will_be_working_with_hex_strings_a_lot}


</br></br></br>

# Encoding - Base64

base64의 개념을 이해한다..

그 정확한 원리에 대한 개념까지는 아직 요구하지 않는다. 

ex.py
```python
import base64

val = "72bca9b68fc16ac7beeb8f849dca1d8a783e8acf9679bf9269f7bf"

val = bytes.fromhex(val)

print(base64.b64encode(val).decode())
```
flag

    crypto/Base+64+Encoding+is+Web+Safe/

살면서 이런 플래그 형식은 처음이다.

</br></br></br>

# Encoding - Bytes and Big Integers

long_to_bytes라는 유용한 함수에 대한 소개다.

구현도 개간단하지만 이미 있는 함수를 잘 쓰면 당연히 편하다.

ex.py
```python
from Crypto.Util.number import *

val = 11515195063862318899931685488813747395775516287289682636499965282714637259206269

print(long_to_bytes(val).decode())
```
flag

    crypto{3nc0d1n6_4ll_7h3_w4y_d0wn}

</br></br></br>

# Encoding - Encoding Challange

구현 능력과 다양한 암호의 종류를 알아본다. 

Pwntools의 기초를 익히고, json 문법에 대해서도 공부해야 해결할 수 있었다. 

ex.py
```python
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13377, level = 'debug')

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

for _ in range(100):
    received = json_recv()

    decoded = ""
    encoding = received["type"]
    encoded = received["encoded"]
    print("encoding: " + str(encoding))
    print("encoded: " + str(encoded))

    if encoding == "base64":
        decoded = base64.b64decode(encoded.encode()).decode() # wow so encode
    elif encoding == "hex":
        decoded = bytes.fromhex(encoded).decode()
    elif encoding == "rot13":
        decoded = codecs.decode(encoded, 'rot_13')
    elif encoding == "bigint":
        decoded = long_to_bytes(int(encoded, 16)).decode()
    elif encoding == "utf-8":
        decoded = ""
        for i in range(len(encoded)):
            decoded += chr(encoded[i])

    print(decoded)
    to_send = {
        "decoded": decoded
    }
    json_send(to_send)

r.interactive()
```
flag

    crypto{3nc0d3_d3c0d3_3nc0d3}
---
</br></br></br>

# XOR - XOR Starter

xor의 개념의 시작

label을 13과 xor하면 aloha라니, 작지만 신기한 정보였다. 

ex.py
```python
label = "label"
xor_string = ""

for i in range(5):
	xor_string += chr(ord(label[i]) ^ 13)

print(f"crypto{{{xor_string}}}")
```

flag

    crypto{aloha}

</br></br></br>

# XOR - XOR Properties

```
KEY1 = a6c8b6733c9b22de7bc0253266a3867df55acde8635e19c73313
KEY2 ^ KEY1 = 37dcb292030faa90d07eec17e3b1c6d8daf94c35d4c9191a5e1e
KEY2 ^ KEY3 = c1545756687e7573db23aa1c3452a098b71a7fbf0fddddde5fc1
FLAG ^ KEY1 ^ KEY3 ^ KEY2 = 04ee9855208a2cd59091d04767ae47963170d1660df7f56f5faf
```

```
FLAG = (FLAG ^ KEY1 ^ KEY3 ^ KEY2) ^ (KEY2 ^ KEY3) ^ (KEY1)
```

ex.py
```python
from Crypto.Util.number import *

a = 0xa6c8b6733c9b22de7bc0253266a3867df55acde8635e19c73313
b = 0x37dcb292030faa90d07eec17e3b1c6d8daf94c35d4c9191a5e1e
c = 0xc1545756687e7573db23aa1c3452a098b71a7fbf0fddddde5fc1
d = 0x04ee9855208a2cd59091d04767ae47963170d1660df7f56f5faf

print(long_to_bytes(a ^ c ^ d).decode())
```

flag

    crypto{x0r_i5_ass0c1at1v3}

</br></br></br>

# XOR - Favourite byte
그 다음 문제에서 사용되는 풀이와 비슷하다는 생각이 들긴 한다.

첫 글자가 c임을 이용해서 xor되는 바이트를 알아낸다. 

ex.py
```python
from Crypto.Util.number import *

val = 0x73626960647f6b206821204f21254f7d694f7624662065622127234f726927756d

s = long_to_bytes(val)

x = s[0] ^ ord('c')

flag = ""
for i in range(len(s)):
	flag += chr(s[i] ^ x)

print(flag)
```

flag

    crypto{0x10_15_my_f4v0ur173_by7e}

</br></br></br>

# XOR - You either know, XOR you don't

플래그의 앞 글자가 crypto{ 인 것을 이용하는 것이라고 힌트에 적혀 있다. 

키의 앞 7글자를 알아낼 수 있다. 

```python
from Crypto.Util.number import *

val = 0x0e0b213f26041e480b26217f27342e175d0e070a3c5b103e2526217f27342e175d0e077e263451150104

enc = long_to_bytes(val)

xor = b"crypto{"

key = ""
for i in range(7):
	key += chr(enc[i] ^ xor[i])

print(key)
```

```
myXORke
```

myXORkey일 것이라고 rough하게 추측할 수 있다. 

ex.py
```python
from Crypto.Util.number import *

val = 0x0e0b213f26041e480b26217f27342e175d0e070a3c5b103e2526217f27342e175d0e077e263451150104

enc = long_to_bytes(val)

key = b"myXORkey"
dec = ""

for i in range(len(enc)):
	dec += chr(enc[i] ^ key[i % 8])

print(dec)
```

flag
```
crypto{1f_y0u_Kn0w_En0uGH_y0u_Kn0w_1t_4ll}
```

</br></br></br>

# XOR - Lemur XOR

두 이미지를 xor해야겠다는 아이디어는 생각하였다. 

그런데 PIL같은 파이썬 툴을 깔아봐도 대체 한 픽셀당 비교는 어떻게 해야 할 지 감이 안 잡혔다. 

xor 대신에 가능한 기능인 add나 subtract를 해도 결과가 나오기는 하였다.

![](./lemur.PNG)

flag
```
crypto{X0Rly_n0t!}
```
---
</br></br></br>

# Mathematics - GCD
최대공약수를 구하는 문제이다. 

간단히 구현해보자.

ex.py
```python
def gcd(a, b):
	if a == 0:
		return b
	if b == 0:
		return a

	if a < b:
		return gcd(a, b % a)

	return gcd(a % b, b)

print(gcd(66528, 52920))
```

answer
```
1512
```

</br></br></br>

# Mathematics - Extended GCD
```
Using the two primes p = 26513, q = 32321, find the integers u,v such that

p * u + q * v = gcd(p,q)
```
확장 유클리드를 구현하기 귀찮기에 pow 함수를 사용해서 풀도록 하겠다. 

ex.py
```python
p = 26513
q = 32321

# p * u + q * v = gcd(p,q) = 1

u = pow(p, -1, q)
v = (1 - p * u) // q

print(u)
print(v)
```

answer
```
-8404
```
</br></br></br>

# Mathematics - Modular 1
```
11 ≡ x mod 6
8146798528947 ≡ y mod 17

The solution is the smaller of the two integers.
```
간지나게 간단한 것도 파이썬으로 짜보자. 

ex.py
```python
x = 11 % 6
y = 8146798528947 % 17

if x < y:
	print(x)
else:
	print(y)
```

answer
```
4
```
</br></br></br>

# Mathematics - Modular 2

    p = 65537. Calculate 273246787654 ^ 65536 mod 65537.

p가 소수임을 알려줬다. 

페르마의 소정리를 생각하면 답이 나온다. 

answer
```
1
```

</br></br></br>

# Mathematics - Modular Inverting

pow 함수를 써야되나 했는데

그냥 인간 두뇌 선에서 문제가 풀린다. 

    3 * d ≡ 1 mod 13?

answer
```
9
```
---
</br></br></br>

# Data Formats - Privacy-Enhanced Mail?

pem 파일을 열고 d만 얻으면 되는데 그걸 구글링하느라 며칠이 걸렸다.
ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ

ex.py
```python
from Crypto.PublicKey import RSA

rsa_key = RSA.importKey(open('pem.pem', "rb").read())

print(rsa_key.d)
```

answer
```
15682700288056331364787171045819973654991149949197959929860861228180021707316851924456205543665565810892674190059831330231436970914474774562714945620519144389785158908994181951348846017432506464163564960993784254153395406799101314760033445065193429592512349952020982932218524462341002102063435489318813316464511621736943938440710470694912336237680219746204595128959161800595216366237538296447335375818871952520026993102148328897083547184286493241191505953601668858941129790966909236941127851370202421135897091086763569884760099112291072056970636380417349019579768748054760104838790424708988260443926906673795975104689
```

</br></br></br>

# Data Formats - CERTainly not

openssl을 통해서 der을 pem으로 만들고 위 문제와 같은 방법을 사용했다. 

```
openssl x509 -inform DER -outform PEM -text -in der.der -out pem.pem
```

ex.py
```python
from Crypto.PublicKey import RSA

rsa_key = RSA.importKey(open('pem.pem', "rb").read())

print(rsa_key.n)
```

answer
```
22825373692019530804306212864609512775374171823993708516509897631547513634635856375624003737068034549047677999310941837454378829351398302382629658264078775456838626207507725494030600516872852306191255492926495965536379271875310457319107936020730050476235278671528265817571433919561175665096171189758406136453987966255236963782666066962654678464950075923060327358691356632908606498231755963567382339010985222623205586923466405809217426670333410014429905146941652293366212903733630083016398810887356019977409467374742266276267137547021576874204809506045914964491063393800499167416471949021995447722415959979785959569497
```

</br></br></br>

# Data Formats - SSH Keys

ssh-keygen 명령어를 사용하면 된다. 

```
ssh-keygen -f pub.pub -e -m pem
-----BEGIN RSA PUBLIC KEY-----
MIIBigKCAYEArTy6m2vhhbwx3RVbNVb3ZOenCqqsOXHaJpbtN+OuulLKBSKpIoPB
+ZDbDXn0qWkf4lOxtGSgolkUbgG07Lhzfgs+dul4UL84CkwZExmF3Rf1nRv+v7pq
Lt2dPsCb02YLxJnhHJb4rQaz2ZM4QCtTOcqYDUeKfLHCaZU4Ekm/OApKrpfw4/0o
fn8KOrFN0t4/dqnNuwVRgoaUIhsI47reApB2rs0AP4CggSIi8s6BXCxB4YzgThBK
5760T1giACYQC5MFdq1Gw+INSFmu0CNqt5wdJ5Z4z5448Gke06R+IMtjUiGDQ3Qt
T2fK3gWhZxk14M4UNrdETgTW/mQ4B/BcvikxvoBGpKbttG0agfOjTen6wyzpGfcd
8N9rSbaqqyUwC8uDotzFtFzzutVAU9d91TagGzWBhNoMfplwVTns27GOOgv1dn5s
QSSSmP0hTbPMDlThysKkR9BiOVbBtWGQpV936pPBgyWERGqMqC9xykLdVHv2Vu05
T0WMwKCAetgtAgMBAAE=
-----END RSA PUBLIC KEY-----
```

ex.py
```python
from Crypto.PublicKey import RSA

rsa_key = RSA.importKey(open('pem.pem', "rb").read())

print(rsa_key.n)
```

answer
```
3931406272922523448436194599820093016241472658151801552845094518579507815990600459669259603645261532927611152984942840889898756532060894857045175300145765800633499005451738872081381267004069865557395638550041114206143085403607234109293286336393552756893984605214352988705258638979454736514997314223669075900783806715398880310695945945147755132919037973889075191785977797861557228678159538882153544717797100401096435062359474129755625453831882490603560134477043235433202708948615234536984715872113343812760102812323180391544496030163653046931414723851374554873036582282389904838597668286543337426581680817796038711228401443244655162199302352017964997866677317161014083116730535875521286631858102768961098851209400973899393964931605067856005410998631842673030901078008408649613538143799959803685041566964514489809211962984534322348394428010908984318940411698961150731204316670646676976361958828528229837610795843145048243492909
```

</br></br></br>

# Data Formats - Transparency

thetransparencyflagishere.cryptohack.org

```
crypto{thx_redpwn_for_inspiration}
```

---
</br></br></br>

