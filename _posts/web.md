# JSON Web Tokens - Token Appreciation

verify_signature False는 오류를 무시하고 진행시키는 역할인 듯 하다.

ex.py
```python
import jwt

encoded_jwt = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJmbGFnIjoiY3J5cHRve2p3dF9jb250ZW50c19jYW5fYmVfZWFzaWx5X3ZpZXdlZH0iLCJ1c2VyIjoiQ3J5cHRvIE1jSGFjayIsImV4cCI6MjAwNTAzMzQ5M30.shKSmZfgGVvd2OSB2CGezzJ3N6WAULo3w9zCl_T47KQ"

decoded_jwt = jwt.decode(encoded_jwt, options={"verify_signature": False})

print(decoded_jwt)
```

flag
```
crypto{jwt_contents_can_be_easily_viewed}
```

</br></br></br>

# JSON Web Tokens - JWT Sessions

authorization

</br></br></br>

# JSON Web Tokens - No Way JOSE

다음 코드로 none algorithm으로 encode해서 보내주면 된다. 

```python
import jwt
import base64
import json

print(jwt.encode({"admin": True}, None, algorithm="none"))
```

flag
```
crypto{The_Cryptographic_Doom_Principle}
```
</br></br></br>

# JSON Web Tokens - JWT Secrets

PyJWT 공식 readme.md에서 예시를 들면서 사용한 키는 "secret"이었다. 

문제에 키가 그것이라고 명시되어있었다.

```python
import jwt
import base64
import json

print(jwt.encode({'username': "jerry", 'admin': True}, "secret", algorithm='HS256'))
```

flag
```
crypto{jwt_secret_keys_must_be_protected}
```

</br></br></br>

# JSON Web Tokens - RSA or HMAC?

public key로도 검증이 된다. 그 정확한 원리는 사실 완벽히 이해가 안 됐다.

ex.py
```python
import jwt
import base64
import json
from Crypto.PublicKey import RSA

rsa_pub = "-----BEGIN RSA PUBLIC KEY-----\nMIIBCgKCAQEAvoOtsfF5Gtkr2Swy0xzuUp5J3w8bJY5oF7TgDrkAhg1sFUEaCMlR\nYltE8jobFTyPo5cciBHD7huZVHLtRqdhkmPD4FSlKaaX2DfzqyiZaPhZZT62w7Hi\ngJlwG7M0xTUljQ6WBiIFW9By3amqYxyR2rOq8Y68ewN000VSFXy7FZjQ/CDA3wSl\nQ4KI40YEHBNeCl6QWXWxBb8AvHo4lkJ5zZyNje+uxq8St1WlZ8/5v55eavshcfD1\n0NSHaYIIilh9yic/xK4t20qvyZKe6Gpdw6vTyefw4+Hhp1gROwOrIa0X0alVepg9\nJddv6V/d/qjDRzpJIop9DSB8qcF1X23pkQIDAQAB\n-----END RSA PUBLIC KEY-----\n"

print(jwt.encode({'username': "admin", 'admin': True}, rsa_pub, algorithm='HS256'))
```

문제에서 자꾸 패치를 하라는 부분은, 위 코드를 실행하면 키값이 너무 위험하다면서 에러를 내는데, 그 부분을 제외시켜주면 해결된다. 

util.py
```python
# Based on https://github.com/hynek/pem/blob/7ad94db26b0bc21d10953f5dbad3acfdfacf57aa/src/pem/_core.py#L224-L252
_PEMS = {
    b"CERTIFICATE",
    b"TRUSTED CERTIFICATE",
    b"PRIVATE KEY",
    b"PUBLIC KEY",
    b"ENCRYPTED PRIVATE KEY",
    b"OPENSSH PRIVATE KEY",
    b"DSA PRIVATE KEY",
    b"RSA PRIVATE KEY",
    #b"RSA PUBLIC KEY",
    b"EC PRIVATE KEY",
    b"DH PARAMETERS",
    b"NEW CERTIFICATE REQUEST",
    b"CERTIFICATE REQUEST",
    b"SSH2 PUBLIC KEY",
    b"SSH2 ENCRYPTED PRIVATE KEY",
    b"X509 CRL",
}
```
이 문제의 경우는 RSA PUBLIC KEY 스트링만 예외처리해주면 된다. 

flag
```
crypto{Doom_Principle_Strikes_Again}
```

</br></br></br>

# JSON Web Tokens - JSON in JSON

스트링으로 받는 건 역시 취약하다.

```
jerry","admin":"True
```

flag
```
crypto{https://owasp.org/www-community/Injection_Theory}
```

</br></br></br>

# JSON Web Tokens - RSA or HMAC? Part 2

https://github.com/silentsignal/rsa_sign2n
이 사이트에서 참고한 GCD 공격을 참고하여 public key를 생성할 수 있다. 

근데 생각보다 도커 구성에 해야 할 일이 많다. 

또 만들어진 public key를 rsa public key로 바꿔주어야 하는데, 이는 openssl 명령어로 진행할 수 있다. 

그 이후에는 part 1 문제와 동일하다. 
두 rsa public key가 생성되는데, 둘 중 하나만 작동하는 걸 확인할 수 있었다. 

ex.py
```python
import jwt
import base64
import json
from Crypto.PublicKey import RSA

rsa_pub = open("rsakey2.pem", "r").read()
print(jwt.encode({'username': "admin", 'admin': True}, rsa_pub, algorithm='HS256'))
```

flag
```
crypto{thanks_silentsignal_for_inspiration}
```

---
</br></br></br>


# TLS Part 1: The Protocol - Secure Protocols

크롬으로 켜서 자물쇠 버튼을 누르고 인증서 보기를 누르면 확인할 수 있다.

```
Let's Encrypt
```

</br></br></br>

# TLS Part 1: The Protocol - Sharks on the Wire

Wireshark로 파일을 열고 destination이 178.62.74.206인 것들만 골라 세어주면 된다.

```
15
```

</br></br></br>

# TLS Part 1: The Protocol - TLS Handshake

Wireshark로 패킷들을 봐보면 Server Hello,라고 적힌 패킷이 하나 있다.

그 패킷 정보를 보면 
```
Random: 67c6bf8ffda56fcb359fba7f0149f85422223cf021ab1a0af701de5dd2091498
```
요구한 랜덤 정보가 적혀 있다. 

</br></br></br>

# TLS Part 1: The Protocol - Saying Hello

```
https://www.ssllabs.com/ssltest/analyze.html
```
에서 tls1.cryptohack.org에 들어가서 tls1.2에 해당하는 cipher의 정보를 얻을 수 있다. 

TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384이고, 

openssl 이름은 다음과 같다.

```
ECDHE-RSA-AES256-GCM-SHA384
```
---
</br></br></br>

# TLS Part 1: The Protocol - Decrypting TLS 1.2

wireshark - preferences에 들어가서 주어진 pem 파일을 넣어서 키를 설정해주고

14번 패킷을 TLS로 열면 플래그가 포함된 정보가 보인다. 

```
........................................PRI * HTTP/2.0

SM

..............d...........................r.........A.M.....i<.'U.d.z...f.....S..j.?.).z....S..q..<....5.Ay.<.K
..............J."..../Oa.+
...d.}p.
n....6....S.*/*....................H.......v..cU.a..a.....R( .J.W.n.N..F._..u.b
&=L..*VBl(.\.61..!.IjJ..Y..I|.......=......The flag is: crypto{weaknesses_of_non_ephemeral_key_exchange}
```

flag
```
crypto{weaknesses_of_non_ephemeral_key_exchange}
```

</br></br></br>

# TLS Part 1: The Protocol - Decrypting TLS 1.3

거의 같은 문제이지만 이번에는 keylogfile.txt를 사용한다. 

preferences - protocols - TLS에 들어가 설정해주고

14번 패킷을 TLS로 follow해서 열어주면 플래그가 보인다. 

```
PRI * HTTP/2.0

SM

..............d...........................r.........A.M.erY...q....oz...f.....S..j.?.).z....S..q..<....5.Ay.<.K
..............J."..../Oa.+
...d.}p.
n....6....S.*/*............................................................H.......v..cU.a..a.....R( .J.....BS..._..u.b
&=L..*VBl(.\.34..!.IjJ..Y..I|......."......Flag: crypto{export_SSLKEYLOGFILE}
```

flag
```
crypto{export_SSLKEYLOGFILE}
```

</br></br></br>

# TLS Part 1: The Protocol - Authenticated Handshake

위에서 한 것과 같이 keylogfile을 로드하고 시키는 패킷들을 복사해주면 된다. 

마지막에 키 정보는 keylogfile.txt에서 얻을 수 있다. 

어떤 정보가 키인지 헷갈리지 않아야 한다.

어렵지 않은 문제니 코드는 생략하도록 하겠다.

```
b51d7b5fb12aa3d692140d8f1f80732610e99411ca0f6d928b0f60570cbc778e672457a729d7cf3b58bc174f00dc5d30
```

---
</br></br></br>

# Cloud - Megalomaniac 1

논문에도 잘 나와있지만, u를 변조시키고 p랑 m의 크기를 비교하면서 1024번 이분탐색을 하면 된다. 

ex.sage
```python
from Crypto.Hash import SHA256, SHA512
from Crypto.Cipher import AES
from Crypto.Util.number import *
from Crypto.Util.Padding import pad, unpad
from pwn import *
import json
import random

def json_recv():
    line = io.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    io.sendline(request)

io = remote("socket.cryptohack.org", 13408)

io.recvline()
io.recvline()
io.recvline()
io.recvline()

data = json_recv()
master_key_enc = data["master_key_enc"]
n, e = data["share_key_pub"]
share_key_enc = data["share_key_enc"]

share_key_enc_u_rip = share_key_enc[:-64] + "00" * 16 + share_key_enc[-32:]
assert len(share_key_enc) == len(share_key_enc_u_rip)

low = 1 << 512
high = 1 << 1024

while 1:
	if high == low:
		break

	sid = (low + high) // 2
	print(high - low)
	json_send({
		"action": "wait_login"
		})
	io.recvline()
	io.recvline()
	json_send({
		"action": "send_challenge",
		"SID_enc": bytes.hex(long_to_bytes(int(pow(sid, e, n)))),
		"share_key_enc": share_key_enc_u_rip,
		"master_key_enc": master_key_enc
		})
	sid_recv = json_recv()["SID"]
	
	if long_to_bytes(sid)[:-16] == bytes.fromhex(sid_recv):
		low = sid + 1
	else:
		high = sid

p = low
q = n // p
assert p * q == n

json_send({
	"action": "get_encrypted_flag"
	})
encrypted_flag = bytes.fromhex(json_recv()["encrypted_flag"])

secret = SHA256.new(long_to_bytes(p) +
                    long_to_bytes(q)).digest()
flag = unpad(AES.new(secret, AES.MODE_ECB).decrypt(encrypted_flag), 16)


print(flag)

io.interactive()
```

flag
```
crypto{M4lleaBl3_3nCRypt1on_g0n3_wr0nG_:'(}
```

</br></br></br>

# Cloud - Megalomaniac 2

사실 megalomaniac 1을 볼때부터 coppersmith를 쓸 생각을 했는데 괜히 논문 읽느라 이제서야 그 방법을 썼다.

ex.sage
```python
from Crypto.Hash import SHA256, SHA512
from Crypto.Cipher import AES
from Crypto.Util.number import *
from Crypto.Util.Padding import pad, unpad
from pwn import *
import json
import random

def json_recv():
    line = io.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    io.sendline(request)

io = remote("socket.cryptohack.org", 13409)

io.recvline()
io.recvline()
io.recvline()
io.recvline()

data = json_recv()
master_key_enc = data["master_key_enc"]
n, e = data["share_key_pub"]
share_key_enc = data["share_key_enc"]

share_key_enc_u_rip = share_key_enc[:-64] + "00" * 16 + share_key_enc[-32:]
assert len(share_key_enc) == len(share_key_enc_u_rip)



sid = 1 << 1026
json_send({
	"action": "wait_login"
	})
io.recvline()
io.recvline()
json_send({
	"action": "send_challenge",
	"SID_enc": bytes.hex(long_to_bytes(int(pow(sid, e, n)))),
	"share_key_enc": share_key_enc_u_rip,
	"master_key_enc": master_key_enc
	})
sid_recv = bytes.fromhex(json_recv()["SID"])
approx = bytes_to_long(sid_recv + b"\x00" * 16)



P.<x> = PolynomialRing(Zmod(n), implementation='NTL')
f = x + approx - sid
d = f.small_roots(X=2**290, beta=0.4, epsilon=1/32)

p = Integer(gcd(d[0] + approx - sid, n))
q = n // p
assert p * q == n

json_send({
	"action": "get_encrypted_flag"
	})
encrypted_flag = bytes.fromhex(json_recv()["encrypted_flag"])

secret = SHA256.new(long_to_bytes(p) +
                    long_to_bytes(q)).digest()
flag = unpad(AES.new(secret, AES.MODE_ECB).decrypt(encrypted_flag), 16)


print(flag)


io.interactive()
```

flag
```
crypto{W4s_th4t_rea11y_Any_hard3r??}
```

</br></br></br>

# Cloud - Megalomaniac 3

u를 한 블록 우리가 필요로 하는 값(node_key_enc) 블록을 넣어주고 변조된 값에서 u를 복구해주면 된다. 패딩 길이 주의!

ex.sage
```python
from Crypto.Hash import SHA256, SHA512
from Crypto.Cipher import AES
from Crypto.Util.number import *
from Crypto.Util.Padding import pad, unpad
from pwn import *
import json
import random

def json_recv():
    line = io.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    io.sendline(request)

io = remote("socket.cryptohack.org", 13410)

io.recvline()
io.recvline()
io.recvline()
io.recvline()
data = json_recv()
master_key_enc = data["master_key_enc"]
n, e = data["share_key_pub"]
share_key_enc = data["share_key_enc"]

io.recvline()
io.recvline()
data = json_recv()
node_key_enc = data["node_key_enc"]
file_enc = data["file_enc"]

io.recvline()
io.recvline()
data = json_recv()
n, e, p = data["share_key"]
q = n // p
assert p * q == n

share_key_enc_u_rip = share_key_enc[:-64] + node_key_enc + share_key_enc[-32:]
assert len(share_key_enc) == len(share_key_enc_u_rip)



json_send({
	"action": "wait_login"
	})
io.recvline()
io.recvline()


json_send({
	"action": "send_challenge",
	"SID_enc": bytes.hex(long_to_bytes(int(pow(p, e, n)))),
	"share_key_enc": share_key_enc_u_rip,
	"master_key_enc": master_key_enc
	})
sid_recv = bytes.fromhex(json_recv()["SID"])
approx = bytes_to_long(sid_recv + b"\x00" * 16)

val = (approx // p + 1) * p
assert val >> 128 == approx >> 128

u_fake = long_to_bytes(((val // p) * pow(p, -1, q)) % q)
u_real = long_to_bytes(pow(p, -1, q))

idx = 0
while u_fake[idx] == u_real[idx]:
	idx += 1


node_key = u_fake[idx:idx + 16]
file = unpad(AES.new(node_key, AES.MODE_ECB).decrypt(bytes.fromhex(file_enc)), 16)

print(file)

io.interactive()
```

flag
```
crypto{1ntegr1ty_ch3cks_are_n0T_0nly_th3Re_t0_mak3_crYptogr4phers_H4ppy!}
```

---
</br></br></br>