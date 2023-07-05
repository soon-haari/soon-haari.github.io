# One Time Pad - Gotta Go Fast

time.time()이 같은 값을 가지게 빨리 1초 내에 두 입출력을 받으면 된다. 

물론 한번에 안 될 수도 있으니 될 때까지 실행한다. 

ex.py
```python
import time
from Crypto.Util.number import *
import hashlib
from pwn import *
import json
import codecs

r = remote('socket.cryptohack.org', 13372)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

flag_len = len(b'crypto{????????????????????}')

while 1:
	json_send({
		"option": "get_flag"
		})
	enc_flag = bytes.fromhex(json_recv()["encrypted_flag"])
	json_send({
		"option": "encrypt_data",
		"input_data": bytes.hex(b"\x00" * flag_len)
		})
	key = bytes.fromhex(json_recv()["encrypted_data"])

	pt = xor(enc_flag, key)
	if pt[:6] == b"crypto":
		print(pt)
		break

r.interactive()
```

flag
```
crypto{t00_f4st_t00_furi0u5}
```

</br></br></br>

# One Time Pad - No Leaks

플래그 바이트별로 한 바이트라도 일치하면 에러를 내뿜고 ciphertext를 안 준다. 
즉 계속 돌려서 255개가 등장하면 남은 하나가 플래그 바이트라는 뜻이다.

ex.py
```python
import time
from Crypto.Util.number import *
import hashlib
from pwn import *
import json
import codecs
import base64

r = remote('socket.cryptohack.org', 13370)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

r.recvline()

flag_len = len("crypto{????????????}")

poss = []

for i in range(20):
	line = [i for i in range(256)]
	poss.append(line)

while 1:

	json_send({
		"msg": "request"
		})
	res = json_recv()
	if not "ciphertext" in res:
		continue

	ct = base64.b64decode(res["ciphertext"].encode())
	
	for i in range(20):
		if ct[i] in poss[i]:
			poss[i].remove(ct[i])

	current = ""
	for i in range(20):
		current += str(len(poss[i])) + " "

	print(current)

	tot = 0
	for i in range(20):
		tot += len(poss[i])

	if tot == 20:
		break

flag = ""
for i in range(20):
	flag += chr(poss[i][0])

print(flag)

r.interactive()
```

flag
```
crypto{unr4nd0m_07p}
```

---
</br></br></br>

# PRNGS - Lo-Hi Card Game

푸는 방식 자체는 간단한다. 

a -> b -> c일 때, 

b = a \* m + i

c = b \* m + i

즉 m = (c - b) / (b - a), i = b - a \* m을 이용해 그 뒤에 값들을 다 구할 수 있다.

어렵지는 않지만 구현이 귀찮다.

```python
from Crypto.Util.number import *
from pwn import *
import json

r = remote('socket.cryptohack.org', 13383)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

VALUES = ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six',
          'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']
SUITS = ['Clubs', 'Hearts', 'Diamonds', 'Spades']

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

    def __str__(self):
        return f"{self.value} of {self.suit}"

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return VALUES.index(self.value) > VALUES.index(other.value)

cards = [str(Card(value, suit)) for suit in SUITS for value in VALUES]

def rebase(n):
    if n < 52:
        return [n]
    else:
        return [n % 52] + rebase(n // 52)

def card_2_idx(card):
	for i in range(52):
		if cards[i] == card:
			return i

def gen_num(arr):
	val = 0
	for i in range(len(arr)):
		val += (52 ** i) * arr[len(arr) - i - 1]
	return val

rounds = []

idxs = []

while 1:
	game = json_recv()
	print(game)
	idx = card_2_idx(game["hand"])
	msg = game["msg"].split()

	if "10" in msg:
		rounds.append(10)
	elif "11" in msg:
		rounds.append(11)
	
	idxs.append(idx)


	num = idx % 13 + 1
	if num > 7:
		json_send({
			"choice": "l"
			})
	else:
		json_send({
			"choice": "h"
			})


	if len(rounds) == 4:
		break

print(idxs)

num1 = idxs[:rounds[0]]
num2 = idxs[rounds[0]:rounds[0] + rounds[1]]
num3 = idxs[rounds[0] + rounds[1]:rounds[0] + rounds[1] + rounds[2]]

print(num1)
print(num2)
print(num3)

num1 = gen_num(num1)
num2 = gen_num(num2)
num3 = gen_num(num3)


q = 2 ** 61 - 1

assert num1 < q
assert num2 < q
assert num3 < q

mul = ((num3 - num2) * pow(num2 - num1, -1, q)) % q
inc = (num2 - num1 * mul) % q

new_idxs = []

assert num2 % q == (mul * num1 + inc) % q
assert num3 % q == (mul * num2 + inc) % q

num = num3

for i in range(20):
	num = (mul * num + inc) % q
	new_idxs = rebase(num) + new_idxs


print(new_idxs)
new_idxs.pop()

while 1:
	game = json_recv()
	print(game)
	idx = card_2_idx(game["hand"])
	
	current_val = idx % 13
	next_val = new_idxs.pop() % 13
	# print(next_val)

	if current_val > next_val:
		json_send({
			"choice": "l"
			})
	else:
		json_send({
			"choice": "h"
			})


	if game["round"] == 200:
		break

r.interactive()

```

flag
```
crypto{shuffl3_tr4ck1n6_i5_1t_l3g4l?}
```

</br></br></br>

# PRNGS - Nothing Up My Sleeve

ECC가 포함되어 있길래 ECC를 공부하고 나서 컴백했다. 

먼저 P256은 모든 파라미터가 정의되어 있는 커브라는 걸 알았다. 

문제 해석을 끝내자마자 든 생각이 Q에 커브의 근(y^2 = 0이 되는 x, y)를 보내주면 되지 않을까 생각했는데, P256은 근이 없는 커브라는 걸 알게 되었다. 

그런데 난수 생성 과정을 잘 봐보면 내가 Q = P로 설정할 경우 그냥 곱한 값의 x를 시드로 다시 설정하는 것을 볼 수 있었다. 그리고 사용되는 값은 다음 라운드에 사용될 시드의 하위 240비트이기 때문에, 16비트 브루트 포스만으로 다음의 값을 예측할 수 있다. 

ex.sage
```python
from Crypto.Util.number import *
from pwn import *
import json
import random

def rebase(n, b=37):
    if n < b:
        return [n]
    else:
        return [n % b] + rebase(n//b, b)

def renum(l, b=37):
    n = 0
    for i in range(len(l)):
        n += l[i] * b**(len(l) - i - 1)
    return n

r = remote("socket.cryptohack.org", 13387)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

p = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
a = 0xffffffff00000001000000000000000000000000fffffffffffffffffffffffc
b = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
x = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296
y = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5
E = EllipticCurve(GF(p), [a, b])
point = E(x, y)

r.recvline()
json_send({
    "x": hex(x),
    "y": hex(y)
    })
json_recv()

first_round = []
second_round = []

while 1:
    r.sendline(f"{{\"choice\": \"ODD\"}}")
    data = json_recv()
    print(data)

    first_round.append(data["spin"])
    if data["msg"][:1] == "G":
        break

while 1:
    r.sendline(f"{{\"choice\": \"ODD\"}}")
    data = json_recv()
    print(data)

    second_round.append(data["spin"])
    if data["msg"][:1] == "G":
        break

print(first_round)
print(len(first_round))
print(second_round)
print(len(second_round))


seed = renum(first_round)
second = renum(second_round)
num = point * seed
add = point * (1 << 240)

third = 0

for i in range(1 << 16):
    new_x = int(num.xy()[0])

    if (new_x & (2**(8 * 30) - 1)) == second:
        third = rebase(int((point * new_x).xy()[0]) & (2**(8 * 30) - 1))
        break

    num = num + add

assert third != 0

while 1:
    r.sendline(f"{{\"choice\": {third.pop()}}}")
    data = json_recv()
    print(data)

    second_round.append(data["spin"])
    if data["msg"][:1] == "C":
        break

r.interactive()
```

flag
```
crypto{No_Str1ngs_Att4ch3d}
```

</br></br></br>

# PRNGS - RSA vs RNG

문제를 봐보면 P를 랜덤으로 생성하고, 거기서 get_num을 소수가 나올 때까지 돌리고 그걸 Q로 놓은 RSA가 구현되어 있다. 

그리고 get_num은 점화식이다. a를 곱하고 b를 더해주는 것을 반복한다.

그래서 결론은 Q = a1 \* P + b1의 꼴로 쓸 수 있다는 것이다. 몇번 get_num을 했는지는 모르기 때문에 1번부터 될때까지 계속 돌려주어야 한다. 하지만 512비트 부근에는 소수가 1000개중 한개 이상은 있기 때문에 문제가 되지 않는다. 

그리고 알 수 있는 점은 P \* (a1 \* P + b1) = N (mod 2^512)라는 식을 이용하면 된다. 

P에 관한 2차방정식이기 때문에 근의공식 푸는 것처럼 풀면 된다. 

mod가 소수가 아니라 2^512이기 때문에 삽질 꽤나 했다. 

2^512를 기준으로 한 루트 c를 구할 때는 c를 c' \* 2^(2k)의 꼴로 표현하고 (c'은 당연히 홀수) 

x = x' \* 2^k으로 놓고 잘 지지고 볶으면 된다. (x'도 당연히 홀수)

Mod(c, MOD).sqrt()를 이용해서 근을 하나만 찾고

(x1 - x2) \* (x1 + x2) = 0 (mod 2^512 혹은 그 이하)를 생각하면 x1, x2가 모두 홀수기에 모든 근들을 찾기 간편하다.

ex.sage
```python
from Crypto.Util.number import *

data = {"N": 95397281288258216755316271056659083720936495881607543513157781967036077217126208404659771258947379945753682123292571745366296203141706097270264349094699269750027004474368460080047355551701945683982169993697848309121093922048644700959026693232147815437610773496512273648666620162998099244184694543039944346061, "E": 65537, "ciphertext": "04fee34327a820a5fb72e71b8b1b789d22085630b1b5747f38f791c55573571d22e454bfebe0180631cbab9075efa80796edb11540404c58f481f03d12bb5f3655616df95fb7a005904785b86451d870722cc6a0ff8d622d5cb1bce15d28fee0a72ba67ba95567dc5062dfc2ac40fe76bc56c311b1c3335115e9b6ecf6282cca"}

N = data["N"]
E = data["E"]
ct = bytes.fromhex(data["ciphertext"])

MOD = 2**512
A = 2287734286973265697461282233387562018856392913150345266314910637176078653625724467256102550998312362508228015051719939419898647553300561119192412962471189
B = 4179258870716283142348328372614541634061596292364078137966699610370755625435095397634562220121158928642693078147104418972353427207082297056885055545010537

vec = [1, 0]

def pow2(c):
	if c % 2 == 1:
		return 0

	return pow2(c // 2) + 1

def get_sqrt(c):
	pow_2 = pow2(c)

	if pow_2 % 2 == 1:
		return []

	sq_list = []

	sq = 0

	try:
		sq = int(Mod(c // (2 ** pow_2), MOD // (2 ** pow_2)).sqrt())
	except:
		return []

	for i in range(2 ** (pow_2 // 2)):
		sq_list.append((sq * 2 ** (pow_2 // 2) + i * 2 ** (512 - pow_2 // 2)) % MOD)
		sq_list.append((-sq * 2 ** (pow_2 // 2) + i * 2 ** (512 - pow_2 // 2)) % MOD)
		sq_list.append(((sq + 2 ** (512 - pow_2 - 1)) * 2 ** (pow_2 // 2) + i * 2 ** (512 - pow_2 // 2)) % MOD)
		sq_list.append((-(sq + 2 ** (512 - pow_2 - 1)) * 2 ** (pow_2 // 2) + i * 2 ** (512 - pow_2 // 2)) % MOD)
	
	for sqq in sq_list:
		assert sqq ** 2 % MOD == c

	return sq_list
	

cnt = 0
while 1:
	vec[0] = (vec[0] * A) % MOD
	vec[1] = (vec[1] * A + B) % MOD
	cnt += 1

	if vec[1] % 2 == 1:
		continue

	# vec[0] * p ^ 2 + vec[1] * p = N (MOD)

	b = vec[1] * pow(vec[0], -1, MOD)
	c = N * pow(vec[0], -1, MOD)
	# p ^ 2 + b * p = c (MOD)

	b = (b // 2) % MOD
	c = (c + b ** 2) % MOD

	# (p + b) ^ 2 = c

	sqrt_list = get_sqrt(c)


	for sqrt in sqrt_list:
		assert (sqrt ** 2) % MOD == c % MOD
		p = (sqrt - b) % MOD
		assert (vec[0] * p ** 2 + vec[1] * p) % MOD == N % MOD

		q = (vec[0] * p + vec[1]) % MOD

		if p * q == N:
			phi = (p - 1) * (q - 1)
			D = pow(E, -1, phi)
			pt = pow(bytes_to_long(ct), D, N)
			print(long_to_bytes(pt))
			exit()

```

flag
```
crypto{pseudorandom_shamir_adleman}
```

</br></br></br>

# PRNGS - Trust Games

48비트 중 8개의 8비트를 알려주고, 8개의 값이 mul, inc로 연결되어 있다. 

특별한 알고리즘은 생각나지 않았고, 연산이 매우 간단해서 2^40이면 가정 내 pc로 브루트포스가 가능해보였다. 

c로 해야되서 c파일, 파이썬 파일 두 개를 만들어서 진행했다. 

c에도 socket기능이 있다고 하는게 공부하기가 귀찮았다. 

c코드만 적도록 하겠다. 

main.c
```c
#include <stdio.h>

typedef long long ll;

int main(int argc, const char * argv[]) {
    ll datas[8] = {126, 131, 157, 102, 183, 124, 37, 132};
    ll mul = 0x1337deadbeef;
    ll add = 0xb;
    ll val;
    ll start = datas[0] << 40;
    ll end = (datas[0] + 1) << 40;
    for(ll i = start; i < end; i++){
        if((i & 0xffffffff) == 0)
            printf("%lld\n", (i - start) >> 32);
        
        val = i * mul + add;
        
        if(((val >> 40) & 0xff) != datas[1])
            continue;
        val = val * mul + add;
        
        if(((val >> 40) & 0xff) != datas[2])
            continue;
        val = val * mul + add;
        
        if(((val >> 40) & 0xff) != datas[3])
            continue;
        val = val * mul + add;
    
        if(((val >> 40) & 0xff) != datas[4])
            continue;
        val = val * mul + add;
        printf("huh\n");
        if(((val >> 40) & 0xff) != datas[5])
            continue;
        val = val * mul + add;
        
        if(((val >> 40) & 0xff) != datas[6])
            continue;
        val = val * mul + add;
        
        if(((val >> 40) & 0xff) != datas[7])
            continue;
        val = val * mul + add;
        
        
        printf("%lld\n", i);
        break;
    }
    
    return 0;
}

```

flag
```
crypto{L4ttice_C0mpl1ant_G4m3}
```

정풀은 또 LCG라는 기법이었다. 공부해야겠다.

---
</br></br></br>

# LFSR - L-Win

clock을 해석해보면 384개의 수 중 4개를 골라서 xor해서 앞에껄 없에고 뒤에 넣는다.

항상 같은 인덱스 4개이고, 이는 문제에서 비밀로 하였다. 

이러한 쌍들이 많이 존재하므로 모든 쌍을 검사해서 4개의 인덱스를 구해서 역연산을 구현하면 기존 플래그를 구할 수 있다. 

</br>

384를 n이라고 했을 때

nC4, 즉 O(n^4)시간복잡도를 구현하면 시간이 너무 오래 걸릴 것 같았다. 
하지만 다 풀고나서 생각해보니 한 개는 무조건 0이 되어야 정보 손실이 없다는 상식을 이용하면 O(n^3)으로 할 수도 있었을 것 같다.

아무튼 머리를 잘 굴리면 O(n^2logn)으로도 풀린다. 

문제를 변형시켜보면 대략 1700쌍 정도의 정보가 존재하는데, 이 정보를 각각 다른 인덱스의 비트에 저장해서 1700비트짜리 수 384개를 만들고 1700비트짜리 결과값을 만든다.

384개 중 4개를 xor연산하면 결과값이 나오게 되는 4개를 찾으면 되는데,

384개 중 2개를 고른 384^2개를 set에 저장한 뒤
set에 있는 모든 원소와 결과값을 xor한 값이 set에 존재하는지만 확인하면 된다. 

즉 O(n^2logn)이다. 

ex.py
```python
from Crypto.Util.number import *

f = open("output.txt", "r").read()[:-1]

data_n = 2048
cycle = 384

datas = []
for i in range(data_n):
	if f[i] == '0':
		datas.append(0)
	else:
		datas.append(1)

numbers = [0] * 384
result = 0

for i in range(data_n - cycle):
	result |= datas[i + cycle] << i

	for j in range(cycle):
		numbers[j] |= datas[i + j] << i

double_set = set()
double_list = []

for i in range(cycle):
	for j in range(cycle):
		double_set.add(numbers[i] ^ numbers[j])
		double_list.append([numbers[i] ^ numbers[j], i, j])

secret = [] # 0, 6, 15, 16

def add_idx(n):
	for element in double_list:
		if element[0] == n:
			secret.append(element[1])
			secret.append(element[2])
			return

for n in double_set:
	xored = n ^ result
	if xored in double_set:
		add_idx(n)
		add_idx(xored)
		break

for i in range(16 * 48):
	b = datas[5] ^ datas[14] ^ datas[15] ^ datas[383]
	datas = [b] + datas[:-1]

m = 0

for i in range(384):
	m |= datas[383 - i] << i

print(long_to_bytes(m))
```

flag
```
crypto{minimal_polynomial_in_an_arbitrary_field}
```

</br></br></br>

# LFSR - Jeff's LFSR

아무리 생각해도 key_0 19비트 브루트포스밖에 생각이 안 나서 고민했는데

알고보니 그게 정풀이었다....

key_0을 다 만들면 key_1, key_2가 50프로가량 복구되는데, 
그 안에서 5개의 값이 xor해서 0이 되게 계속 검사해주면 된다. 
오류가 안 나는 경우는 딱 한 경우밖에 없다. 

파이썬이 느려서 키를 찾는 과정은 c로 하였고, 파이썬에서 플래그를 구했다.

c코드가 많이 난잡한 점 양해 부탁드립니다. 

main.c
```c
#include <stdio.h>

int main() {
    int tangled_b[256] = {1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1};
    int key_0[300] = {0,};
    int key_1[300];
    int key_2[300];
    int cnt = 0;
    while(1){
        for(int i = 0; i < 256; i++){
            key_0[i + 19] = key_0[i] ^ key_0[i + 1] ^ key_0[i + 2] ^ key_0[i + 5];
        }
        for(int i = 0; i < 256; i++){
            if(key_0[i] == 1){
                key_1[i] = tangled_b[i];
                key_2[i] = 2;
            }
            else{
                key_1[i] = 2;
                key_2[i] = tangled_b[i];
            }
        }
        int check = 1;
        while(1){
            int count_good = 0;
            for(int i = 0; i + 23 < 256; i++){
                int count2 = 0;
                if(key_2[i] == 2)
                    count2++;
                if(key_2[i + 1] == 2)
                    count2++;
                if(key_2[i + 3] == 2)
                    count2++;
                if(key_2[i + 5] == 2)
                    count2++;
                if(key_2[i + 23] == 2)
                    count2++;
                
                if(count2 == 0){
                    if(key_2[i] ^ key_2[i + 1] ^ key_2[i + 3] ^ key_2[i + 5] ^ key_2[i + 23]){
                        check = 0;
                        break;
                    }
                    count_good++;
                }
                else if(count2 == 1){
                    if(key_2[i] == 2)
                        key_2[i] = (key_2[i] ^ key_2[i + 1] ^ key_2[i + 3] ^ key_2[i + 5] ^ key_2[i + 23]) & 1;
                    else if(key_2[i + 1] == 2)
                        key_2[i + 1] = (key_2[i] ^ key_2[i + 1] ^ key_2[i + 3] ^ key_2[i + 5] ^ key_2[i + 23]) & 1;
                    else if(key_2[i + 3] == 2)
                        key_2[i + 3] = (key_2[i] ^ key_2[i + 1] ^ key_2[i + 3] ^ key_2[i + 5] ^ key_2[i + 23]) & 1;
                    else if(key_2[i + 5] == 2)
                        key_2[i + 5] = (key_2[i] ^ key_2[i + 1] ^ key_2[i + 3] ^ key_2[i + 5] ^ key_2[i + 23]) & 1;
                    else if(key_2[i + 23] == 2)
                        key_2[i + 23] = (key_2[i] ^ key_2[i + 1] ^ key_2[i + 3] ^ key_2[i + 5] ^ key_2[i + 23]) & 1;
                }
            }
            if(check == 0)
                break;
            if(count_good == 256 - 23)
                break;
        }
        
        while(1){
            int count_good = 0;
            for(int i = 0; i + 27 < 256; i++){
                int count2 = 0;
                if(key_1[i] == 2)
                    count2++;
                if(key_1[i + 1] == 2)
                    count2++;
                if(key_1[i + 2] == 2)
                    count2++;
                if(key_1[i + 5] == 2)
                    count2++;
                if(key_1[i + 27] == 2)
                    count2++;
                
                if(count2 == 0){
                    if(key_1[i] ^ key_1[i + 1] ^ key_1[i + 2] ^ key_1[i + 5] ^ key_1[i + 27]){
                        check = 0;
                        break;
                    }
                    count_good++;
                }
                else if(count2 == 1){
                    if(key_1[i] == 2)
                        key_1[i] = (key_1[i] ^ key_1[i + 1] ^ key_1[i + 2] ^ key_1[i + 5] ^ key_1[i + 27]) & 1;
                    else if(key_1[i + 1] == 2)
                        key_1[i + 1] = (key_1[i] ^ key_1[i + 1] ^ key_1[i + 2] ^ key_1[i + 5] ^ key_1[i + 27]) & 1;
                    else if(key_1[i + 2] == 2)
                        key_1[i + 2] = (key_1[i] ^ key_1[i + 1] ^ key_1[i + 2] ^ key_1[i + 5] ^ key_1[i + 27]) & 1;
                    else if(key_1[i + 5] == 2)
                        key_1[i + 5] = (key_1[i] ^ key_1[i + 1] ^ key_1[i + 2] ^ key_1[i + 5] ^ key_1[i + 27]) & 1;
                    else if(key_1[i + 27] == 2)
                        key_1[i + 27] = (key_1[i] ^ key_1[i + 1] ^ key_1[i + 2] ^ key_1[i + 5] ^ key_1[i + 27]) & 1;
                }
            }
            if(check == 0)
                break;
            if(count_good == 256 - 27)
                break;
        }
        
        if(check == 1){
            printf("Yippie! key is \n[");
            for(int i = 0; i < 19; i++)
                printf("%d, ", key_0[i]);
            for(int i = 0; i < 27; i++)
                printf("%d, ", key_1[i]);
            for(int i = 0; i < 23; i++){
                if(i < 22)
                    printf("%d, ", key_2[i]);
                else
                    printf("%d]\n", key_2[i]);
            }
        }
        key_0[0]++;
        for(int i = 0; i < 18; i++){
            if(key_0[i] == 2){
                key_0[i + 1]++;
                key_0[i] = 0;
            }
        }
        if(key_0[18] == 2)
            break;
        cnt++;
    }
    printf("yay finished %d\n", cnt);
    return 0;
}

```

ex.py
```python

from Crypto.Util.number import *
from Crypto.Random.random import getrandbits
import os
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

key = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

iv = bytes.fromhex('310c55961f7e45891022668eea77f805')
enc_flag = bytes.fromhex('2aa92761b36a4aad9a578d6cd7a62c52ba0709cb560c0ecff33a09e4af43bff0a1c865023bf28b387df91d6319f0e103d39dda88a88c14cfcec94c8ad02a6fb3152a4466c1a184f69184349e576d8950cac0a5b58bf30e67e5269883596a33a6')

def decrypt_flag(key):
    sha1 = hashlib.sha1()
    sha1.update(str(key).encode('ascii'))
    key = sha1.digest()[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(enc_flag), 16)
    return plaintext

key_int = 0
for i in range(69):
	key_int |= key[i] << (68 - i)

print(decrypt_flag(key_int))
```

flag
```
crypto{Geffe_generator_is_a_textbook_example_to_show_correlation_attacks_on_LFSR}
```

</br></br></br>

# LFSR - LFSR Destroyer

lfsr algebraic annihilator이라는 미친 힌트를 받았다.

https://doc.sagemath.org/html/en/reference/cryptography/sage/crypto/boolean_function.html

이곳에서 Boolean polynomial의 annihilator에 대해 자세히 나와 있다. 

또한 https://rkm0959.tistory.com/229 에서 사용법을 자세이 알게 될 수 있었던 것 같다. 

실제로 이 문제의 괴상한 다항식은 2차의 annihilator이 존재한다. 

이 뜻은, f값이 1이면 g값은 무조건 0이 확정이라는 뜻, 20000개의 정보 중 무려 10000개 이상의 선형 식을 알 수 있게 된다. 

그리고 128개 + 128C2 = 8256개의 변수를 가진 선형 연립방정식을 풀어주기만 하면 된다. 

나는 시간제한 15초를 맞추지 못하고 30초가 조금 더 걸리는 시간이 거의 걸렸는데 플래그를 그냥 주는 것이 의아했지만, 어차피 성능 좋은 데스크탑에서는 15초 이내로 들어갈 것 같기는 했다. 

ex.sage
```python
import os
import json
import signal

'''
from sage.crypto.boolean_function import BooleanFunction

f = [[0, 1, 2, 3], [0, 1, 2, 4, 5], [0, 1, 2, 5], [0, 1, 2], [0, 1, 3, 4, 5], [0, 1, 3, 5], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3, 4, 5], [0, 2, 3], [0, 3, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2], [1, 3, 5], [1, 3], [1, 4], [1], [2, 4, 5], [2, 4], [2], [3, 4], [4, 5], [4], [5]]
f.append([])
R.<x0,x1,x2,x3,x4,x5> = BooleanPolynomialRing(6)
x = [x0, x1, x2, x3, x4, x5]

poly = 0

for i in f:
	add = 1
	for j in i:
		add *= x[j]
	poly += add

B = BooleanFunction(poly)

anni = B.algebraic_immunity(annihilator=True)[1]
assert poly * anni == 0

print(anni)
'''


def clock():
	global key
	key = key[1:] + [key[0] + key[1] + key[2] + key[7]]

R.<k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31, k32, k33, k34, k35, k36, k37, k38, k39, k40, k41, k42, k43, k44, k45, k46, k47, k48, k49, k50, k51, k52, k53, k54, k55, k56, k57, k58, k59, k60, k61, k62, k63, k64, k65, k66, k67, k68, k69, k70, k71, k72, k73, k74, k75, k76, k77, k78, k79, k80, k81, k82, k83, k84, k85, k86, k87, k88, k89, k90, k91, k92, k93, k94, k95, k96, k97, k98, k99, k100, k101, k102, k103, k104, k105, k106, k107, k108, k109, k110, k111, k112, k113, k114, k115, k116, k117, k118, k119, k120, k121, k122, k123, k124, k125, k126, k127> = BooleanPolynomialRing(128)
key = [k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31, k32, k33, k34, k35, k36, k37, k38, k39, k40, k41, k42, k43, k44, k45, k46, k47, k48, k49, k50, k51, k52, k53, k54, k55, k56, k57, k58, k59, k60, k61, k62, k63, k64, k65, k66, k67, k68, k69, k70, k71, k72, k73, k74, k75, k76, k77, k78, k79, k80, k81, k82, k83, k84, k85, k86, k87, k88, k89, k90, k91, k92, k93, k94, k95, k96, k97, k98, k99, k100, k101, k102, k103, k104, k105, k106, k107, k108, k109, k110, k111, k112, k113, k114, k115, k116, k117, k118, k119, k120, k121, k122, k123, k124, k125, k126, k127]


term_to_num = {}

for i in range(128):
	term_to_num[key[i]] = i

cnt = 128
for i in range(128):
	for j in range(i + 1, 128):
		term_to_num[key[i] * key[j]] = cnt
		cnt += 1
term_n = 128 + 128 * 127 // 2
assert cnt == term_n

key[0] = 1

for _ in range(128 * 2):
	clock()

mat = []
expect = [0] * 20000


for i in range(20000):
	print(i)
	# [0, 16, 32, 64, 96, 127]
	
	x = [key[0], key[16], key[32], key[64], key[96], key[127]]

	'''
	f = [[0, 1, 2, 3], [0, 1, 2, 4, 5], [0, 1, 2, 5], [0, 1, 2], [0, 1, 3, 4, 5], [0, 1, 3, 5], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3, 4, 5], [0, 2, 3], [0, 3, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2], [1, 3, 5], [1, 3], [1, 4], [1], [2, 4, 5], [2, 4], [2], [3, 4], [4, 5], [4], [5]]

	poly = 1
	for j in f:
		term = 1
		for k in j:
			term *= x[k]
		poly += term
	# x0*x1 + x0*x2 + x0*x5 + x1*x2 + x1*x4 + x1*x5 + x2*x4 + x2 + x4*x5 + x5
	'''
	anni = x[0]*x[1] + x[0]*x[2] + x[0]*x[5] + x[1]*x[2] + x[1]*x[4] + x[1]*x[5] + x[2]*x[4] + x[2] + x[4]*x[5] + x[5]

	'''
	assert anni * poly == 0
	print(anni)
	'''

	vec = [0] * term_n

	for term in list(anni):
		try:
			vec[term_to_num[term]] = 1
		except:
			expect[i] = 1

	mat.append(vec)

	clock()


from pwn import *
from Crypto.Util.number import *
import time

def json_recv():
    line = io.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    io.sendline(request)

io = remote('socket.cryptohack.org', 13404)


start = time.time()



io.recvline()
json_send({
	"option": "encrypt",
	"plaintext": bytes.hex(b"\x00" * 2500)
	})

data = bytes.fromhex(json_recv()["ciphertext"])

ct_bit = []
for c in data:
	for i in range(8):
		if c & (1 << (7 - i)) > 0:
			ct_bit.append(1)
		else:
			ct_bit.append(0)

assert len(ct_bit) == 20000


A = []
b = []
for i in range(20000):
	if ct_bit[i] == 1:
		A.append(mat[i])
		b.append([expect[i]])
	if len(A) == 8500:
		break

print(len(A))
print(len(b))

A = Matrix(GF(2), A)
b = Matrix(GF(2), b)


mid = time.time()
print(f"mid time: {mid - start}")

res = A.solve_right(b)

key = 0

for i in range(128):
	key <<= 1
	key += int(res[i, 0])

key |= (1 << 127)

json_send({
	"option": "get_flag",
	"key": int(key)
	})

end = time.time()

print(f"solving time: {end - mid}")
print(f"total time: {end - start}")




io.interactive()
```

k0 ~ k127을 더 간편하게 정의할 수 있는 방법이 있는지 궁금하다.

flag
```
crypto{f1lT3r3d_Lf$Rs_4r3_5ubt13_4nd_y0u_c4n_e4siLy_m1ss_s0m3th1ng}
```

---
</br></br></br>

# Elgamal - Bit by Bit

문제를 요약하자면

g ^ x, g ^ y를 알려주고 g ^ (x \* y) \* m을 알려줄 때
m을 구하라는 문제'처럼' 보인다.

저걸 하려면 그냥 Diffie-Hellman을 뚫으라는 문제 같지만 
한 비트별로 연산을 하고 
padding = 2 ^ (8 \* e)이고, 
me = 2 ^ (8 \* e) << 1 + mbit, 즉 2 ^ (8 \* e + 1) + mbit이다. 

이를 어떻게 딱 반만 판별할 수 있을 까 생각하다가 legendre symbol을 생각했다. 

g가 제발 legendre symbol이 1이기를 기대하면서 검사해봤는데 다행히 1이었다. 
즉 mbit = 0이라면 me = 2 ^ (8 \* e + 1) * g ^ (x \* y)가 되어 
2 \* me는 legendre symbol이 1이 된다. 

이를 이용해서 판별하면 m의 모든 비트를 딸 수 있다. 

ex.py

```python
from Crypto.Util.number import *

f = open("output.txt", "r")

q = 117477667918738952579183719876352811442282667176975299658506388983916794266542270944999203435163206062215810775822922421123910464455461286519153688505926472313006014806485076205663018026742480181999336912300022514436004673587192018846621666145334296696433207116469994110066128730623149834083870252895489152123
g = 104831378861792918406603185872102963672377675787070244288476520132867186367073243128721932355048896327567834691503031058630891431160772435946803430038048387919820523845278192892527138537973452950296897433212693740878617106403233353998322359462259883977147097970627584785653515124418036488904398507208057206926

def legendre(n):
	return pow(n, (q - 1) // 2, q)

print(pow(g, (q - 1) // 2, q))

m = 0
idx = 0

while 1:
	s1 = f.readline()
	s2 = f.readline()
	if len(s1) == 0:
		break

	pubkey = "0x"
	i = 0
	while s1[i] != 'x':
		i += 1
	i += 1
	while 1:
		if s1[i] != ')':
			pubkey += s1[i]
		else:
			break
		i += 1
	pubkey = int(pubkey, 16)

	c1 = "0x"
	c2 = "0x"
	i = 0

	while s2[i] != 'x':
		i += 1
	i += 1
	while 1:
		if s2[i] != ',':
			c1 += s2[i]
		else:
			break
		i += 1
	c1 = int(c1, 16)

	while s2[i] != 'x':
		i += 1
	i += 1
	while 1:
		if s2[i] != ')':
			c2 += s2[i]
		else:
			break
		i += 1
	c2 = int(c2, 16)

	if legendre(c2 * 2) != 1:
		m |= 1 << idx

	idx += 1

print(long_to_bytes(m))
```

flag
```
crypto{s0m3_th1ng5_4r3_pr3served_4ft3r_encrypti0n}
```

---
</br></br></br>

# Secret Sharing Schemes - Armory

flag, sha256(flag), sha256(sha256(flag))를 가지고 조작을 하는데

별로 어렵지 않은데 사실 왜 100점인지 모르겠다.

ex.py
```python
import hashlib
from Crypto.Util.number import *

PRIME = 77793805322526801978326005188088213205424384389488111175220421173086192558047

f1 = 105622578433921694608307153620094961853014843078655463551374559727541051964080
b = 25953768581962402292961757951905849014581503184926092726593265745485300657424

f2 = bytes_to_long(hashlib.sha256(long_to_bytes(f1)).digest())

f0 = (b - f1 ** 2 * (f2 + 1)) % PRIME

print(long_to_bytes(f0))

```

flag
```
crypto{fr46m3n73d_b4ckup_vuln?}
```

</br></br></br>

# Secret Sharing Schemes - Toshi's Treasure

hyper이 아주 나쁜 놈이다. 돈을 그냥 뺏는 게 아니라 아주 전략적으로 뺏는다. 

먼저 SSSS share이라는 것을 처음 봐서 생소했지만 위키피디아에 잘 설명되어 있다. 

hyper이 x = 6, 다른 친구들은 x = 2, 3, 4, 5이기 때문에 hyper에 의해서 조작되는 부분은

(x - 2) \* (x - 3) \* (x - 4) \* (x - 5) / ((6 - 2) \* (6 - 3) \* (6 - 4) * (6 - 5)) \* y6

의 상수항이 더해진다.

즉 5 \* y6이다. 

이것만 이해하고 시키는대로 잘 하면 백만달러를 훔칠 수 있다. 

ex.py
```python
from Crypto.Util.number import *
from pwn import *
import json

r = remote("socket.cryptohack.org", 13384)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

my_1k_wallet_privkey = 0x8b09cfc4696b91a1cc43372ac66ca36556a41499b495f28cc7ab193e32eadd30

q = 2 ** 521 - 1

hyper_info = int(json_recv()["y"], 16)
json_recv()
json_recv()
json_recv()
json_recv()

json_send({
    "sender": "hyper",
    "x": 6,
    "y": hex(0)
    })

error_privkey = int(json_recv()["privkey"], 16)
json_recv()
json_recv()

original_wallet = (error_privkey + 5 * hyper_info) % q
fake_info = ((my_1k_wallet_privkey - error_privkey) * pow(5, -1, q)) % q

json_send({
    "sender": "hyper",
    "x": 6,
    "y": hex(fake_info)
    })

json_recv()
json_recv()
json_recv()

json_send({
    "privkey": hex(original_wallet)
    })

r.interactive()
```

flag
```
crypto{shoulda_used_verifiable_secret_sharing}
```

---
</br></br></br>

# Password Complexity - Bruce Schneier's Password

int64 범위 내에서 합이 소수, 곱이 소수가 되게 하면 된다.

모든 문자가 홀수여야 함은 추측할 수 있고, 몇 가지 조건만 넣어주면 의외로 결과가 잘 나온다. 

ex.py
```python
import numpy as np
import random
import re
from pwn import *
from Crypto.Util.number import *
import json
import codecs

r = remote('socket.cryptohack.org', 13400)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)

password = "Aa"

array = np.array(list(map(ord, password)))

while(1):
	password += "1"
	array = np.array(list(map(ord, password)))
	if isPrime(int(array.prod())) and isPrime(int(array.sum())):
		break

r.readline()
json_send({"password": password})

r.interactive()
```

flag
```
crypto{https://www.schneierfacts.com/facts/1341}
```

</br></br></br>

# Password Complexity - Bruce Schneier's Password: Part 2

각 글자별로 들어가는 횟수를 xi, ord(글자)를 ai라고 하자. 

그리고 우리가 목표하는 소수를 p라고 놓으면
```
a1x1 + a2x2 + ... akxk = p
```
를 만족한다. 

그렇다면 곱셈은 어떻게 정의할까, discrete_log를 사용하면 또 하나의 선형 식이 만들어진다. 

나의 경우는 3을 밑으로 하고 ai의 3을 기준으로 한 discrete_log가 존재하는 경우만 세기로 했다. 입력 가능한 문자들 중 17종류만이 살아남았다. 
```
log3(a1)x1 + log3(a2)x2 + ... log3(ak)xk = log3(p) 
```

이 두 식을 가지고 근을 어떻게 구할지 생각하다가 첫 번째 식을 이용해서 변수 하나를 소거하고

두 번째 식에 대입해서 변수가 k - 1개인 식을 LLL로 풀 생각을 했다. 

(만약 구한다고 쳐도 처음 소거한 변수를 구했는데 크기가 작으려면 ai * xi = t에서 t가 ai의 배수일 확률, 즉 약 1/ai의 확률로 브루트 포스를 해야 한다. 50번정도면 솔직히 할만 하다고 생각했다. )

그런데 이미 선형 방정식 쌍들이 여러 개 있을 때 풀어주는 모듈을 추천받았다. 

https://github.com/nneonneo/pwn-stuff/blob/36f0ecd80b05859acca803d4ddfb53454b448329/math/solvelinmod.py

이를 사용한 코드는 다음과 같다. 

ex.sage
```python
from pwn import *
from Crypto.Util.number import *
import json
from solvelinmod import solve_linear_mod

r = remote('socket.cryptohack.org', 13401)

def json_recv():
    line = r.recvline()
    return json.loads(line.decode())

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.sendline(request)



mod = 2^64
F = IntegerModRing(mod)

arr = []

for i in range(10):
	if (i + ord('0')) % 2 == 1:
		arr.append(i + ord('0'))

for i in range(26):
	if (i + ord('A')) % 2 == 1:
		arr.append(i + ord('A'))

for i in range(26):
	if (i + ord('a')) % 2 == 1:
		arr.append(i + ord('a'))

sum_arr = []
mul_arr = []

for n in arr:
	try:
		mul_arr.append(int(F(n).log(F(3))))
		sum_arr.append(n)
	except:
		continue

l = len(sum_arr)

print(sum_arr)

for i in range(l):
	assert pow(3, mul_arr[i], mod) == sum_arr[i]


prime = 0
while 1:
	prime = getPrime(15)
	try:
		F(prime).log(F(3))
		break
	except:
		continue

# prime = 20113
print(prime)
sum_result = prime
mul_result = int(F(prime).log(F(3)))
assert pow(3, mul_result, mod) == sum_result

l = 17

x = [0] * l

sum_eq = 0
mul_eq = 0
for i in range(l):
	x[i] = var(f'x{i}')
	sum_eq += sum_arr[i] * x[i]
	mul_eq += mul_arr[i] * x[i]

# print(sum_eq == sum_result)
# print(mul_eq == mul_result)

x_dict = {}
for i in range(l):
	x_dict[x[i]] = 50

# print(x_dict)

ans = solve_linear_mod([(sum_eq == sum_result, mod), (mul_eq == mul_result, mod)], x_dict)
print(ans)

result = []

for i in range(l):
	result.append(ans[x[i]])
	if result[i] < 0:
		print("fail")
		exit()

send = ""

for i in range(l):
	send += chr(sum_arr[i]) * result[i]

print(send)

'''
password = "11111111111111111111111111333333333333333333333999999999999999999999999999999AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCCCCCCCCCCCCIIIIIIIIIIIIIIIIIIIIIKKKKKKKKKKKKKKKKKQQQQQQQQQQQQQQSSSSSSSSSSSSSSSSSYYYYYYYYYYYYYYYYYYYYYYYYYaaaaaaaaaaaaaaaaaaaaaaaaaaacccccccccccccccccccccccccccccccccciiiiiiiiiiiiiiiiiiiiiikkkkkkkkkkkqqqqqqqqqqqqqqqqqqqssssssssssssssssyyyyyyyyyyyyyyyyyyyyyyy"

json_send({
	"password": password
	})

r.interactive()
'''
```

flag
```
crypto{!fact_in_#bot-chat}
```

하지만 LLL 사용 시 특정 값을 고정해두는 방법은 꼭 배워보고 싶다. 

---
</br></br></br>