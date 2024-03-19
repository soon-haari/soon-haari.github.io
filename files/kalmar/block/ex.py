from Crypto.Hash import SHA256
from pwn import *
from Crypto.Util.number import *
from tqdm import trange
import random

# io = process(["python3", "chal.py"])
io = remote("blockchain.chal-kalmarc.tf", 1337)

cur = "user"
io.recvuntil(b"your public key is ")
cur_pk = int(io.recvline())
io.recvuntil(b"your private key is ")
cur_sk = int(io.recvline())

slot = -1
bal = 1000
found = 9000
tot = 10000

accs = {cur : [cur, cur_pk, cur_sk]}

def gen():
	# global slot
	global bal
	global tot

	ran = random.randrange(10000000)
	io.sendline(b"create")
	name = f"user{ran}"

	h = SHA256.new()
	h.update(cur.encode() + name.encode())
	nonce =int(h.hexdigest()*7,16)
	sig = pow(nonce, cur_sk, cur_pk)


	to_send = cur + " " + name + " " + str(sig)
	io.sendline(to_send.encode())

	io.recvuntil(b"key: ")
	newpk = int(io.recvline())
	io.recvuntil(b"key: ")
	newsk = int(io.recvline())

	accs[name] = [name, newpk, newsk]

	bal -= 10
	tot -= 10
	# slot += 1

def send_and_ownership(k):
	# global slot
	global bal
	global tot
	global cur
	global cur_pk
	global cur_sk

	acc = accs[k]
	newname, npk, nsk = acc

	io.sendline(b"send")
	
	h = SHA256.new()
	h.update(cur.encode() + b':' + newname.encode() + b':' + str(bal).encode())
	nonce = int(h.hexdigest()*7,16)

	sig = pow(nonce, cur_sk, cur_pk)

	to_send = cur + " " + newname + " " + str(bal) + " " + str(sig)
	# print(io.recvline())
	io.sendline(to_send.encode())

	bal -= 1
	tot -= 1
	cur = newname
	cur_pk = npk
	cur_sk = nsk

	# print("huh")

	# slot += 1

def mint():
	# global slot
	global bal
	global tot

	io.sendline(b"mintblock")
	io.sendline(cur.encode())

	bal += 20
	tot += 20
	# slot += 1

	# io.interactive()
	# exit()
	io.recvuntil(b"account user")
	io.recvuntil(b"now has balance")

	tick()

# lottery:::245:::22101678010650976631742345786870302496057205354134020663594626775559015212915474730001428022329176438157539734055678774426195517957268907794661852075133595544918687304729399592641305365116371397789010730423191617875910815795528934192378983947939827952341980133767591705634054450547935215117842252107244560563099089646770221891984158442311022431895194534392275457658819720288670997740652549938476386971503438842022867225661329987407421633520178917476020484752504418545119436858653138045344785388673849200410165016530306954362385504697901146206139347329947859506542964591523464701801413963059030997103480795212509449009
# lottery:::245:::22101678010650976631742345786870302496057205354134020663594626775559015212915474730001428022329176438157539734055678774426195517957268907794661852075133595544918687304729399592641305365116371397789010730423191617875910815795528934192378983947939827952341980133767591705634054450547935215117842252107244560563099089646770221891984158442311022431895194534392275457658819720288670997740652549938476386971503438842022867225661329987407421633520178917476020484752504418545119436858653138045344785388673849200410165016530306954362385504697901146206139347329947859506542964591523464701801413963059030997103480795212509449009

def tick(t=False):
	global slot
	slot += 1

	if t:
		io.sendline(b"tick")

	# res = io.recvuntil(b"what would you like to do?").split(b"\n")
	# print(res)
	# tot += 20 * (len(res) - 3) // 2

def balan():
	global bal
	global tot
	io.sendline(b"balance")

	io.recvuntil(f"{cur} has balance ".encode())
	bal = int(io.recvline())
	io.recvuntil(b"total float = ")
	tot = int(io.recvline())

def view():
	print(f"{slot = }")
	print(f"{bal = }")
	print(f"{tot = }")
	print()
	# print(f"{accs.keys() = }")

tick()

for _ in range(10):
	view()
	gen()

cnt = 0

while True:
	cnt += 1
	if random.randrange(30) == 1:
		gen()
		continue

	if random.randrange(30) == 1:
		balan()
		view()
		continue


	if slot > 950:
		io.sendline(b"win")
		h = SHA256.new()
		h.update(b'I herebye declare myself victorious!!!!!!!')
		nonce =int(h.hexdigest()*7,16)

		sig = pow(nonce, cur_sk, cur_pk)
		to_send = cur + " " + str(sig)
		io.sendline(to_send.encode())
		io.interactive()
	view()

	h = SHA256.new()
	heh = b'lottery:::' + str(slot).encode() + b':::' + str(cur_pk).encode()
	h.update(b'lottery:::' + str(slot).encode() + b':::' + str(cur_pk).encode())
	lottery_roll =int(h.hexdigest(),16)/2**256 
	winning_prob = 1 - (1 - 0.2)**(bal / tot)

	if lottery_roll <= winning_prob * 0.9:
		# print(lottery_roll)
		# print(winning_prob)
		# print(f"myheh: {heh}")
		mint()
		# io.interactive()
		# input()
		# io.interactive()
		continue

	suc = False
	for k in accs:
		name, pk, sk = accs[k]

		h = SHA256.new()
		h.update(b'lottery:::' + str(slot).encode() + b':::' + str(pk).encode())
		lottery_roll =int(h.hexdigest(),16)/2**256 

		winning_prob = 1 - (1 - 0.2)**(bal / tot * 0.7)

		if lottery_roll <= winning_prob:
			# print(lottery_roll)
			# print(winning_prob)
			# print("SENDING")
			suc = True
			send_and_ownership(k)
			break

	if suc:
		continue

	tick(t=True)




io.interactive()