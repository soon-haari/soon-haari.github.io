class Params:
	# clearly a mathematician and not a programmer came up with these names
	# because a dozen single-letter names would ordinarily be insane
	w = 32              # word size
	n = 624             # degree of recursion
	m = 397             # middle term
	r = 31              # separation point of one word
	a = 0x9908b0df      # bottom row of matrix A
	u = 11              # tempering shift
	s = 7               # tempering shift
	t = 15              # tempering shift
	l = 18              # tempering shift
	b = 0x9d2c5680      # tempering mask
	c = 0xefc60000      # tempering mask

def undo_xor_rshift(x, shift):
	''' reverses the operation x ^= (x >> shift) '''
	result = x
	for shift_amount in range(shift, Params.w, shift):
		result ^= (x >> shift_amount)
	return result

def undo_xor_lshiftmask(x, shift, mask):
	''' reverses the operation x ^= ((x << shift) & mask) '''
	window = (1 << shift) - 1
	for _ in range(Params.w // shift):
		x ^= (((window & x) << shift) & mask)
		window <<= shift
	return x

def temper(x):
	''' tempers the value to improve k-distribution properties '''
	x ^= (x >> Params.u)
	x ^= ((x << Params.s) & Params.b)
	x ^= ((x << Params.t) & Params.c)
	x ^= (x >> Params.l)
	return x
	
def untemper(x):
	''' reverses the tempering operation '''
	x = undo_xor_rshift(x, Params.l)
	x = undo_xor_lshiftmask(x, Params.t, Params.c)
	x = undo_xor_lshiftmask(x, Params.s, Params.b)
	x = undo_xor_rshift(x, Params.u)
	return x