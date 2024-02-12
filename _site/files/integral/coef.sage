R.<a, b> = PolynomialRing(QQ, 'a, b')
b = a^2

coefs = [1, 0, 0, 0, 0]

for i in range(1, 5):
	new_coefs = [0] * 5

	for j in range(i):
		# 1 / (x - ab^j)(x - ab^i)
		# i > j
		
		# = 1 / (ab^j - ab^i) * 1 / (x - ab^j)
		# + 1 / (ab^i - ab^j) * 1 / (x - ab^i)

		new_coefs[j] += 1 / (a * b^j - a * b^i) * coefs[j]
		new_coefs[i] += 1 / (a * b^i - a * b^j) * coefs[j]

	coefs = new_coefs

print(coefs)

test = 0

for i in range(5):
	test += coefs[i] / (-a * b^i)

print(test)