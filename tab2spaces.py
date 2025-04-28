import os

dirname = "_posts/"
# dirname = "capstone/"

post_list = os.listdir(dirname)

for fname in post_list:
	if fname[0] == ".":
		continue

	f = open(dirname + fname, "r")
	content = f.read()
	f.close()

	content = content.replace("\t", "    ")

	os.remove(dirname + fname)

	f = open(dirname + fname, "w")
	f.write(content)
	f.close()

	print(f"Post \"{fname}\" replaced!!")