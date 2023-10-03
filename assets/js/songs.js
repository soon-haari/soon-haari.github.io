const p = document.createElement("to_append");

let songs = [
	["cat flag", "/flag"],
	//0
	["어어어 어어어 푸푸푸 또 허허허 우우우적 거거거 리더던 시 저저절 나라면", "https://open.spotify.com/track/1IJxbEXfgiKuRx6oXMX87e"],
	["Wonder why connected feelings lingering to us", "https://open.spotify.com/track/3cqZ2Vsxt5Gi5ZLx3h5DAW"],
	["Went low, went high, what matters is now", "https://open.spotify.com/track/1rIKgCH4H52lrvDcz50hS8"],
	["나 웃을게요 많이 그대를 위해 많이", "https://open.spotify.com/track/3eOXiUUfz9OhyVqoREsJYe"],
	["네가 흘린 눈물이 마법의 주문이 되어", "https://open.spotify.com/track/2AAeXb4diy9tyBZqEhgnpE"],
	//5
	["하지만 내 노래는 누굴 위한 걸까", "https://open.spotify.com/track/1nlsjS6O7emQKryh2Azatg"],
	["Hold on, hold on, fuck that", "https://open.spotify.com/track/5mCPDVBb16L4XQwDdbRUpz"],
	["But no more syrup left in my bottle, damn", "https://open.spotify.com/track/3BbS4oLUDNOLctshheQucc"],
	["Cause every closed door is just a intro of a brand new story", "https://open.spotify.com/track/4Xnuh5fEHbHpxI3leaQSLN"],
	["Now the crowd's gone, is it over? Is it over?", "https://open.spotify.com/track/0i2THDeAhJma8FrUVy90No"],
	//10
	["Why did I ever go back?", "https://open.spotify.com/track/6q6nBdrtcJubvRog3tRhEn"],
	["Everyone else has left now", "https://open.spotify.com/track/23TPP1eeElFfvYVznskwCY"],
	["Where were you while we were getting high?", "https://open.spotify.com/track/2V5SGZFsF1yoDakHRwrmma"],
	["Oh, oh, do you shiver?", "https://open.spotify.com/track/4LdwkMY8upUvA99H0oo3LM"],
	["I've been to London, I've been to Paris, Even way out there to Tokyo", "https://open.spotify.com/track/59dLtGBS26x7kc0rHbaPrq"],
	//15
	["I really like it here in your arms", "https://open.spotify.com/track/6dquCx5KAW5jCgGgoTlghL"],
	["But baby, watch me freak out", "https://open.spotify.com/track/1CmlXPPNDBi7gjx3N2BhGP"],
	["'어디긴 니 마음이지'라는 본심을", "https://open.spotify.com/track/5tEouf2s1SPwAIkOHnvWtQ"],
	["I can't get you off my mind", "https://open.spotify.com/track/1qEmFfgcLObUfQm0j1W2CK"],

]
song = songs[Math.floor(Math.random() * songs.length)]
link = song[1]
name = song[0]

if (name == "cat flag"){
	name = "🚩 " + "cat flag" + " 🚩"
}
else{
	name = "♪ " + name + " ♫"
}

p.innerHTML = `<ul><li><a href = "${link}"}>${name}</a></li></ul>`;

document.body.children[0].children[0].children[2].append(p)