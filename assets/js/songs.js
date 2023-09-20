const p = document.createElement("to_append");

let songs = [
	["어어어 어어어 푸푸푸 또 허허허 우우우적 거거거 리더던 시 저저절 나라면", "https://open.spotify.com/track/1IJxbEXfgiKuRx6oXMX87e"],
	["Wonder why connected feelings lingering to us", "https://open.spotify.com/track/3cqZ2Vsxt5Gi5ZLx3h5DAW"],
	["Went low, went high, what matters is now", "https://open.spotify.com/track/1rIKgCH4H52lrvDcz50hS8"],
	["나 웃을게요 많이 그대를 위해 많이", "https://open.spotify.com/track/3eOXiUUfz9OhyVqoREsJYe"],
	["네가 흘린 눈물이 마법의 주문이 되어", "https://open.spotify.com/track/2AAeXb4diy9tyBZqEhgnpE"],
	["하지만 내 노래는 누굴 위한 걸까", "https://open.spotify.com/track/1nlsjS6O7emQKryh2Azatg"],
	["Hold on, hold on, fuck that", "https://open.spotify.com/track/5mCPDVBb16L4XQwDdbRUpz"],
	["But no more syrup left in my bottle, damn", "https://open.spotify.com/track/3BbS4oLUDNOLctshheQucc"],
]
song = songs[Math.floor(Math.random() * songs.length)]
link = song[1]
name = song[0]

p.innerHTML = `<ul><li><a href = "${link}"}>${name}</a></li></ul>`;

document.body.children[0].children[0].children[2].append(p)