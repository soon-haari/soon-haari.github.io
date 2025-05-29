// These songs are exactly in my playlist.
// https://open.spotify.com/playlist/1W9ljrnbLjtPHyOCxuopnf

let songs = [
	["cat flag", "/flag"],
	//0
	["어어어 어어어 푸푸푸 또 허허허 우우우적 거거거 리더던 시 저저절 나라면", "https://open.spotify.com/track/1IJxbEXfgiKuRx6oXMX87e"],
	["Wonder why connected feelings lingering to us", "https://open.spotify.com/track/3cqZ2Vsxt5Gi5ZLx3h5DAW"],
	["Went low, went high, what matters is now", "https://open.spotify.com/track/1rIKgCH4H52lrvDcz50hS8"],
	["네가 흘린 눈물이 마법의 주문이 되어", "https://open.spotify.com/track/2AAeXb4diy9tyBZqEhgnpE"],
	["Hold on, hold on, fuck that", "https://open.spotify.com/track/5mCPDVBb16L4XQwDdbRUpz"],
	//5
	["But no more syrup left in my bottle, damn", "https://open.spotify.com/track/3BbS4oLUDNOLctshheQucc"],
	["Cause every closed door is just a intro of a brand new story", "https://open.spotify.com/track/4Xnuh5fEHbHpxI3leaQSLN"],
	["Now the crowd's gone, is it over? Is it over?", "https://open.spotify.com/track/0i2THDeAhJma8FrUVy90No"],
	["Why did I ever go back?", "https://open.spotify.com/track/6q6nBdrtcJubvRog3tRhEn"],
	["Everyone else has left now", "https://open.spotify.com/track/23TPP1eeElFfvYVznskwCY"],
	//10
	["Oh, oh, do you shiver?", "https://open.spotify.com/track/4LdwkMY8upUvA99H0oo3LM"],
	["I've been to London, I've been to Paris, Even way out there to Tokyo", "https://open.spotify.com/track/59dLtGBS26x7kc0rHbaPrq"],
	["그게 나쁘던 좋던 말야", "https://open.spotify.com/track/6ZY5lLjDmK6Bzon5vseYLn"],
	["지겨워 난 누가? 네가 짖고 있나 으으음", "https://open.spotify.com/track/7EXHK5NtyxsOeVGBA42peN"],
	["Now payback is a bad bitch", "https://open.spotify.com/track/0yvPEnB032fojLfVluFjUv"],
	//15
	["너를 처음 바라본 순간 나도 모르게", "https://open.spotify.com/track/0ziY7wJn4xAdWdEaI6tVds"],
	["Put your wings on me, wings on me", "https://open.spotify.com/track/3RiPr603aXAoi4GHyXx0uy"],
	["When all you do is walk the other way?", "https://open.spotify.com/track/3Fj47GNK2kUF0uaEDgXLaD"],
	["잠깐이면 돼 잠깐이면", "https://open.spotify.com/track/1Z8I2cvV9JQZqB1YA0O3PY"],
	["You don't make it easy, no no", "https://open.spotify.com/track/3KkXRkHbMCARz0aVfEt68P"],
	//20
	["'Cause I felt sad love, I felt bad love, Sometimes happy love, Turns into givin' up", "https://open.spotify.com/track/7aGyRfJWtLqgJaZoG9lJhE"],
	["When all inside you burn like a star", "https://open.spotify.com/track/0LD0K8ZYI0KNe4qN43Ofr7"],
	["Think I lost my mind, reality's so hard to find", "https://open.spotify.com/track/2hwOoMtWPtTSSn6WHV7Vp5"],
	["Maybe I struggled by the lines and I don't need to say", "https://open.spotify.com/track/4Wi5YeExODRDiBlaIEGl7c"],
	["And you showin' off, but it's alright", "https://open.spotify.com/track/3CA9pLiwRIGtUBiMjbZmRw"],
	//25
	["I love myself", "https://open.spotify.com/track/3ODXRUPL44f04cCacwiCLC"],
	["And the world don't respect you, And the culture don't accept you, But you think it's all love", "https://open.spotify.com/track/7A3ETuOrieyx9KSF0sn0Zf"],
	["I call a bitch a bitch, a ho a ho, a woman a woman", "https://open.spotify.com/track/6cfVDaIdvDtYH91RqC6Wox"],
	["You can have all my shine, I'll give you the light", "https://open.spotify.com/track/2Fw5S2gaOSZzdN5dFoC2dj"],
	["A fool if I take it all for granted", "https://open.spotify.com/track/0NfYAsKygCYwPA2BgTZ1qg"],
	//30
	["Anyway they don't know you like I do", "https://open.spotify.com/track/0w2piYWj1F2bzUftzGJgK9"],
	["Ain't nobody prayin' for me", "https://open.spotify.com/track/2LTlO3NuNVN70lp2ZbVswF"],
	["Sometimes I wake up", "https://open.spotify.com/track/0jFxJ1US3dmwQn7DJi269h"],
	["Fuck you, goodnight, thank you much for your service", "https://open.spotify.com/track/4oFtLSgHyZPNYDCcANhTnO"],
	["What did the Asian say?", "https://open.spotify.com/track/2fgAJeky8Cpsr8Cfvn2NVp"],
	//35
]

let song = songs[Math.floor(Math.random() * songs.length)]
let link = song[1]
let name_ = song[0]

if (name_ == "cat flag"){
	name_ = "🚩 " + name_ + " 🚩"
}
else{
	name_ = "♪ " + name_ + " ♫"
}

const p = document.createElement("to_append");
p.innerHTML = `<ul><li><a href = "${link}" target="_blank">${name_}</a></li></ul>`;

// document.body.children[0].children[0].children[1].append(p)
lis = document.getElementsByTagName('li')
lis[lis.length - 1].append(p)
