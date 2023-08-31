const p = document.createElement("p");
p.style.textAlign = "center";
p.style.fontSize = "13pt";

var gift_year = 2023;
var gift_month = 8;
var gift_day = 26;

var now = new Date();
var year = now.getFullYear();
var month = now.getMonth() + 1;
var day = now.getDate();

var gift = new Date(gift_year, gift_month - 1, gift_day);
var today = new Date(year, month - 1, day);

diff_time = today.getTime() - gift.getTime()
diff_date = diff_time / (1000 * 60 * 60 * 24)

p.innerHTML = `08/26<br><br>💚💚<br><br>D + ${diff_date}`;
//p.innerHTML = `${today} ${gift}`
document.body.append(p);

