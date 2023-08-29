const p = document.createElement("p");
p.style.textAlign = "center";
p.style.fontSize = "13pt";

var gift = new Date(2023, 8, 26);
var now = new Date();

var year = now.getFullYear();
var month = now.getMonth() + 1;
var day = now.getDate();

var today = new Date(year, month, day);

diff_time = today.getTime() - gift.getTime()
diff_date = diff_time / (1000 * 60 * 60 * 24)

p.innerHTML = `08/26<br><br>💚💚<br><br>D + ${diff_date}`;
document.body.append(p);

