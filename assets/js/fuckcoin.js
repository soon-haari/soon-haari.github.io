const serverUrl = "https://api.upbit.com";
const amount = 4184924293.29221859;
async function fetchData() {
    try {
      const response = await fetch(`${serverUrl}/v1/ticker?markets=KRW-BTT`);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      const price = data[0]["trade_price"];
      const money = price * amount;
      return money; // Save the data in a variable and return it
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error);
    }
  }

function myfunc(){
    fetchData().then(money => {
        money = Math.round(money);
        const moneyDisplay = document.getElementById("money-display");
        moneyDisplay.textContent = `${money}₩`;
    }).catch(error => {
        console.error('Error fetching money:', error);
    });   
}

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

while(true){
    myfunc();
    await wait(1000);
}
