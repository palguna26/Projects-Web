//calculator
const display = document.getElementById("display");

function appendToDisplay(input){
    display.value += input;

}

function clearDisplay(){
    display.value = "";

}

function calculate(){
    try{
        display.value = eval(display.value).toFixed(2);

    }
    catch(error){
        display.value = "symbolError";
    }
}

function backspace(){
    var value = document.getElementById("display").value;
    document.getElementById("display").value = value.substr(0, value.length - 1);
}