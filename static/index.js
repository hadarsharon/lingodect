const textInputElementId = "textInput";
const fileInputElementId = "fileInput";
const loaderElementId = "loaderDiv";
const multiLanguageSwitchElementId = "multiLanguageSwitch"
const speechTranscriptionSwitchElementId = "speechTranscriptionSwitch"

function checkInput() {
    if (document.getElementById(textInputElementId).value) {
        document.getElementById(fileInputElementId).setAttribute("disabled", "true");
        document.getElementById(speechTranscriptionSwitchElementId).setAttribute("disabled", "true");
        document.getElementById(multiLanguageSwitchElementId).removeAttribute("disabled");
    } else {
        document.getElementById(fileInputElementId).removeAttribute("disabled");
        document.getElementById(speechTranscriptionSwitchElementId).removeAttribute("disabled");
        document.getElementById(multiLanguageSwitchElementId).setAttribute("disabled", "true");
    }
    if (document.getElementById(fileInputElementId).value) {
        document.getElementById(textInputElementId).setAttribute("disabled", "true");
        document.getElementById(textInputElementId).placeholder = "Not available while file-based input is selected.";
        document.getElementById(multiLanguageSwitchElementId).setAttribute("disabled", "true");
        document.getElementById(speechTranscriptionSwitchElementId).removeAttribute("disabled");
    } else {
        document.getElementById(textInputElementId).removeAttribute("disabled");
        document.getElementById(textInputElementId).placeholder = "";
        document.getElementById(multiLanguageSwitchElementId).removeAttribute("disabled");
        document.getElementById(speechTranscriptionSwitchElementId).setAttribute("disabled", "true");
    }
}

function clearInput() {
    document.getElementById(textInputElementId).value = null;
    document.getElementById(fileInputElementId).value = null;
    checkInput(); // make sure whatever needs to be re-enabled is done
}

function fakeText(texts) {
    document.getElementById(textInputElementId).value = texts[Math.floor(Math.random() * texts.length)];
}

function showLoader() {
    document.getElementById(loaderElementId).hidden = false;
}