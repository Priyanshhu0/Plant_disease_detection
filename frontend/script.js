const SERVER_URL = "http://127.0.0.1:8000";

const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const serverStatusEl = document.getElementById("serverStatus");

const resultCard = document.getElementById("resultCard");
const resultLabel = document.getElementById("resultLabel");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");
const resultNote = document.getElementById("resultNote");

let selectedFile = null;

/* ---------- Check backend is alive ---------- */
async function checkServer() {
    try {
        const res = await fetch(SERVER_URL + "/");
        if (!res.ok) throw new Error("not ok");
        serverStatusEl.textContent = "Server connected";
        serverStatusEl.classList.add("status-good");
    } catch (err) {
        serverStatusEl.textContent = "Backend offline.";
        serverStatusEl.classList.add("status-bad");
    }
}
checkServer();

uploadBox.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = "block";
        uploadPrompt.style.display = "none";
    };
    reader.readAsDataURL(file);

    predictBtn.disabled = false;
    resultCard.style.display = "none";
});

/* ---------- Send image to backend for prediction ---------- */
predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = "Analyzing...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(SERVER_URL + "/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error("Prediction failed");

        const data = await response.json();
        showResult(data);
    } catch (err) {
        resultCard.style.display = "block";
        resultLabel.textContent = "Error";
        confidenceBar.style.width = "0%";
        confidenceValue.textContent = "0%";
        resultNote.textContent = "Could not reach the server. Make sure the backend is running.";
    }

    predictBtn.disabled = false;
    predictBtn.textContent = "Analyze Leaf";
});

/* ---------- Display the prediction result ---------- */
function showResult(data) {
    resultCard.style.display = "block";

    if (data.error) {
        resultLabel.textContent = "Not Available";
        confidenceBar.style.width = "0%";
        confidenceValue.textContent = "0%";
        resultNote.textContent = data.error;
        return;
    }

    resultLabel.textContent = data.prediction;
    confidenceBar.style.width = data.confidence + "%";
    confidenceValue.textContent = data.confidence.toFixed(1) + "%";
    resultNote.textContent = "This is a prediction from a trained model and is not a substitute for expert agricultural advice.";
}