const predictionElement = document.getElementById("prediction");

async function fetchPrediction() {
    try {
        const response = await fetch("/get_prediction");
        if (response.ok) {
            const data = await response.json();
            predictionElement.textContent = data.prediction || "No action detected";
        } else {
            predictionElement.textContent = "Error fetching prediction";
        }
    } catch (error) {
        console.error("Error fetching prediction:", error);
        predictionElement.textContent = "Error fetching prediction";
    }
}

setInterval(fetchPrediction, 500);
