// Function to fetch data from a JSON file
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch file. HTTP Status: ${response.status}`);
        }
        console.log(`Successfully fetched file from ${url}.`);
        return await response.json();
    } catch (error) {
        console.error(`Error fetching file from ${url}:`, error);
        return null;
    }
}

// Function to fetch and unzip a JSON file
async function fetchAndUnzipJSON(url, filename) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ZIP file. HTTP Status: ${response.status}`);
        }
        console.log(`Successfully fetched ZIP file from ${url}.`);

        const data = await response.arrayBuffer();
        const zip = await JSZip.loadAsync(data);
        console.log("Successfully loaded ZIP file.");

        const file = zip.file(filename);
        if (!file) {
            throw new Error(`${filename} not found in the ZIP archive.`);
        }

        const content = await file.async('string');
        return JSON.parse(content);
    } catch (error) {
        console.error(`Error fetching and unzipping JSON file from ${url}:`, error);
        throw error;
    }
}

// Function to display the first objects from the JSON files
async function displayFirstObjects() {
    const pathReports = await fetchAndUnzipJSON('tcgaPathReports.json.zip', 'tcgaPathReports.json');
    const slideEmbeddings = await fetchAndUnzipJSON('tcgaSlideEmbeddings.json.zip', 'tcgaSlideEmbeddings.json');

    document.getElementById('pathReports').textContent = JSON.stringify(pathReports[0], null, 2);
    document.getElementById('slideEmbeddings').textContent = JSON.stringify(slideEmbeddings[0], null, 2);

    // Find nearest neighbors
    const neighbors = await findNearestNeighbors(pathReports, slideEmbeddings);
    document.getElementById('neighbors').textContent = JSON.stringify(neighbors, null, 2);
}

// Function to find nearest neighbors using @tensorflow-models/knn-classifier
async function findNearestNeighbors(pathReports, slideEmbeddings) {
    const knnClassifier = window.knnClassifier.create();

    // Add embeddings to the KNN classifier
    slideEmbeddings.forEach((embedding, i) => {
        //console.log(`Embedding ${i}:`, embedding);
        if (embedding && embedding.embedding) { // Corrected property access
            console.log(`Adding embedding ${i}:`, embedding.embedding);
            knnClassifier.addExample(tf.tensor(embedding.embedding), i);
        } else {
            console.error(`Embedding ${i} is missing or malformed:`, embedding);
        }
    });

    // Find nearest neighbors for each path report
    const neighbors = await Promise.all(pathReports.map(async report => {
        console.log('Processing report:', report);
        if (report && report.embeddings) {
            const embedding = tf.tensor(report.embeddings);
            console.log('Converted embedding to tensor:', embedding);
            const result = await knnClassifier.predictClass(embedding, 5); // Find 5 nearest neighbors
            console.log('Prediction result:', result);
            return result.confidences;
        } else {
            console.error('Report is missing or malformed:', report);
            return null;
        }
    }));

    return neighbors;
}

// Call the function to display the first objects
displayFirstObjects();