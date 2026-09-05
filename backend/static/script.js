/* =========================================================
   ELEMENTS
========================================================= */

const imageForm =
    document.getElementById("imageForm");

const imageFile =
    document.getElementById("imageFile");

const pdfForm =
    document.getElementById("pdfForm");

const pdfFile =
    document.getElementById("pdfFile");

const selectedImageName =
    document.getElementById("selectedImageName");

const selectedPdfName =
    document.getElementById("selectedPdfName");

const imageLoading =
    document.getElementById("imageLoading");

const pdfLoading =
    document.getElementById("pdfLoading");

const imageResult =
    document.getElementById("imageResult");

const pdfResult =
    document.getElementById("pdfResult");

const imageSubmitButton =
    document.getElementById("imageSubmitButton");

const pdfSubmitButton =
    document.getElementById("pdfSubmitButton");

const healthButton =
    document.getElementById("healthButton");

const healthResult =
    document.getElementById("healthResult");


/* =========================================================
   STATE
========================================================= */

let activeReportType = null;

let lastImagePredictionData = null;

let lastPdfPredictionData = null;


/* =========================================================
   CLEAR PREVIOUS REPORT
========================================================= */

function clearPreviousReport() {

    if (imageResult) {

        imageResult.classList.add("hidden");

        imageResult.innerHTML = "";

    }


    if (pdfResult) {

        pdfResult.classList.add("hidden");

        pdfResult.innerHTML = "";

    }


    if (imageLoading) {

        imageLoading.classList.add("hidden");

    }


    if (pdfLoading) {

        pdfLoading.classList.add("hidden");

    }


    lastImagePredictionData = null;

    lastPdfPredictionData = null;

    activeReportType = null;
}


/* =========================================================
   IMAGE FILE DISPLAY
========================================================= */

function displaySelectedImage() {

    if (!selectedImageName || !imageFile) {
        return;
    }


    if (
        imageFile.files &&
        imageFile.files.length > 0
    ) {

        const fileName =
            imageFile.files[0].name;


        selectedImageName.innerHTML = `

            <span class="file-small-icon">
                ✓
            </span>

            <span>
                Selected: ${escapeHtml(fileName)}
            </span>

        `;

    } else {

        selectedImageName.innerHTML = `

            <span class="file-small-icon">
                📁
            </span>

            <span>
                No file selected
            </span>

        `;

    }
}


/* =========================================================
   PDF FILE DISPLAY
========================================================= */

function displaySelectedPdf() {

    if (!selectedPdfName || !pdfFile) {
        return;
    }


    if (
        pdfFile.files &&
        pdfFile.files.length > 0
    ) {

        const fileName =
            pdfFile.files[0].name;


        selectedPdfName.innerHTML = `

            <span class="file-small-icon">
                ✓
            </span>

            <span>
                Selected: ${escapeHtml(fileName)}
            </span>

        `;

    } else {

        selectedPdfName.innerHTML = `

            <span class="file-small-icon">
                📁
            </span>

            <span>
                No file selected
            </span>

        `;

    }
}


/* =========================================================
   FILE EVENTS
========================================================= */

if (imageFile) {

    imageFile.addEventListener(
        "change",
        displaySelectedImage
    );

}


if (pdfFile) {

    pdfFile.addEventListener(
        "change",
        displaySelectedPdf
    );

}


/* =========================================================
   IMAGE ANALYSIS
========================================================= */

if (imageForm) {

    imageForm.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();


            if (
                !imageFile.files ||
                imageFile.files.length === 0
            ) {

                showError(
                    imageResult,
                    "Please select a CT image first."
                );

                return;

            }


            const selectedFile =
                imageFile.files[0];

            const selectedFileName =
                selectedFile.name;


            clearPreviousReport();


            selectedImageName.innerHTML = `

                <span class="file-small-icon">
                    ✓
                </span>

                <span>
                    Selected: ${escapeHtml(selectedFileName)}
                </span>

            `;


            const formData =
                new FormData();


            formData.append(
                "image",
                selectedFile
            );


            imageLoading.classList.remove(
                "hidden"
            );


            imageSubmitButton.disabled =
                true;


            imageSubmitButton.innerHTML = `

                <span class="button-icon">
                    ⏳
                </span>

                <span>
                    Analyzing...
                </span>

            `;


            try {

                const response =
                    await fetch(
                        "/api/predict",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                if (
                    !response.ok ||
                    !data.success
                ) {

                    throw new Error(
                        data.error ||
                        "Image analysis failed."
                    );

                }


                lastImagePredictionData =
                    data;


                activeReportType =
                    "image";


                renderImageResult(data);


            } catch (error) {

                console.error(
                    "Image analysis error:",
                    error
                );


                showError(
                    imageResult,
                    error.message ||
                    "Unable to analyze the image."
                );


            } finally {

                imageLoading.classList.add(
                    "hidden"
                );


                imageSubmitButton.disabled =
                    false;


                imageSubmitButton.innerHTML = `

                    <span class="button-icon">
                        🔍
                    </span>

                    <span>
                        Analyze CT Image
                    </span>

                `;

            }

        }
    );

}


/* =========================================================
   PDF ANALYSIS
========================================================= */

if (pdfForm) {

    pdfForm.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();


            if (
                !pdfFile.files ||
                pdfFile.files.length === 0
            ) {

                showError(
                    pdfResult,
                    "Please select a PDF first."
                );

                return;

            }


            const selectedFile =
                pdfFile.files[0];

            const selectedFileName =
                selectedFile.name;


            clearPreviousReport();


            selectedPdfName.innerHTML = `

                <span class="file-small-icon">
                    ✓
                </span>

                <span>
                    Selected: ${escapeHtml(selectedFileName)}
                </span>

            `;


            const formData =
                new FormData();


            formData.append(
                "pdf",
                selectedFile
            );


            pdfLoading.classList.remove(
                "hidden"
            );


            pdfSubmitButton.disabled =
                true;


            pdfSubmitButton.innerHTML = `

                <span class="button-icon">
                    ⏳
                </span>

                <span>
                    Analyzing...
                </span>

            `;


            try {

                const response =
                    await fetch(
                        "/api/predict-pdf",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                if (
                    !response.ok ||
                    !data.success
                ) {

                    throw new Error(
                        data.error ||
                        "PDF analysis failed."
                    );

                }


                console.log(
                    "PDF analysis response:",
                    data
                );


                lastPdfPredictionData =
                    data;


                activeReportType =
                    "pdf";


                renderPdfResult(data);


            } catch (error) {

                console.error(
                    "PDF analysis error:",
                    error
                );


                showError(
                    pdfResult,
                    error.message ||
                    "Unable to analyze the PDF."
                );


            } finally {

                pdfLoading.classList.add(
                    "hidden"
                );


                pdfSubmitButton.disabled =
                    false;


                pdfSubmitButton.innerHTML = `

                    <span class="button-icon">
                        📊
                    </span>

                    <span>
                        Analyze PDF
                    </span>

                `;

            }

        }
    );

}


/* =========================================================
   IMAGE RESULT
========================================================= */

function renderImageResult(data) {

    const result =
        data.result || {};


    const probability =
        Number(
            result.malignant_probability || 0
        );


    const percentage =
        (
            probability * 100
        ).toFixed(1);


    const nonMalignant =
        (
            Number(
                result.non_malignant_probability || 0
            ) * 100
        ).toFixed(1);


    const predictionText =
        result.prediction ||
        "UNKNOWN";


    let predictionClass =
        "prediction-normal";


    let predictionIcon =
        "✓";


    if (
        predictionText === "MALIGNANT"
    ) {

        predictionClass =
            "prediction-danger";

        predictionIcon =
            "⚠";

    } else if (
        predictionText === "UNCERTAIN"
    ) {

        predictionClass =
            "prediction-warning";

        predictionIcon =
            "!";

    }


    imageResult.innerHTML = `

        <div class="result-card">


            <div class="result-header">

                <div class="result-header-left">

                    <div class="result-icon">
                        🫁
                    </div>


                    <div>

                        <div class="result-title">
                            CT Analysis Complete
                        </div>


                        <div class="result-subtitle">
                            ${escapeHtml(
                                data.filename || ""
                            )}
                        </div>

                    </div>

                </div>


                <div class="result-status-badge">
                    COMPLETED
                </div>

            </div>



            <div class="prediction-box">

                <div class="prediction-label">
                    MODEL PREDICTION
                </div>


                <div
                    class="prediction-value ${predictionClass}"
                >

                    <span class="prediction-status-icon">
                        ${predictionIcon}
                    </span>


                    ${escapeHtml(
                        predictionText
                    )}

                </div>

            </div>



            <div class="probability-section">

                <div class="probability-header">

                    <span>
                        Malignant probability
                    </span>


                    <strong>
                        ${percentage}%
                    </strong>

                </div>


                <div class="probability-bar">

                    <div
                        class="probability-fill"
                        style="width:${percentage}%"
                    ></div>

                </div>

            </div>



            <div class="result-details">


                <div class="detail-row">

                    <span>
                        Non-malignant probability
                    </span>


                    <span class="detail-value">
                        ${nonMalignant}%
                    </span>

                </div>



                <div class="detail-row">

                    <span>
                        Decision threshold
                    </span>


                    <span class="detail-value">
                        ${result.threshold ?? 0.50}
                    </span>

                </div>



                <div class="detail-row">

                    <span>
                        Uncertainty status
                    </span>


                    <span class="detail-value">

                        ${
                            result.uncertainty
                                ? "Detected"
                                : "Not detected"
                        }

                    </span>

                </div>



                <div class="detail-row">

                    <span>
                        Model
                    </span>


                    <span class="detail-value">
                        MobileNetV2 V5
                    </span>

                </div>

            </div>



            <div class="report-actions">


                <button
                    type="button"
                    class="download-report-button"
                    id="downloadImageReportButton"
                >

                    <span>
                        ↓
                    </span>

                    <span>
                        Download Report
                    </span>

                </button>



                <button
                    type="button"
                    class="another-report-button"
                    id="anotherImageButton"
                >

                    <span>
                        ＋
                    </span>

                    <span>
                        Generate Another Report
                    </span>

                </button>

            </div>



            <div class="research-note">

                <span>
                    ⓘ
                </span>


                <span>

                    This result is generated by an academic
                    research prototype and should not be used
                    for clinical diagnosis or medical
                    decision-making.

                </span>

            </div>

        </div>

    `;


    imageResult.classList.remove(
        "hidden"
    );


    const downloadButton =
        document.getElementById(
            "downloadImageReportButton"
        );


    const anotherButton =
        document.getElementById(
            "anotherImageButton"
        );


    if (downloadButton) {

        downloadButton.addEventListener(
            "click",
            function () {

                downloadReport(
                    "image",
                    lastImagePredictionData
                );

            }
        );

    }


    if (anotherButton) {

        anotherButton.addEventListener(
            "click",
            function () {

                clearPreviousReport();


                if (imageFile) {

                    imageFile.value = "";

                }


                if (selectedImageName) {

                    selectedImageName.innerHTML = `

                        <span class="file-small-icon">
                            📁
                        </span>

                        <span>
                            No file selected
                        </span>

                    `;

                }

            }
        );

    }

}


/* =========================================================
   PDF RESULT
========================================================= */

function renderPdfResult(data) {

    const pages =
        Array.isArray(data.pages)
            ? data.pages
            : [];


    const pageCount =
        Number(
            data.pages_processed ??
            pages.length ??
            0
        );


    const imageResults =
        Array.isArray(data.image_results)
            ? data.image_results
            : [];


    const textResults =
        Array.isArray(data.text_results)
            ? data.text_results
            : [];


    const summary =
        data.summary || {};


    const imageSummary =
        summary.image || {};


    const textSummary =
        summary.text || {};


    const imageProbability =
        typeof imageSummary.average_malignant_probability === "number"
            ? imageSummary.average_malignant_probability
            : null;


    const imagePercentage =
        imageProbability !== null
            ? (
                imageProbability * 100
            ).toFixed(1)
            : null;


    const textProbability =
        typeof textSummary.average_probability === "number"
            ? textSummary.average_probability
            : null;


    const textPercentage =
        textProbability !== null
            ? (
                textProbability * 100
            ).toFixed(1)
            : null;


    const documentType =
        data.document_type ||
        "unknown";


    let finalPrediction =
        "ANALYSIS COMPLETE";


    let predictionClass =
        "prediction-normal";


    let predictionIcon =
        "✓";


    if (
        documentType === "image" &&
        imageSummary.prediction
    ) {

        finalPrediction =
            imageSummary.prediction;

    } else if (
        documentType === "text" &&
        textSummary.prediction
    ) {

        finalPrediction =
            textSummary.prediction;

    } else if (
        documentType === "mixed"
    ) {

        finalPrediction =
            "MIXED ANALYSIS";

    }


    if (
        finalPrediction === "MALIGNANT"
    ) {

        predictionClass =
            "prediction-danger";

        predictionIcon =
            "⚠";

    } else if (
        finalPrediction === "UNCERTAIN"
    ) {

        predictionClass =
            "prediction-warning";

        predictionIcon =
            "!";

    }


    let analysisCards = "";


    /* =====================================================
       IMAGE ANALYSIS CARD
    ====================================================== */

    if (
        imageProbability !== null
    ) {

        analysisCards += `

            <div class="pdf-analysis-item">


                <div class="pdf-analysis-item-header">

                    <div class="pdf-analysis-item-icon">
                        🫁
                    </div>


                    <div>

                        <div class="pdf-analysis-item-title">
                            Image Analysis
                        </div>


                        <div class="pdf-analysis-item-subtitle">

                            ${
                                imageSummary.images_processed ??
                                imageResults.length ??
                                0
                            }
                            image(s) processed

                        </div>

                    </div>

                </div>



                <div class="pdf-analysis-prediction">

                    <span>
                        Prediction
                    </span>


                    <strong
                        class="${
                            imageSummary.prediction === "MALIGNANT"
                                ? "text-danger"
                                : ""
                        }"
                    >

                        ${
                            escapeHtml(
                                imageSummary.prediction ||
                                "N/A"
                            )
                        }

                    </strong>

                </div>



                <div class="probability-section compact">

                    <div class="probability-header">

                        <span>
                            Average malignant probability
                        </span>


                        <strong>
                            ${imagePercentage}%
                        </strong>

                    </div>


                    <div class="probability-bar">

                        <div
                            class="probability-fill"
                            style="width:${imagePercentage}%"
                        ></div>

                    </div>

                </div>

            </div>

        `;

    }


    /* =====================================================
       TEXT ANALYSIS CARD
    ====================================================== */

    if (
        textProbability !== null ||
        textResults.length > 0
    ) {

        analysisCards += `

            <div class="pdf-analysis-item">


                <div class="pdf-analysis-item-header">

                    <div class="pdf-analysis-item-icon">
                        📝
                    </div>


                    <div>

                        <div class="pdf-analysis-item-title">
                            Report Text Analysis
                        </div>


                        <div class="pdf-analysis-item-subtitle">

                            ${
                                textSummary.pages_with_text_prediction ??
                                textResults.length ??
                                0
                            }
                            page(s) analyzed

                        </div>

                    </div>

                </div>



                <div class="pdf-analysis-prediction">

                    <span>
                        Finding
                    </span>


                    <strong>

                        ${
                            escapeHtml(
                                textSummary.prediction ||
                                "TEXT ANALYZED"
                            )
                        }

                    </strong>

                </div>



                ${
                    textPercentage !== null
                        ? `

                            <div class="probability-section compact">

                                <div class="probability-header">

                                    <span>
                                        Lesion probability
                                    </span>


                                    <strong>
                                        ${textPercentage}%
                                    </strong>

                                </div>


                                <div class="probability-bar">

                                    <div
                                        class="probability-fill"
                                        style="width:${textPercentage}%"
                                    ></div>

                                </div>

                            </div>

                        `
                        : ""
                }

            </div>

        `;

    }


    /* =====================================================
       COMPLETE PDF RESULT
    ====================================================== */

    pdfResult.innerHTML = `

        <div class="result-card">


            <div class="result-header">

                <div class="result-header-left">

                    <div class="result-icon">
                        📄
                    </div>


                    <div>

                        <div class="result-title">
                            PDF Analysis Complete
                        </div>


                        <div class="result-subtitle">
                            ${escapeHtml(
                                data.filename || ""
                            )}
                        </div>

                    </div>

                </div>


                <div class="result-status-badge">
                    COMPLETED
                </div>

            </div>



            <div class="prediction-box">

                <div class="prediction-label">
                    DOCUMENT ANALYSIS
                </div>


                <div
                    class="prediction-value ${predictionClass}"
                >

                    <span class="prediction-status-icon">
                        ${predictionIcon}
                    </span>


                    ${escapeHtml(
                        finalPrediction
                    )}

                </div>

            </div>



            <div class="pdf-stat-grid">


                <div class="pdf-stat">

                    <div class="pdf-stat-icon">
                        📑
                    </div>


                    <div>

                        <div class="pdf-stat-label">
                            DOCUMENT TYPE
                        </div>


                        <div class="pdf-stat-value">
                            ${escapeHtml(
                                formatDocumentType(
                                    documentType
                                )
                            )}
                        </div>

                    </div>

                </div>



                <div class="pdf-stat">

                    <div class="pdf-stat-icon">
                        📄
                    </div>


                    <div>

                        <div class="pdf-stat-label">
                            TOTAL PAGES
                        </div>


                        <div class="pdf-stat-value">
                            ${pageCount}
                        </div>

                    </div>

                </div>



                <div class="pdf-stat">

                    <div class="pdf-stat-icon">
                        🫁
                    </div>


                    <div>

                        <div class="pdf-stat-label">
                            IMAGE PAGES
                        </div>


                        <div class="pdf-stat-value">
                            ${
                                data.image_pages ??
                                0
                            }
                        </div>

                    </div>

                </div>



                <div class="pdf-stat">

                    <div class="pdf-stat-icon">
                        📝
                    </div>


                    <div>

                        <div class="pdf-stat-label">
                            TEXT PAGES
                        </div>


                        <div class="pdf-stat-value">
                            ${
                                data.text_pages ??
                                0
                            }
                        </div>

                    </div>

                </div>

            </div>



            ${
                analysisCards
                    ? `

                        <div class="pdf-analysis-section">

                            <div class="pdf-section-title">
                                ANALYSIS BREAKDOWN
                            </div>


                            <div class="pdf-analysis-grid">

                                ${analysisCards}

                            </div>

                        </div>

                    `
                    : ""
            }



            <div class="result-details">


                ${
                    imageProbability !== null
                        ? `

                            <div class="detail-row">

                                <span>
                                    Image threshold
                                </span>


                                <span class="detail-value">
                                    ${
                                        imageSummary.threshold ??
                                        0.50
                                    }
                                </span>

                            </div>

                        `
                        : ""
                }



                ${
                    textProbability !== null
                        ? `

                            <div class="detail-row">

                                <span>
                                    Text threshold
                                </span>


                                <span class="detail-value">
                                    ${
                                        textSummary.threshold ??
                                        0.63
                                    }
                                </span>

                            </div>

                        `
                        : ""
                }



                ${
                    imageProbability !== null
                        ? `

                            <div class="detail-row">

                                <span>
                                    Image model
                                </span>


                                <span class="detail-value">
                                    MobileNetV2 V5
                                </span>

                            </div>

                        `
                        : ""
                }



                ${
                    textProbability !== null
                        ? `

                            <div class="detail-row">

                                <span>
                                    Text model
                                </span>


                                <span class="detail-value">
                                    TF-IDF + Logistic Regression
                                </span>

                            </div>

                        `
                        : ""
                }

            </div>



            <div class="report-actions">


                <button
                    type="button"
                    class="download-report-button"
                    id="downloadPdfReportButton"
                >

                    <span>
                        ↓
                    </span>


                    <span>
                        Download Report
                    </span>

                </button>



                <button
                    type="button"
                    class="another-report-button"
                    id="anotherPdfButton"
                >

                    <span>
                        ＋
                    </span>


                    <span>
                        Generate Another Report
                    </span>

                </button>

            </div>



            <div class="research-note">

                <span>
                    ⓘ
                </span>


                <span>

                    This result is generated by an academic
                    research prototype and should not be used
                    for clinical diagnosis or medical
                    decision-making.

                </span>

            </div>

        </div>

    `;


    pdfResult.classList.remove(
        "hidden"
    );


    /* =====================================================
       PDF BUTTONS
    ====================================================== */

    const downloadButton =
        document.getElementById(
            "downloadPdfReportButton"
        );


    const anotherButton =
        document.getElementById(
            "anotherPdfButton"
        );


    if (downloadButton) {

        downloadButton.addEventListener(
            "click",
            function () {

                downloadReport(
                    "pdf",
                    lastPdfPredictionData
                );

            }
        );

    }


    if (anotherButton) {

        anotherButton.addEventListener(
            "click",
            function () {

                clearPreviousReport();


                if (pdfFile) {

                    pdfFile.value = "";

                }


                if (selectedPdfName) {

                    selectedPdfName.innerHTML = `

                        <span class="file-small-icon">
                            📁
                        </span>

                        <span>
                            No file selected
                        </span>

                    `;

                }

            }
        );

    }

}


/* =========================================================
   DOWNLOAD REPORT
========================================================= */

async function downloadReport(
    reportType,
    predictionData
) {

    if (!predictionData) {

        alert(
            "No analysis report is available."
        );

        return;

    }


    let endpoint =
        "/api/generate-image-report";


    let filename =
        "lung_cancer_image_analysis_report.pdf";


    if (
        reportType === "pdf"
    ) {

        endpoint =
            "/api/generate-pdf-report";


        filename =
            "lung_cancer_pdf_analysis_report.pdf";

    }


    try {

        const response =
            await fetch(
                endpoint,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            predictionData
                        )
                }
            );


        if (!response.ok) {

            let errorMessage =
                "Unable to generate report.";


            try {

                const errorData =
                    await response.json();


                errorMessage =
                    errorData.error ||
                    errorMessage;

            } catch (error) {

                console.error(error);

            }


            throw new Error(
                errorMessage
            );

        }


        const blob =
            await response.blob();


        const url =
            window.URL.createObjectURL(
                blob
            );


        const link =
            document.createElement(
                "a"
            );


        link.href =
            url;


        link.download =
            filename;


        document.body.appendChild(
            link
        );


        link.click();


        link.remove();


        window.URL.revokeObjectURL(
            url
        );


    } catch (error) {

        console.error(
            "Report download error:",
            error
        );


        alert(
            error.message ||
            "Unable to download the report."
        );

    }

}


/* =========================================================
   HEALTH CHECK
========================================================= */

if (healthButton) {

    healthButton.addEventListener(
        "click",
        async function () {

            healthButton.disabled =
                true;


            healthButton.textContent =
                "Checking...";


            try {

                const response =
                    await fetch(
                        "/api/health"
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        "Backend health check failed."
                    );

                }


                healthResult.innerHTML = `

                    <strong>
                        ✓ Backend is running
                    </strong>

                    <br>

                    Image Model:
                    ${escapeHtml(
                        data.image_model || "N/A"
                    )}

                    <br>

                    Image Threshold:
                    ${data.image_threshold ?? "N/A"}

                    <br>

                    Text Model:
                    ${escapeHtml(
                        data.text_model || "N/A"
                    )}

                    <br>

                    Text Threshold:
                    ${data.text_threshold ?? "N/A"}

                    <br>

                    Model Loaded:
                    ${
                        data.model_loaded
                            ? "Yes"
                            : "No"
                    }

                `;


                healthResult.classList.remove(
                    "hidden"
                );


            } catch (error) {

                console.error(error);


                healthResult.innerHTML = `

                    <strong>
                        ✕ Backend unavailable
                    </strong>

                    <br>

                    ${escapeHtml(
                        error.message ||
                        "Unable to connect to Flask."
                    )}

                `;


                healthResult.classList.remove(
                    "hidden"
                );


            } finally {

                healthButton.disabled =
                    false;


                healthButton.textContent =
                    "Check Status";

            }

        }
    );

}


/* =========================================================
   ERROR
========================================================= */

function showError(
    element,
    message
) {

    if (!element) {
        return;
    }


    element.innerHTML = `

        <div
            class="result-card"
            style="
                border-color:#fecaca;
                background:#fffafa;
            "
        >

            <div class="prediction-box">

                <div class="prediction-label">
                    ERROR
                </div>


                <div
                    class="prediction-value"
                    style="color:#dc2626;"
                >
                    Analysis Failed
                </div>


                <div
                    style="
                        margin-top:8px;
                        color:#7f1d1d;
                        font-size:10px;
                        line-height:1.5;
                    "
                >

                    ${escapeHtml(message)}

                </div>

            </div>

        </div>

    `;


    element.classList.remove(
        "hidden"
    );
}


/* =========================================================
   DOCUMENT TYPE FORMATTER
========================================================= */

function formatDocumentType(type) {

    if (!type) {
        return "Unknown";
    }


    return String(type)

        .replace(
            /_/g,
            " "
        )

        .replace(
            /\b\w/g,
            function (letter) {
                return letter.toUpperCase();
            }
        );
}


/* =========================================================
   HTML ESCAPE
========================================================= */

function escapeHtml(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "";

    }


    return String(value)

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );

}