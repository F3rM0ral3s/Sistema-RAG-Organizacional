/**
 * Gaceta UNAM RAG Frontend
 *
 * Event-based polling: submit → poll every 1s → display result.
 * Rate limiting is handled server-side; the frontend simply disables
 * the submit button while a query is in-flight.
 */

const API_BASE = "/api";
const POLL_INTERVAL_MS = 1000;

// Generate a persistent user_id per browser session
const USER_ID = sessionStorage.getItem("user_id") || (() => {
    const id = crypto.randomUUID();
    sessionStorage.setItem("user_id", id);
    return id;
})();

const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const btnText = submitBtn.querySelector(".btn-text");
const btnLoading = submitBtn.querySelector(".btn-loading");
const statusBar = document.getElementById("status-bar");
const statusText = document.getElementById("status-text");
const resultContainer = document.getElementById("result-container");
const answerContent = document.getElementById("answer-content");
const sourcesList = document.getElementById("sources-list");
const expandedSection = document.getElementById("expanded-section");
const expandedList = document.getElementById("expanded-list");
const errorContainer = document.getElementById("error-container");
const errorMessage = document.getElementById("error-message");

let pollTimer = null;

/** Convert LLM text output to readable HTML */
function renderAnswer(text) {
    // Escape HTML
    let html = text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // Bold: **text**
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic: *text*
    html = html.replace(/(?<!\w)\*(.+?)\*(?!\w)/g, "<em>$1</em>");

    // Split into lines for block-level processing
    const lines = html.split("\n");
    const result = [];
    let inList = false;

    for (const line of lines) {
        const listMatch = line.match(/^\s*[\*\-]\s+(.*)/);
        if (listMatch) {
            if (!inList) { result.push("<ul>"); inList = true; }
            result.push(`<li>${listMatch[1]}</li>`);
        } else {
            if (inList) { result.push("</ul>"); inList = false; }
            const trimmed = line.trim();
            if (trimmed === "") {
                result.push("<br>");
            } else {
                result.push(`<p>${trimmed}</p>`);
            }
        }
    }
    if (inList) result.push("</ul>");

    return result.join("");
}

/** Extract issue number from source_pdf path */
function extractIssueRef(src) {
    if (src.source_pdf) {
        const m = src.source_pdf.match(/issue_(\d+)/);
        if (m) return `Gaceta No. ${m[1]}`;
    }
    if (src.doc_id) return src.doc_id;
    return null;
}

function setLoading(loading) {
    submitBtn.disabled = loading;
    input.disabled = loading;
    btnText.classList.toggle("hidden", loading);
    btnLoading.classList.toggle("hidden", !loading);
    statusBar.classList.toggle("hidden", !loading);
}

function hideAll() {
    resultContainer.classList.add("hidden");
    errorContainer.classList.add("hidden");
    statusBar.classList.add("hidden");
}

function showError(msg) {
    hideAll();
    errorMessage.textContent = msg;
    errorContainer.classList.remove("hidden");
    setLoading(false);
}

function showResult(data) {
    hideAll();
    setLoading(false);

    if (data.status === "rejected") {
        showError(data.rejection_reason || "Consulta rechazada.");
        return;
    }

    // Answer
    answerContent.innerHTML = renderAnswer(data.answer || "Sin respuesta.");

    // Sources
    sourcesList.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
        data.sources.forEach((src, i) => {
            const card = document.createElement("div");
            card.className = "source-card";

            const meta = document.createElement("div");
            meta.className = "source-meta";
            const issueRef = extractIssueRef(src);
            if (issueRef) meta.innerHTML += `<span>${issueRef}</span>`;
            if (src.issue_date) meta.innerHTML += `<span>${src.issue_date}</span>`;
            if (src.chunk_index) meta.innerHTML += `<span>Fragmento ${src.chunk_index}</span>`;
            meta.innerHTML += `<span>Relevancia: ${(src.score * 100).toFixed(1)}%</span>`;

            const text = document.createElement("div");
            text.className = "source-text";
            text.textContent = src.text;

            card.appendChild(meta);
            card.appendChild(text);
            sourcesList.appendChild(card);
        });
    }

    // Expanded queries
    expandedList.innerHTML = "";
    if (data.expanded_queries && data.expanded_queries.length > 0) {
        expandedSection.classList.remove("hidden");
        data.expanded_queries.forEach(q => {
            const li = document.createElement("li");
            li.textContent = q;
            expandedList.appendChild(li);
        });
    } else {
        expandedSection.classList.add("hidden");
    }

    resultContainer.classList.remove("hidden");
}

async function pollResult(queryId) {
    try {
        const resp = await fetch(`${API_BASE}/query/${queryId}`);
        if (!resp.ok) {
            showError(`Error al consultar estado: ${resp.status}`);
            return;
        }
        const data = await resp.json();

        if (data.status === "processing") {
            // Keep polling
            pollTimer = setTimeout(() => pollResult(queryId), POLL_INTERVAL_MS);
            return;
        }

        // processed or rejected
        showResult(data);
    } catch (err) {
        showError(`Error de conexión: ${err.message}`);
    }
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    hideAll();
    setLoading(true);
    if (pollTimer) clearTimeout(pollTimer);

    try {
        const resp = await fetch(`${API_BASE}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, user_id: USER_ID }),
        });

        if (resp.status === 429) {
            showError("Ya tienes una consulta en proceso. Espera a que termine.");
            return;
        }

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            showError(err.detail || `Error: ${resp.status}`);
            return;
        }

        const data = await resp.json();

        if (data.status === "rejected") {
            showResult(data);
            // Need to fetch full result for rejection_reason
            const full = await fetch(`${API_BASE}/query/${data.query_id}`);
            if (full.ok) showResult(await full.json());
            return;
        }

        statusText.textContent = "Procesando consulta...";
        pollTimer = setTimeout(() => pollResult(data.query_id), POLL_INTERVAL_MS);
    } catch (err) {
        showError(`Error de conexión: ${err.message}`);
    }
});

// Allow Ctrl+Enter to submit
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        form.dispatchEvent(new Event("submit"));
    }
});
