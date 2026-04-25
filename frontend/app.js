/**
 * Gaceta UNAM RAG Frontend
 *
 * Event-based polling: submit → poll until terminal status → display result.
 * Rate limiting is handled server-side; the frontend simply disables
 * the submit button while a query is in-flight.
 */

const API_BASE = "/api";
const POLL_INTERVAL_MS = 1000;
const POLL_MAX_ATTEMPTS = 600; // ~10 min at 1 Hz

const STATUS = Object.freeze({
    PROCESSING: "processing",
    PROCESSED: "processed",
    REJECTED: "rejected",
    FAILED: "failed",
});

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
let pollController = null;

function renderAnswer(text) {
    let html = text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/(?<!\w)\*(.+?)\*(?!\w)/g, "<em>$1</em>");

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
            result.push(trimmed === "" ? "<br>" : `<p>${trimmed}</p>`);
        }
    }
    if (inList) result.push("</ul>");
    return result.join("");
}

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

function appendMetaSpan(parent, text) {
    const span = document.createElement("span");
    span.textContent = text;
    parent.appendChild(span);
}

function showResult(data) {
    hideAll();
    setLoading(false);

    if (data.status === STATUS.REJECTED) {
        showError(data.rejection_reason || "Consulta rechazada.");
        return;
    }
    if (data.status === STATUS.FAILED) {
        showError(data.answer || "Error al procesar la consulta.");
        return;
    }

    answerContent.innerHTML = renderAnswer(data.answer || "Sin respuesta.");

    sourcesList.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
        data.sources.forEach((src) => {
            const card = document.createElement("div");
            card.className = "source-card";

            const meta = document.createElement("div");
            meta.className = "source-meta";
            const issueRef = extractIssueRef(src);
            if (issueRef) appendMetaSpan(meta, issueRef);
            if (src.issue_date) appendMetaSpan(meta, src.issue_date);
            if (src.chunk_index) appendMetaSpan(meta, `Fragmento ${src.chunk_index}`);
            appendMetaSpan(meta, `Relevancia: ${(src.score * 100).toFixed(1)}%`);

            const text = document.createElement("div");
            text.className = "source-text";
            text.textContent = src.text;

            card.appendChild(meta);
            card.appendChild(text);
            sourcesList.appendChild(card);
        });
    }

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

function cancelPoll() {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    if (pollController) { pollController.abort(); pollController = null; }
}

async function pollResult(queryId, attempt = 0) {
    if (attempt >= POLL_MAX_ATTEMPTS) {
        showError("La consulta tardó demasiado. Intenta nuevamente.");
        return;
    }
    pollController = new AbortController();
    try {
        const resp = await fetch(`${API_BASE}/query/${queryId}`, {
            signal: pollController.signal,
        });
        if (!resp.ok) {
            showError(`Error al consultar estado: ${resp.status}`);
            return;
        }
        const data = await resp.json();
        if (data.status === STATUS.PROCESSING) {
            pollTimer = setTimeout(() => pollResult(queryId, attempt + 1), POLL_INTERVAL_MS);
            return;
        }
        showResult(data);
    } catch (err) {
        if (err.name === "AbortError") return;
        showError(`Error de conexión: ${err.message}`);
    }
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    hideAll();
    setLoading(true);
    cancelPoll();

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
        statusText.textContent = "Procesando consulta...";
        pollTimer = setTimeout(() => pollResult(data.query_id), POLL_INTERVAL_MS);
    } catch (err) {
        showError(`Error de conexión: ${err.message}`);
    }
});

input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        form.dispatchEvent(new Event("submit"));
    }
});
