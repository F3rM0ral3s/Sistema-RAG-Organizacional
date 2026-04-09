# Gaceta UNAM — Sistema RAG Organizacional

RAG sobre documentos de la Gaceta UNAM con query expansion, retrieval híbrido (denso + sparse) con Reciprocal Rank Fusion y detección de jailbreak vía LLM-as-judge.

## Arquitectura

```
Frontend (HTML/JS) → NGINX :80 → FastAPI :8000 → Qdrant :6333
                                       ↓
                          llama-server light :8080  (guard + expander)
                          llama-server gen   :8082  (generación RAG)
```

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| NGINX | `:80` | Proxy reversa, archivos estáticos |
| FastAPI | `:8000` | API backend, pipeline RAG |
| llama-server light | `:8080` | Qwen3.5-4B Q8_0, 16 slots × 2K ctx (guard + expander) |
| llama-server gen | `:8082` | Qwen3.5-4B Q8_0, 2 slots × 48K ctx (generación) |
| Qdrant | `:6333` | Colección `rag_documents` (vectores densos + sparse) |

## Pipeline RAG

1. **Guard (LLM-as-judge)** — `llama-light` clasifica si la query es jailbreak o no
2. **Rate limiter** — Una consulta por usuario a la vez (HTTP 429)
3. **Expansión de consultas** — 3 reformulaciones vía `llama-light`
4. **Embedding** — `BAAI/bge-m3` (fp16, CUDA), denso 1024-d + sparse léxico
5. **Retrieval híbrido + RRF** — Búsqueda denso + sparse en Qdrant para cada query, fusión con RRF (k=60), top 30 chunks
6. **Generación** — `llama-gen` produce la respuesta sobre los fragmentos recuperados

### Flujo de eventos

```
POST /api/query  → { query_id, status: "processing" | "rejected" }
GET  /api/query/{id} → polling → "processing" | "processed" | "rejected"
```

## Instalación

```bash
python3 -m venv ../venv --system-site-packages
source ../venv/bin/activate
pip install -r requirements.txt
```

El modelo GGUF se descarga automáticamente al iniciar `start_llama.sh` si no existe en `../models/`.

## Uso

```bash
# llama-servers (light + generator)
bash start_llama.sh

# Backend
source ../venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# NGINX
nginx -c nginx/nginx.conf

# Exponer con ngrok
python tunnel.py <NGROK_AUTHTOKEN>
```

## API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/query` | POST | Enviar consulta `{ "query": "...", "user_id": "..." }` |
| `/api/query/{id}` | GET | Obtener estado/resultado |
| `/api/health` | GET | Health check |

## Estructura

```
app/
├── backend/
│   ├── main.py              # FastAPI, endpoints, pipeline
│   ├── config.py            # Configuración centralizada
│   ├── models.py            # Esquemas Pydantic
│   └── rag/
│       ├── embedder.py      # BGE-M3 (denso + sparse)
│       ├── retriever.py     # Qdrant híbrido + RRF
│       ├── expander.py      # Expansión de consultas (llama-light)
│       ├── generator.py     # Generación de respuestas (llama-gen)
│       └── guard.py         # Detección de jailbreak (LLM-as-judge)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── nginx/
│   └── nginx.conf
├── start_llama.sh
└── tunnel.py
```
