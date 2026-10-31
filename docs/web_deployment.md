# DORA Web Beta

DORA includes a browser-based beta alongside its desktop application. The web client is a React, TypeScript, and Vite single-page app. Its FastAPI service reuses the local analyzer, ML recommendation, dataset search, and academic research modules.

The web beta currently accepts CSV files up to 10 MB. It does not require sign-in. Uploaded datasets are written to temporary files while an analysis or cleaning request runs, then removed when the request finishes. Do not upload sensitive data to a publicly accessible deployment.

## Run locally

From the repository root, install and start the API:

```powershell
python -m pip install -r requirements-web.txt
python -m uvicorn backend.web_api:app --host 127.0.0.1 --port 8000
```

In a second terminal, install and start the web client:

```powershell
npm ci --prefix web
npm run dev
```

Open the Vite URL shown in the terminal. The client defaults to `http://127.0.0.1:8000` for its API. Set `VITE_API_URL` in `web/.env` if the API uses another address. The API's current CORS middleware allows all origins; `WEB_ORIGINS` is declared in the deployment configuration but is not currently applied by the middleware.

## Deploy

The Render Blueprint in [`../render.yaml`](../render.yaml) is the primary deployment configuration. It creates a Python API service and a static web client, then wires `VITE_API_URL` to the API service URL. Push the repository to a Git provider connected to Render and create a Blueprint from `render.yaml`. Render assigns the public URLs after deployment; no deployed URL is currently recorded in this repository. The API health check is `/api/health`.

The repository also includes a Netlify configuration for hosting the static client. Deploy the `web/` directory as the site root and set `VITE_API_URL` to the separately deployed API URL; this is required for analysis and research and also routes dataset search through FastAPI. A Netlify Function and redirect are included as an optional dataset-search-only proxy, but they do not replace the API service.

## API endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Liveness check. |
| `/api/ml/catalog` | GET | Task labels and method catalog. |
| `/api/datasets/profile` | POST | Dataset profile, preview, and ML task estimate. |
| `/api/datasets/analyze` | POST | Statistics, missing values, outliers, duplicates, correlations, quality checks, and ML estimate. |
| `/api/datasets/clean` | POST | Cleans a CSV and returns it as a download with an audit report header. |
| `/api/datasets/search` | GET | Searches Hugging Face, Kaggle, OpenML, and Zenodo dataset catalogs. |
| `/api/research/search` | POST | Searches academic literature and returns ranking, landscape, trend, and gap analysis. |

The profile, analyze, and clean endpoints accept the raw CSV body and read the original filename from `?filename=`. The cleaning endpoint accepts `strategy=impute` or `strategy=drop_rows` and an optional comma-separated `columns` filter. The cleaned CSV is returned in the response body; the `X-Dora-Report` header contains the base64-encoded cleaning summary.

Dataset analysis reports up to 120 columns and correlation matrices up to 24 columns. Dataset-search results can be filtered with `q`, `sources`, and `limit` query parameters. Research search expects JSON containing a query of 3–300 characters and accepts an optional result limit, dataset fingerprint, and ML result.

## Access and service limits

- No sign-in is required. Anyone who can access the client can submit requests to the API.
- CSV uploads are limited to 10 MB; only `.csv` files are accepted by the analysis and cleaning endpoints.
- One upload-based analysis or cleaning job runs at a time per API instance.
- Profile, analyze, and clean requests share the `PROFILE_REQUESTS_PER_MINUTE` limit (default 10 per caller). Research uses `RESEARCH_REQUESTS_PER_MINUTE` (default 6 per caller). Dataset search currently has no API rate limit.
- Research results are cached in memory for 10 minutes. Rate limits, cache, and analysis concurrency are process-local; multiple API instances do not share them.
- Uploaded temporary files are deleted after each request. The API does not persist uploaded datasets to a database.

## Web and desktop scope

The web beta includes dataset upload, preview, profiling, analysis, basic missing-value cleaning and CSV download, plus dataset discovery and academic research. Its research workflow includes literature search, ranking when a dataset fingerprint is available, trend and landscape summaries, and candidate research gaps. Browser reports can be printed or saved as PDF.

The desktop application additionally supports spreadsheet and Parquet inputs, the visual Node Editor, notebook and kernel tooling, project version history, local runtime package installation, and its broader local workflows. The web beta does not run model evaluations or execute user-supplied code.

## Preview hosting

The Render configuration uses free compute for an initial preview. Free web services may sleep after inactivity and take time to wake. Review [Render's current plans](https://render.com/pricing) and [free service limits](https://render.com/docs/free) before relying on a free deployment for continuous public traffic.
