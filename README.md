# OpenRouter Observability Proxy

A small local proxy between an OpenAI-compatible client and OpenRouter. It logs every chat completion to SQLite and shows tokens, cost, latency and content in a dashboard. The only change a client needs is its `base_url`.

## Setup

Requires Python 3.11 or newer.

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Or with pip, inside a virtual environment:

```bash
pip install -e ".[dev]"
```

Optionally copy the example settings. The proxy reads `.env` from the directory you start it in.

```bash
cp .env.example .env
```

## Run

```bash
uv run python -m app.main
```

The proxy listens on `http://127.0.0.1:8000`, and the dashboard is at the same address.

## Usage

Point any OpenAI-compatible client at the proxy and keep using your OpenRouter key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="<OPENROUTER_API_KEY>",
)

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(response.choices[0].message.content)
```

Streaming works the same way with `stream=True`.

To group requests in the dashboard, send a session ID. The proxy stores it and does not forward it to OpenRouter:

```python
client.chat.completions.create(
    model="openrouter/free",
    messages=[{"role": "user", "content": "Say hello"}],
    extra_headers={"X-Obs-Session-Id": "checkout-flow-42"},
)
```

Every proxied response carries an `X-Obs-Request-Id` header. For chat completions, it matches the ID of the logged record.

## Endpoints

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | Proxied and logged, streaming and non-streaming |
| `GET /v1/models` | Passed through to OpenRouter, not logged |
| `GET /` | Summary stats and request list, with filters for model, status and session |
| `GET /requests/{id}` | Metadata, messages, output, tool calls and errors for one request |
| `GET /health` | Status, database state, queued records and dropped records |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Upstream base URL |
| `OPENROUTER_API_KEY` | empty | Used only when the client sends no `Authorization` header |
| `DATABASE_PATH` | `./data/observability.db` | SQLite file |
| `LOG_CONTENT` | `true` | When false, message and output content are not stored |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8000` | Port |

`LOG_CONTENT` accepts `true`, `1`, `yes` or `on`. Any other value turns content logging off, so a typo cannot store prompts by accident. With logging off, errors keep only their `code` and `type`, because error messages can echo prompt text.

## Request statuses

| Status | Meaning |
|---|---|
| `success` | The upstream response completed without an error |
| `upstream_error` | OpenRouter returned an HTTP error, reported an error mid-stream, or lost the connection mid-stream |
| `client_disconnected` | The client left before the response finished; partial output is kept and the upstream connection is closed |
| `proxy_error` | The proxy could not reach OpenRouter (returned as 502) or failed internally |

Cost comes only from `usage.cost` in OpenRouter's response and is never estimated. It is empty if OpenRouter does not report it.

## Tests

```bash
uv run pytest
```

The tests use a fake upstream and never call the OpenRouter API.

## How this differs from an observability SDK

Observability SDKs such as LangSmith or Langfuse run inside your application. This proxy runs beside it and only sees the HTTP traffic.

| | This proxy | Observability SDK |
|---|---|---|
| Setup | Change `base_url` | Install a package, then wrap clients or instrument functions |
| Languages | Anything that can call an OpenAI-compatible API | Languages the SDK supports |
| What it captures | Each chat completion: tokens, cost, latency, TTFT and content | LLM calls plus application steps such as retrieval and tool execution |
| Trace shape | A flat list, grouped by `X-Obs-Session-Id` | Usually nested traces across a whole run |
| Cost | Exactly what OpenRouter reports in `usage.cost` | Depends on the tool; some estimate from token counts |
| If it goes down | LLM calls fail, because they travel through it | LLM calls usually continue, because tracing is a side channel |
| Where data lives | A local SQLite file | Usually the vendor's cloud or a server you host |
| Extras | None: no evals, prompt management or alerts | Often evals, prompt management and team features |

Choose the proxy when you want quick, local visibility into LLM calls with no code changes, or when your apps use several languages. Choose an SDK when you need to see what your application did between LLM calls, or need team features. The two can run together, since an SDK can trace an app whose client points at this proxy.

## Limitations

- This is a local MVP with no dashboard authentication. Do not expose it publicly without adding auth.
- It sees only LLM traffic that passes through the proxy, so it does not capture retrieval or local tool execution.
- SQLite is intended for local use.
- [OpenRouter Broadcast](https://openrouter.ai/docs/guides/features/broadcast.md) already exports traces to external tools without a proxy.
