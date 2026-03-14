from prometheus_client import Counter, Histogram

HTTP_LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
)

LLM_LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0
)

http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=HTTP_LATENCY_BUCKETS,
)

http_server_errors_total = Counter(
    "http_server_errors_total",
    "Total number of HTTP 5xx responses",
    ["method", "path", "status_code"],
)

app_errors_total = Counter(
    "app_errors_total",
    "Total number of internal application errors",
    ["component", "error_type"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["model"],
    buckets=LLM_LATENCY_BUCKETS,
)

llm_prompt_tokens_total = Counter(
    "llm_prompt_tokens_total",
    "Total prompt tokens used by LLM requests",
    ["model"],
)

llm_completion_tokens_total = Counter(
    "llm_completion_tokens_total",
    "Total completion tokens used by LLM requests",
    ["model"],
)

llm_total_tokens_total = Counter(
    "llm_total_tokens_total",
    "Total tokens used by LLM requests",
    ["model"],
)

agent_tool_invocations_total = Counter(
    "agent_tool_invocations_total",
    "Total number of agent tool invocations",
    ["tool_name"],
)

KNOWN_PATHS = {
    "/": "/",
    "/health": "/health",
    "/query": "/query",
    "/metrics": "/metrics",
}


def normalize_path(path: str) -> str:
    if path in KNOWN_PATHS:
        return KNOWN_PATHS[path]

    if path.startswith("/history/"):
        return "/history/{thread_id}"

    return "unknown"


def record_http_request(method: str, path: str, status_code: int, duration: float) -> None:
    normalized_path = normalize_path(path)

    if normalized_path == "/metrics":
        return

    http_requests_total.labels(
        method=method,
        path=normalized_path,
        status_code=str(status_code),
    ).inc()

    http_request_duration_seconds.labels(
        method=method,
        path=normalized_path,
    ).observe(duration)


def record_http_server_error(method: str, path: str, status_code: int) -> None:
    normalized_path = normalize_path(path)

    if normalized_path == "/metrics":
        return

    if status_code >= 500:
        http_server_errors_total.labels(
            method=method,
            path=normalized_path,
            status_code=str(status_code),
        ).inc()


def record_llm_usage(
    model: str,
    duration: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    llm_request_duration_seconds.labels(model=model).observe(duration)
    llm_prompt_tokens_total.labels(model=model).inc(prompt_tokens)
    llm_completion_tokens_total.labels(model=model).inc(completion_tokens)
    llm_total_tokens_total.labels(model=model).inc(total_tokens)


def record_error(component: str, error_type: str) -> None:
    app_errors_total.labels(
        component=component,
        error_type=error_type,
    ).inc()


def record_tool_invocation(tool_name: str) -> None:
    agent_tool_invocations_total.labels(tool_name=tool_name).inc()