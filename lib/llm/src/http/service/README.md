# HTTP Request Logging

This module provides comprehensive HTTP request logging with timestamps and duration tracking for each request ID.

## Features

- **Request ID Tracking**: Each HTTP request gets a unique UUID that is logged throughout its lifecycle
- **Timestamp Logging**: Start and end timestamps are logged for each request
- **Duration Tracking**: Total request duration is calculated and logged
- **Endpoint-Specific Logging**: Different endpoints (completions, chat_completions, embeddings) have tailored logging
- **Error Logging**: Failed requests are logged with error details and duration
- **Streaming Support**: Special handling for streaming requests with event count tracking
- **Middleware Integration**: Optional middleware for additional request context

## Log Format

### Request Start
```
INFO HTTP request started
  request_id: "550e8400-e29b-41d4-a716-446655440000"
  start_timestamp: 1704067200000
  endpoint: "chat_completions"
  model: "llama-2-7b"
```

### Request Success
```
INFO HTTP request completed successfully
  request_id: "550e8400-e29b-41d4-a716-446655440000"
  end_timestamp: 1704067201500
  duration_ms: 1500
  streaming: false
```

### Request Error
```
ERROR HTTP request failed
  request_id: "550e8400-e29b-41d4-a716-446655440000"
  end_timestamp: 1704067200250
  duration_ms: 250
  error: "Model not found"
  error_type: "model_not_found"
```

### Streaming Request Completion
```
INFO HTTP streaming request completed
  request_id: "550e8400-e29b-41d4-a716-446655440000"
  end_timestamp: 1704067205000
  duration_ms: 5000
  event_count: 42
  streaming: true
```

## Configuration

### Basic Usage
```rust
use dynamo_llm::http::service::service_v2::HttpService;

let service = HttpService::builder()
    .port(8080)
    .enable_request_logging(true)  // Enable detailed request logging (default: true)
    .enable_trace_layer(false)     // Enable tower-http trace layer (default: false)
    .build()?;
```

### With Middleware
```rust
let service = HttpService::builder()
    .port(8080)
    .enable_request_logging(true)
    .enable_trace_layer(true)
    .build()?;
```

## Middleware Features

The optional middleware provides additional context:

- **Client IP Address**: Extracted from connection info or X-Forwarded-For headers
- **User Agent**: HTTP User-Agent header
- **Content Length**: Request and response content lengths
- **HTTP Method and Path**: Request method and matched path

### Middleware Log Example
```
INFO HTTP request received
  request_id: "middleware-generated"
  start_timestamp: 1704067200000
  method: "POST"
  path: "/v1/chat/completions"
  client_ip: "192.168.1.100"
  user_agent: "curl/7.68.0"
  content_length: 256

INFO HTTP request completed
  request_id: "middleware-generated"
  end_timestamp: 1704067201000
  duration_ms: 1000
  status_code: 200
  response_content_length: 1024
```

## Request Context Helper

For custom handlers, use the `RequestContext` helper:

```rust
use dynamo_llm::http::service::middleware::RequestContext;

async fn custom_handler(headers: HeaderMap) -> Result<Response, Error> {
    let ctx = RequestContext::from_headers(&headers);

    ctx.log_start("custom_endpoint", Some("my-model"));

    // ... process request ...

    match result {
        Ok(_) => ctx.log_success(false, Some("custom processing complete")),
        Err(e) => ctx.log_error(&e.to_string(), "processing_error"),
    }

    // Response handling...
}
```

## Integration with Existing Metrics

The logging system works alongside the existing Prometheus metrics:

- **Request Counters**: Incremented for success/error status
- **Duration Histograms**: Request duration tracking
- **Inflight Gauges**: Active request monitoring

## Log Levels

- `INFO`: Normal request lifecycle events (start, success, stream start)
- `ERROR`: Request failures, errors, and exceptions
- `DEBUG`: Detailed middleware information (when trace layer is enabled)
- `TRACE`: Very detailed request processing information

## Performance Considerations

- Logging is asynchronous and non-blocking
- Timestamp calculation uses high-resolution system time
- Request IDs are generated using UUID v4 for uniqueness
- Structured logging format enables efficient log parsing and analysis

## Example Output

```
2024-01-01T12:00:00.000Z INFO HTTP request started request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890" start_timestamp=1704067200000 endpoint="chat_completions" model="llama-2-7b"
2024-01-01T12:00:01.500Z INFO HTTP request completed successfully request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890" end_timestamp=1704067201500 duration_ms=1500 streaming=false
```

This logging system provides comprehensive observability for HTTP requests, enabling effective monitoring, debugging, and performance analysis of the Dynamo LLM HTTP service.