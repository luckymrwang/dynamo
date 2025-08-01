# Building Custom Inference Flows with Dynamo Runtime

## Introduction
This guide demonstrates how to build your own pipeline using only the dynamo runtime (NATS for request transport and ETCD for dyanmo worker discovery/registration). We purposefully do not include any helpers from dynamo runtime (rust based openai api server, tokenizer/detokenizer, register_llm hooks, etc). Hopefully this allows you to easily put together your own pipelines!

In this example, we will build a simple pipeline that takes in a chat completion request, tokenizes it, routes to the best available engine based on KV cache metrics, and then sends it to an LLM for inference. At every step, we will also explain how much of this code is handled in our production ready components.

## Components

All components communicate via NATS and register themselves in etcd using the `DistributedRuntime`.

### 1. Frontend (FastAPI + NATS Client)
- Accepts OpenAI-compatible chat completion requests
- Sends requests to Processor via NATS
- Returns streaming responses to client

### 2. Processor (Tokenizer Service)  
- Receives chat requests from Frontend
- Tokenizes input using model tokenizer
- Sends tokenized request to Router
- Handles response detokenization

### 3. Router (Load Balancer)
- Similar to `examples/deployments/router_standalone/router.py`
- Receives tokenized requests from Processor
- Uses metrics/KV cache data to select best Engine
- Returns Engine worker ID to Processor

### 4. Engine (LLM Inference)
- Similar to `lib/bindings/python/examples/hello_world/server_sglang_tok.py`
- Performs actual LLM inference
- Publishes metrics and KV events for Router
- Returns token streams

## Key Patterns

### Using DistributedRuntime

All components use the `@dynamo_worker` decorator and `DistributedRuntime`:

```python
from dynamo.runtime import DistributedRuntime, dynamo_worker

@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("your_namespace").component("your_component")
    await component.create_service()
    
    endpoint = component.endpoint("your_endpoint")
    # Set up your service logic
    await endpoint.serve_endpoint(your_handler)
```

### NATS Communication

Components communicate by calling endpoints on other components:

```python
# Get client for another component
client = await runtime.namespace("namespace").component("component").endpoint("endpoint").client()

# Send request
response = await client.your_method(request_data)
```

### Service Registration

The runtime handles etcd registration automatically when you call `create_service()`. Services are discoverable by namespace/component/endpoint.

## Component Details

### Frontend
- **Namespace**: `inference`
- **Component**: `frontend` 
- **Endpoints**: `chat_completions`
- **Input**: OpenAI chat completion request
- **Output**: Streaming chat completion response
- **Calls**: Processor's `process` endpoint

### Processor  
- **Namespace**: `inference`
- **Component**: `processor`
- **Endpoints**: `process`
- **Input**: Chat completion request
- **Output**: Chat completion response
- **Calls**: Router's `find_worker` and Engine's `generate`

### Router
- **Namespace**: `inference` 
- **Component**: `router`
- **Endpoints**: `find_worker`
- **Input**: Tokenized request + metadata
- **Output**: Selected worker ID
- **Dependencies**: Subscribes to Engine metrics via ZMQ

### Engine
- **Namespace**: `inference`
- **Component**: `engine` 
- **Endpoints**: `generate`
- **Input**: Tokenized request
- **Output**: Token stream
- **Dependencies**: Publishes metrics and KV events

## Metrics & Load Balancing

For Router/Engine coordination, see `components/backends/sglang/src/dynamo/sglang/worker/main.py`:

- Engines publish metrics via `WorkerMetricsPublisher`
- Router subscribes to metrics via ZMQ sockets
- KV cache events published via `ZmqKvEventPublisher`
- Router uses `RadixTree` for cache-aware routing

## Getting Started

1. **Setup**: Install dynamo runtime and dependencies
2. **Start Infrastructure**: 
   ```bash
   nats-server -js
   etcd
   ```
3. **Run Components** in order:
   ```bash
   python engine.py --worker-id 0
   python engine.py --worker-id 1  
   python router.py
   python processor.py
   python frontend.py
   ```

## Example File References

- **Router Logic**: `examples/deployments/router_standalone/router.py`
- **Engine Pattern**: `lib/bindings/python/examples/hello_world/server_sglang_tok.py`  
- **Metrics Setup**: `components/backends/sglang/src/dynamo/sglang/worker/main.py`

## Key Differences from Standard Dynamo

- No `register_llm()` calls - you handle model loading directly
- No automatic request routing - you implement custom routing logic
- Manual metrics publishing - you control what metrics to expose
- Custom request/response formats - not bound to standard LLM protocols

This approach gives you full control over the inference pipeline while leveraging Dynamo's robust transport and discovery infrastructure.