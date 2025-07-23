<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Hello World Example

This is the simplest Dynamo example demonstrating a basic service using Dynamo's distributed runtime. It showcases the fundamental concepts of creating endpoints and workers in the Dynamo runtime system.

## Architecture

```
Client (dynamo_worker)
      │
      ▼
┌─────────────┐
│   Backend   │  Dynamo endpoint (/generate)
└─────────────┘
```

## Components

- **Backend**: A Dynamo service with an endpoint that receives text input and streams back greetings for each comma-separated word
- **Client**: A Dynamo worker that connects to the backend service and processes streaming responses

## Implementation Details

The example demonstrates:

1. **Endpoint Definition**: Using the `@dynamo_endpoint` decorator to create streaming endpoints
2. **Worker Setup**: Using the `@dynamo_worker()` decorator to create distributed runtime workers
3. **Service Creation**: Creating services and endpoints using the distributed runtime API
4. **Streaming Responses**: Yielding data for real-time streaming
5. **Client Integration**: Connecting to services and processing streams
6. **Logging**: Basic logging configuration with `configure_dynamo_logging`

## Getting Started

### Prerequisites

```{note}
Make sure that `etcd` and `nats` are running
```

### Running the Example

1. Start the backend service:
```bash
cd examples/runtime/hello_world
python hello_world.py
```

2. In a separate terminal, run the client:
```bash
cd examples/runtime/hello_world
python client.py
```

The client will connect to the backend service and stream the results.

### Expected Output

When running the client, you should see streaming output like:
```
Hello world!
Hello sun!
Hello moon!
Hello star!
```

## Code Structure

### Backend Service (`hello_world.py`)

- **`content_generator`**: A dynamo endpoint that processes text input and yields greetings
- **`worker`**: A dynamo worker that sets up the service, creates the endpoint, and serves it

### Client (`client.py`)

- **`worker`**: A dynamo worker that connects to the backend service and processes the streaming response

## Next Steps

After understanding this basic example, explore:

```{seealso}
- [Hello World Configurable](../hello_world_configurable/README.md): Add configuration to your services
- [Simple Pipeline](../simple_pipeline/README.md): Connect multiple services together
- [Multistage Pipeline](../multistage_pipeline/README.md): Build complex multi-stage processing pipelines
```
