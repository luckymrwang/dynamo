# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import time

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="HelloWorld")


"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
"""


@dynamo_endpoint(str, str)
async def content_generator(request: str):
    logger.info(f"Received request: {request}")
    for word in request.split(","):
        time.sleep(1)
        yield f"Hello {word}!\n"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    logger.info(
        f"Primary lease ID: {runtime.etcd_client().primary_lease_id()}/{runtime.etcd_client().primary_lease_id():#x}"
    )
    logger.info(f"Starting worker {runtime.worker_id()}")

    component = runtime.namespace("hello_world").component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(content_generator)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())

# @service(
#     dynamo={"namespace": "hello_world"},
# )
# class Frontend:
#     """A simple frontend HTTP API that forwards requests to the dynamo graph."""

#     def __init__(self) -> None:
#         # Configure logging
#         configure_dynamo_logging(service_name="Frontend")

#     # alternative syntax: @endpoint(transports=[DynamoTransport.HTTP])
#     @api()
#     async def generate(self, request: RequestType):
#         """Stream results from the pipeline."""
#         logger.info(f"Frontend received: {request.text}")

#         def content_generator():
#             for word in request.text.split(","):
#                 time.sleep(1)
#                 yield f"Hello {word}!\n"

#         return StreamingResponse(content_generator())
