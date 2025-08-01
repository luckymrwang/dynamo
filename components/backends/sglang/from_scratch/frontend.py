# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Optional, List

import uvloop
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    messages: List[Message]
    max_tokens: Optional[int] = 25
    temperature: Optional[float] = 0.6


class FrontendRequestHandler:
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime
        self.processor_client = None
        self.app = None

    async def initialize(self):
        """Initialize the frontend service"""
        # Get processor client
        self.processor_client = (
            await self.runtime.namespace("inference")
            .component("processor")
            .endpoint("process")
            .client()
        )

        # Create FastAPI app
        self.app = FastAPI(title="Dynamo")

        # Add routes
        self.setup_routes()

        logging.info("Frontend initialized successfully")

    def setup_routes(self):
        """Setup FastAPI routes"""
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                # Send request to processor
                # generate under the hood is using round robin
                processor_response = await self.processor_client.generate(request.model_dump())
                
                # Collect all content
                full_content = ""
                finished = False
                async for chunk in processor_response:
                    if chunk:
                        if chunk.get("content"):
                            full_content += chunk["content"]
                        if chunk.get("finish_reason"):
                            finished = True
                            break
                        if chunk.get("error"):
                            raise HTTPException(status_code=500, detail=chunk["error"])
                
                if not finished:
                    raise HTTPException(status_code=500, detail="Stream ended before generation completed")

                return {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"message": Message(role="assistant", content=full_content)}],
                }

            except Exception as e:
                logging.error(f"Error in chat completions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

    async def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server"""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        
        logging.info(f"Starting FastAPI server on {host}:{port}")
        await server.serve()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function"""
    # Initialize frontend
    frontend = FrontendRequestHandler(runtime)
    await frontend.initialize()

    # Start FastAPI server
    await frontend.run_server()

if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())