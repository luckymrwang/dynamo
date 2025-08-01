# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import sys
from typing import Optional, List, Dict, Any

import uvloop
from pydantic import BaseModel
from transformers import AutoTokenizer

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


class ProcessorRequestHandler:
    def __init__(self, runtime: DistributedRuntime, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", enable_router: bool = True):
        self.runtime = runtime
        self.router_client = None
        self.engine_client = None
        self.model_name = model_name
        self.enable_router = enable_router
        self.tokenizer = None

    async def initialize(self):
        """Initialize the processor service"""
        # Load tokenizer
        logging.info(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info("Tokenizer loaded successfully")

        # Get router client if enabled
        if self.enable_router:
            self.router_client = (
                await self.runtime.namespace("inference")
                .component("router")
                .endpoint("find_worker")
                .client()
            )
            logging.info("Router client initialized")
        else:
            logging.info("Router disabled - will use round-robin worker selection")

        # Get engine client  
        self.engine_client = (
            await self.runtime.namespace("inference")
            .component("engine")
            .endpoint("generate")
            .client()
        )

        logging.info("Processor initialized successfully")

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the loaded tokenizer"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    def messages_to_text(self, messages: List[Message]) -> str:
        """Convert chat messages to a single text string using chat template if available"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Convert to dict format for chat template
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                text = self.tokenizer.apply_chat_template(
                    message_dicts, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return text
            except Exception as e:
                logging.warning(f"Chat template failed: {e}, falling back to simple format")
        
        # Fallback to simple format
        text_parts = []
        for msg in messages:
            text_parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(text_parts) + "\nassistant:"

    async def generate(self, request: Dict[str, Any]):
        """Process chat completion request"""
        try:
            # Parse the request
            chat_request = ChatCompletionRequest(**request)
            
            # Convert messages to text
            text = self.messages_to_text(chat_request.messages)
            logging.info(f"Processing text: {text[:200]}...")
            
            # Real tokenization
            token_ids = self.tokenize(text)
            num_tokens = len(token_ids)
            
            logging.info(f"Tokenized to {num_tokens} tokens: {token_ids[:10]}...")

            # Get worker ID - either from router or default
            if self.enable_router and self.router_client:
                # Get best worker from router
                router_request = {
                    "local_hashes": token_ids[:10],  # Send first 10 tokens as hashes
                    "num_tokens": num_tokens
                }
                
                router_response = await self.router_client.find_best_worker(router_request)
                worker_id = router_response.get("worker_id", 0)
                logging.info(f"Router selected worker {worker_id}")
            else:
                # Simple fallback - use worker 0 or could implement round-robin
                worker_id = 0
                logging.info(f"Using default worker {worker_id} (router disabled)")

            # Prepare engine request
            engine_request = {
                "token_ids": token_ids,
                "sampling_options": {
                    "temperature": chat_request.temperature,
                },
                "stop_conditions": {
                    "max_tokens": chat_request.max_tokens,
                },
                "model": chat_request.model,
                "worker_id": worker_id
            }

            # Send to engine
            engine_response = await self.engine_client.generate(engine_request)

            # Process engine response (engine returns token IDs, we need to convert to text)
            response_received = False
            async for chunk in engine_response:
                if chunk:
                    response_received = True
                    logging.info(f"Processor received chunk: {chunk}")
                    
                    if chunk.get("error"):
                        yield {"error": chunk["error"]}
                        return
                    
                    # Convert token IDs back to text
                    if chunk.get("token_ids"):
                        output_token_ids = chunk["token_ids"]
                        if output_token_ids:  # Only detokenize if we have tokens
                            content = self.detokenize(output_token_ids)
                            logging.info(f"Generated content: {content[:100]}...")
                            yield {"content": content}
                    
                    if chunk.get("finish_reason"):
                        logging.info(f"Received finish_reason: {chunk['finish_reason']}")
                        yield {"finish_reason": chunk["finish_reason"]}
                        return  # Explicitly end after finish signal
            
            # If we got here without a finish signal, something went wrong
            if response_received:
                logging.warning("Engine response ended without finish_reason")
                yield {"finish_reason": "stop"}  # Ensure we always send finish signal
            else:
                logging.error("No response received from engine")
                yield {"error": "No response from engine"}

        except Exception as e:
            logging.error(f"Error in processor generate: {e}")
            yield {"error": str(e)}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Processor component for inference pipeline")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name/path for tokenizer (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)"
    )
    parser.add_argument(
        "--enable-router",
        action="store_true",
        default=False,
        help="Enable router for worker selection (default: False)"
    )
    parser.add_argument(
        "--tokenize-in-processor",
        action="store_true",
        default=False,
        help="Tokenize in processor (default: False)"
    )
    return parser.parse_args()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function"""
    # Parse command line arguments
    args = parse_args()
    
    logging.info(f"Starting processor with model: {args.model}, router enabled: {args.enable_router}")
    
    # Create processor component
    component = runtime.namespace("inference").component("processor")
    await component.create_service()

    # Initialize processor with args
    processor = ProcessorRequestHandler(
        runtime, 
        model_name=args.model,
        enable_router=args.enable_router
    )
    await processor.initialize()

    # Create process endpoint
    endpoint = component.endpoint("process")
    await endpoint.serve_endpoint(processor.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())