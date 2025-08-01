# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import sys
from typing import Dict, Any

import sglang
import uvloop
from sglang.srt.server_args import ServerArgs

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TEMPERATURE = 0.7


class SGLangEngineHandler:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.engine = sglang.Engine(server_args=server_args)
        logging.info(f"SGLang engine worker initialized successfully")

    async def generate(self, request: Dict[str, Any]):
        """Generate response using SGLang - non-streaming"""
        try:
            logging.info("SGLang engine received request")
            
            token_ids = request.get("token_ids", [])
            sampling_options = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            
            # Build sampling params in SGLang format
            sampling_params = {
                "temperature": sampling_options.get("temperature", DEFAULT_TEMPERATURE),
                "max_new_tokens": stop_conditions.get("max_tokens", 25),
            }
            
            logging.info(f"Input tokens: {len(token_ids)}, generating up to {sampling_params['max_new_tokens']} tokens")
            
            # Non-streaming generation
            logging.info("Calling SGLang async_generate...")
            result = await self.engine.async_generate(
                input_ids=token_ids, 
                sampling_params=sampling_params, 
                stream=False
            )

            logging.info(f"SGLang returned result: {type(result)}")
            
            # SGLang returns a single result dict when not streaming
            output_ids = result["output_ids"]
            finish_reason = result["meta_info"]["finish_reason"]
            
            logging.info(f"Generated {len(output_ids)} tokens, finish_reason: {finish_reason}")
            
            # Return all generated tokens at once in our expected format
            response = {
                "token_ids": output_ids,
                "finish_reason": finish_reason["type"] if finish_reason else "stop"
            }
            logging.info(f"Engine yielding response: {response}")
            yield response
            
            logging.info("SGLang engine completed generation")
            
        except Exception as e:
            logging.error(f"Error in SGLang engine generate: {e}")
            yield {"error": str(e)}

def parse_sglang_args(args: list[str]) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed_args = parser.parse_args(args)
    return ServerArgs.from_cli_args(parsed_args)

@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function"""
    # Parse command line arguments
    args = parse_sglang_args(sys.argv[1:])
    
    logging.info(f"Starting SGLang engine worker with model {args.model_path}")
    
    # Create engine component
    component = runtime.namespace("inference").component("engine")
    await component.create_service()

    # Initialize SGLang engine handler
    engine = SGLangEngineHandler(server_args=args)

    # Create generate endpoint
    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(engine.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())