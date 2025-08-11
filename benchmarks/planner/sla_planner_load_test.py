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

import argparse
import asyncio
import json
import logging
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ResponseMetrics:
    """Track response metrics for analysis"""

    timestamp: float
    ttft: float  # Time to first token
    total_duration: float
    tokens_generated: int
    input_tokens: int
    status_code: int
    error: str = None


class LoadTestConfig:
    """Configuration for load test patterns"""

    def __init__(self, args):
        self.frontend_url = args.frontend_url
        self.pattern = args.pattern
        self.concurrent_users = args.concurrent_users
        self.duration = args.duration
        self.ramp_duration = args.ramp_duration
        self.burst_interval = args.burst_interval
        self.burst_requests = args.burst_requests
        self.request_timeout = args.request_timeout
        self.metrics_interval = args.metrics_interval


class SLAPlannerLoadTest:
    """Main load testing class"""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.session = None
        self.responses: List[ResponseMetrics] = []
        self.active_requests = 0
        self.total_requests = 0
        self.running = True

        # Test payload based on the provided curl command
        self.test_payload = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "messages": [
                {
                    "role": "user",
                    "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
                }
            ],
            "max_tokens": 30,
            "stream": True,  # CRITICAL: Streaming required for SLA planner metrics
        }

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    async def _create_session(self):
        """Create aiohttp session with appropriate settings"""
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Content-Type": "application/json"},
        )

    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    async def send_streaming_request(self, request_id: int) -> ResponseMetrics:
        """Send a single streaming request and measure metrics"""
        start_time = time.time()
        ttft = None
        tokens_generated = 0
        input_tokens = len(self.test_payload["messages"][0]["content"].split())

        try:
            self.active_requests += 1
            async with self.session.post(
                f"{self.config.frontend_url}/v1/chat/completions",
                json=self.test_payload,
            ) as response:
                if response.status != 200:
                    return ResponseMetrics(
                        timestamp=start_time,
                        ttft=0,
                        total_duration=time.time() - start_time,
                        tokens_generated=0,
                        input_tokens=input_tokens,
                        status_code=response.status,
                        error=f"HTTP {response.status}",
                    )

                # Process streaming response
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk_data = json.loads(line[6:])  # Remove 'data: ' prefix
                            if "choices" in chunk_data and chunk_data["choices"]:
                                choice = chunk_data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    # First token received
                                    if ttft is None:
                                        ttft = time.time() - start_time
                                    tokens_generated += 1
                        except json.JSONDecodeError:
                            continue

                total_duration = time.time() - start_time
                return ResponseMetrics(
                    timestamp=start_time,
                    ttft=ttft or total_duration,
                    total_duration=total_duration,
                    tokens_generated=tokens_generated,
                    input_tokens=input_tokens,
                    status_code=response.status,
                )

        except Exception as e:
            return ResponseMetrics(
                timestamp=start_time,
                ttft=0,
                total_duration=time.time() - start_time,
                tokens_generated=0,
                input_tokens=input_tokens,
                status_code=0,
                error=str(e),
            )
        finally:
            self.active_requests -= 1
            self.total_requests += 1

    async def sustained_load_pattern(self):
        """Generate sustained concurrent load"""
        logger.info(
            f"Starting sustained load: {self.config.concurrent_users} concurrent users for {self.config.duration}s"
        )

        end_time = time.time() + self.config.duration

        async def user_session(user_id: int):
            """Simulate a single user sending requests continuously"""
            request_count = 0
            while time.time() < end_time and self.running:
                try:
                    metrics = await self.send_streaming_request(
                        f"{user_id}-{request_count}"
                    )
                    self.responses.append(metrics)

                    # Log every 10th request for this user
                    if request_count % 10 == 0:
                        logger.info(
                            f"User {user_id}: Request {request_count}, TTFT: {metrics.ttft:.3f}s, Tokens: {metrics.tokens_generated}"
                        )

                    request_count += 1

                    # Small delay between requests from same user
                    await asyncio.sleep(1.0)

                except Exception as e:
                    logger.error(f"User {user_id} error: {e}")
                    await asyncio.sleep(5.0)

        # Start all user sessions
        tasks = [user_session(i) for i in range(self.config.concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def burst_load_pattern(self):
        """Generate burst load pattern"""
        logger.info(
            f"Starting burst load: {self.config.burst_requests} requests every {self.config.burst_interval}s for {self.config.duration}s"
        )

        end_time = time.time() + self.config.duration
        burst_count = 0

        while time.time() < end_time and self.running:
            logger.info(
                f"Sending burst {burst_count + 1}: {self.config.burst_requests} requests"
            )

            # Send burst of requests
            tasks = [
                self.send_streaming_request(f"burst-{burst_count}-{i}")
                for i in range(self.config.burst_requests)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, ResponseMetrics):
                    self.responses.append(result)
                else:
                    logger.error(f"Burst request failed: {result}")

            burst_count += 1

            # Wait for next burst
            if time.time() < end_time:
                await asyncio.sleep(self.config.burst_interval)

    async def ramp_up_pattern(self):
        """Generate ramp-up load pattern"""
        logger.info(
            f"Starting ramp-up: 1 to {self.config.concurrent_users} users over {self.config.ramp_duration}s, then sustained for {self.config.duration}s"
        )

        # Ramp-up phase
        ramp_step = self.config.ramp_duration / self.config.concurrent_users
        active_users = []

        for user_id in range(self.config.concurrent_users):
            if not self.running:
                break

            # Start new user
            async def user_session(uid: int):
                request_count = 0
                while self.running:
                    try:
                        metrics = await self.send_streaming_request(
                            f"ramp-{uid}-{request_count}"
                        )
                        self.responses.append(metrics)
                        request_count += 1
                        await asyncio.sleep(2.0)
                    except Exception as e:
                        logger.error(f"Ramp user {uid} error: {e}")
                        await asyncio.sleep(5.0)

            task = asyncio.create_task(user_session(user_id))
            active_users.append(task)

            logger.info(f"Added user {user_id + 1}/{self.config.concurrent_users}")
            await asyncio.sleep(ramp_step)

        # Sustained phase
        logger.info("Ramp-up complete, sustaining load...")
        await asyncio.sleep(self.config.duration)

        # Cancel all user tasks
        for task in active_users:
            task.cancel()

        await asyncio.gather(*active_users, return_exceptions=True)

    async def monitor_metrics(self):
        """Monitor and log test metrics periodically"""
        while self.running:
            await asyncio.sleep(self.config.metrics_interval)

            if self.responses:
                recent_responses = [
                    r
                    for r in self.responses
                    if time.time() - r.timestamp < self.config.metrics_interval
                ]

                if recent_responses:
                    ttfts = [r.ttft for r in recent_responses if r.ttft > 0]
                    durations = [r.total_duration for r in recent_responses]
                    tokens = [r.tokens_generated for r in recent_responses]
                    errors = [
                        r for r in recent_responses if r.error or r.status_code != 200
                    ]

                    logger.info("=== METRICS SUMMARY ===")
                    logger.info(f"Active requests: {self.active_requests}")
                    logger.info(f"Total requests: {self.total_requests}")
                    logger.info(f"Recent requests: {len(recent_responses)}")
                    logger.info(
                        f"Error rate: {len(errors)/len(recent_responses)*100:.1f}%"
                    )

                    if ttfts:
                        logger.info(
                            f"TTFT - avg: {statistics.mean(ttfts):.3f}s, p95: {statistics.quantiles(ttfts, n=20)[18]:.3f}s"
                        )
                    if durations:
                        logger.info(
                            f"Duration - avg: {statistics.mean(durations):.3f}s"
                        )
                    if tokens:
                        logger.info(f"Tokens - avg: {statistics.mean(tokens):.1f}")

                    logger.info("======================")

    async def check_frontend_health(self):
        """Check if frontend is accessible"""
        try:
            async with self.session.get(
                f"{self.config.frontend_url}/health"
            ) as response:
                if response.status == 200:
                    logger.info("Frontend health check passed")
                    return True
                else:
                    logger.warning(
                        f"Frontend health check failed: HTTP {response.status}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Frontend health check failed: {e}")
            return False

    async def run(self):
        """Run the load test"""
        await self._create_session()

        try:
            # Health check
            if not await self.check_frontend_health():
                logger.error(
                    "Frontend is not accessible. Please check that it's running at "
                    + self.config.frontend_url
                )
                return

            logger.info("Starting SLA Planner Load Test")
            logger.info(f"Frontend URL: {self.config.frontend_url}")
            logger.info(f"Pattern: {self.config.pattern}")
            logger.info(
                "NOTE: SLA planner adjusts workers every ~60 seconds based on metrics"
            )
            logger.info("=" * 60)

            # Start metrics monitoring
            metrics_task = asyncio.create_task(self.monitor_metrics())

            # Run load pattern
            if self.config.pattern == "sustained":
                await self.sustained_load_pattern()
            elif self.config.pattern == "burst":
                await self.burst_load_pattern()
            elif self.config.pattern == "ramp":
                await self.ramp_up_pattern()
            else:
                raise ValueError(f"Unknown pattern: {self.config.pattern}")

            # Stop monitoring
            metrics_task.cancel()

            # Final summary
            await self.print_final_summary()

        finally:
            await self._close_session()

    async def print_final_summary(self):
        """Print final test summary"""
        if not self.responses:
            logger.info("No responses recorded")
            return

        successful_responses = [
            r for r in self.responses if r.status_code == 200 and not r.error
        ]
        failed_responses = [
            r for r in self.responses if r.status_code != 200 or r.error
        ]

        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total requests: {len(self.responses)}")
        logger.info(f"Successful: {len(successful_responses)}")
        logger.info(f"Failed: {len(failed_responses)}")
        logger.info(
            f"Success rate: {len(successful_responses)/len(self.responses)*100:.1f}%"
        )

        if successful_responses:
            ttfts = [r.ttft for r in successful_responses if r.ttft > 0]
            durations = [r.total_duration for r in successful_responses]
            tokens = [r.tokens_generated for r in successful_responses]

            if ttfts:
                logger.info("\nTTFT Statistics:")
                logger.info(f"  Average: {statistics.mean(ttfts):.3f}s")
                logger.info(f"  Median: {statistics.median(ttfts):.3f}s")
                logger.info(f"  P95: {statistics.quantiles(ttfts, n=20)[18]:.3f}s")
                logger.info(f"  P99: {statistics.quantiles(ttfts, n=100)[98]:.3f}s")

            if durations:
                logger.info("\nRequest Duration Statistics:")
                logger.info(f"  Average: {statistics.mean(durations):.3f}s")
                logger.info(f"  Median: {statistics.median(durations):.3f}s")

            if tokens:
                logger.info("\nToken Generation:")
                logger.info(f"  Average tokens: {statistics.mean(tokens):.1f}")
                logger.info(f"  Total tokens: {sum(tokens)}")

        logger.info("\nSLA Planner Scaling Notes:")
        logger.info("- Monitor kubectl logs for planner scaling decisions")
        logger.info("- Check 'kubectl get pods' for new worker instances")
        logger.info("- Planner adjusts every ~60 seconds based on metrics")
        logger.info("- Look for TTFT/ITL increases that trigger scaling")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SLA Planner Load Test")
    parser.add_argument(
        "--frontend-url",
        default="http://localhost:8000",
        help="Frontend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--pattern",
        choices=["sustained", "burst", "ramp"],
        default="sustained",
        help="Load pattern (default: sustained)",
    )
    parser.add_argument(
        "--concurrent-users",
        type=int,
        default=5,
        help="Number of concurrent users (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=180,
        help="Test duration in seconds (default: 180)",
    )
    parser.add_argument(
        "--ramp-duration",
        type=int,
        default=60,
        help="Ramp-up duration for ramp pattern (default: 60)",
    )
    parser.add_argument(
        "--burst-interval",
        type=int,
        default=30,
        help="Interval between bursts in seconds (default: 30)",
    )
    parser.add_argument(
        "--burst-requests",
        type=int,
        default=10,
        help="Number of requests per burst (default: 10)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--metrics-interval",
        type=int,
        default=30,
        help="Metrics reporting interval (default: 30)",
    )

    args = parser.parse_args()
    config = LoadTestConfig(args)

    # Run the load test
    load_test = SLAPlannerLoadTest(config)

    try:
        asyncio.run(load_test.run())
    except KeyboardInterrupt:
        logger.info("Load test interrupted by user")
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
