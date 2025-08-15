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

import logging
import os
import select
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import psutil
import requests


def terminate_process(process, logger=logging.getLogger(), immediate_kill=False):
    try:
        logger.info("Terminating PID: %s name: %s", process.pid, process.name())
        if immediate_kill:
            logger.info("Sending Kill: %s %s", process.pid, process.name())
            process.kill()
        else:
            process.terminate()
    except psutil.AccessDenied:
        logger.warning("Access denied for PID %s", process.pid)
    except psutil.NoSuchProcess:
        logger.warning("PID %s no longer exists", process.pid)


def terminate_process_tree(
    pid, logger=logging.getLogger(), immediate_kill=False, timeout=10
):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            terminate_process(child, logger, immediate_kill)

        terminate_process(parent, logger, immediate_kill)

        for child in parent.children(recursive=True):
            try:
                child.wait(timeout)
            except psutil.TimeoutExpired:
                terminate_process(child, logger, immediate_kill=True)
        try:
            parent.wait(timeout)
        except psutil.TimeoutExpired:
            terminate_process(parent, logger, immediate_kill=True)

    except psutil.NoSuchProcess:
        # Process already terminated
        pass


@dataclass
class ManagedProcess:
    command: List[str]
    env: Optional[dict] = None
    health_check_ports: List[int] = field(default_factory=list)
    health_check_urls: List[Any] = field(default_factory=list)
    delayed_start: int = 0
    timeout: int = 300
    working_dir: Optional[str] = None
    display_output: bool = False
    data_dir: Optional[str] = None
    terminate_existing: bool = True
    stragglers: List[str] = field(default_factory=list)
    straggler_commands: List[str] = field(default_factory=list)
    log_dir: str = os.getcwd()

    _logger = logging.getLogger()
    _command_name = None
    _proc: Optional[subprocess.Popen] = None
    _log_path = None
    _output_thread: Optional[threading.Thread] = None

    def __enter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            self._command_name = self.command[0]
            os.makedirs(self.log_dir, exist_ok=True)
            log_name = f"{self._command_name}.log.txt"
            self._log_path = os.path.join(self.log_dir, log_name)

            if self.data_dir:
                self._remove_directory(self.data_dir)

            self._terminate_existing()
            self._start_process()
            time.sleep(self.delayed_start)
            elapsed = self._check_ports(self.timeout)
            self._check_urls(self.timeout - elapsed)

            return self

        except Exception as e:
            self.__exit__(None, None, None)
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        # First, terminate the process to stop output generation
        if self._proc:
            terminate_process_tree(self._proc.pid, self._logger)
            self._proc.wait()

        # Wait for output processing thread to finish
        if (
            hasattr(self, "_output_thread")
            and self._output_thread
            and self._output_thread.is_alive()
        ):
            self._logger.info("Waiting for output thread to finish...")
            self._output_thread.join(timeout=5)  # Give it 5 seconds to finish

        # Now safely close file descriptors
        if self._proc and self._proc.stdout:
            try:
                self._proc.stdout.close()
            except Exception as e:
                self._logger.warning(f"Error closing stdout: {e}")

        if self.data_dir:
            self._remove_directory(self.data_dir)

        for ps_process in psutil.process_iter(["name", "cmdline"]):
            try:
                if ps_process.name() in self.stragglers:
                    self._logger.info(
                        "Terminating Straggler %s %s", ps_process.name(), ps_process.pid
                    )

                    terminate_process_tree(ps_process.pid, self._logger)
                for cmdline in self.straggler_commands:
                    if cmdline in " ".join(ps_process.cmdline()):
                        self._logger.info(
                            "Terminating Straggler Cmdline %s %s %s",
                            ps_process.name(),
                            ps_process.pid,
                            cmdline,
                        )
                        terminate_process_tree(ps_process.pid, self._logger)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have terminated or become inaccessible during iteration
                pass

    def _start_process(self):
        assert self._command_name
        assert self._log_path

        self._logger.info(
            "Running command: %s in %s",
            " ".join(self.command),
            self.working_dir or os.getcwd(),
        )

        # Single subprocess with unified output handling
        self._proc = subprocess.Popen(
            self.command,
            env=self.env or os.environ.copy(),
            cwd=self.working_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Isolate process group to prevent kill 0 from affecting parent
            text=True,  # Use text mode for easier line processing
        )

        # Single output processing function for both cases
        def process_output():
            try:
                # Clear log file first
                with open(self._log_path, "w", encoding="utf-8") as f:
                    pass

                # Process output line by line with non-blocking reads
                while True:
                    # Check if process is still alive
                    if self._proc.poll() is not None:
                        break

                    # Use select to check if there's data available (non-blocking)
                    try:
                        ready, _, _ = select.select(
                            [self._proc.stdout], [], [], 0.1
                        )  # 100ms timeout
                    except (OSError, ValueError):
                        # File descriptor is closed or invalid
                        break

                    if ready:
                        try:
                            line = self._proc.stdout.readline()
                            if not line:  # EOF reached
                                break

                            formatted_line = (
                                f"[{self._command_name.upper()}] {line.rstrip()}"
                            )
                            if self.display_output:
                                print(formatted_line, flush=True)
                            # Write to log file
                            with open(self._log_path, "a", encoding="utf-8") as f:
                                f.write(formatted_line + "\n")
                                f.flush()
                        except (OSError, ValueError):
                            # File descriptor is closed or invalid
                            break
                        except Exception as e:
                            self._logger.warning(f"Line processing error: {e}")
                            break
                    else:
                        # No data available, check if process is done
                        if self._proc.poll() is not None:
                            break

            except Exception as e:
                self._logger.warning(f"Output processing error: {e}")

        self._output_thread = threading.Thread(target=process_output, daemon=True)
        self._output_thread.start()

    def _remove_directory(self, path: str) -> None:
        """Remove a directory."""
        try:
            shutil.rmtree(path, ignore_errors=True)
        except (OSError, IOError) as e:
            self._logger.warning("Warning: Failed to remove directory %s: %s", path, e)

    def _check_ports(self, timeout):
        elapsed = 0.0
        for port in self.health_check_ports:
            elapsed += self._check_port(port, timeout - elapsed)
        return elapsed

    def _check_port(self, port, timeout=30, sleep=0.1):
        """Check if a port is open on localhost."""
        start_time = time.time()
        self._logger.info("Checking Port: %s", port)
        elapsed = 0.0
        while elapsed < timeout:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) == 0:
                    self._logger.info("SUCCESS: Check Port: %s", port)
                    return time.time() - start_time
            time.sleep(sleep)
            elapsed = time.time() - start_time
        self._logger.error("FAILED: Check Port: %s", port)
        raise RuntimeError("FAILED: Check Port: %s" % port)

    def _check_urls(self, timeout):
        elapsed = 0.0
        for url in self.health_check_urls:
            elapsed += self._check_url(url, timeout - elapsed)
        return elapsed

    def _check_url(self, url, timeout=30, sleep=0.1):
        if isinstance(url, tuple):
            response_check = url[1]
            url = url[0]
        else:
            response_check = None
        start_time = time.time()
        self._logger.info("Checking URL %s", url)
        elapsed = 0.0
        while elapsed < timeout:
            try:
                response = requests.get(url, timeout=timeout - elapsed)
                if response.status_code == 200:
                    if response_check is None or response_check(response):
                        self._logger.info("SUCCESS: Check URL: %s", url)
                        return time.time() - start_time
            except requests.RequestException as e:
                self._logger.warning("URL check failed: %s", e)
            time.sleep(sleep)
            elapsed = time.time() - start_time

        self._logger.error("FAILED: Check URL: %s", url)
        raise RuntimeError("FAILED: Check URL: %s" % url)

    def _terminate_existing(self):
        if self.terminate_existing:
            for _proc in psutil.process_iter(["name", "cmdline"]):
                try:
                    if (
                        _proc.name() == self._command_name
                        or _proc.name() in self.stragglers
                    ):
                        self._logger.info(
                            "Terminating Existing %s %s", _proc.name(), _proc.pid
                        )

                        terminate_process_tree(_proc.pid, self._logger)
                    for cmdline in self.straggler_commands:
                        if cmdline in " ".join(_proc.cmdline()):
                            self._logger.info(
                                "Terminating Existing CmdLine %s %s %s",
                                _proc.name(),
                                _proc.pid,
                                _proc.cmdline(),
                            )
                            terminate_process_tree(_proc.pid, self._logger)
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    # Process may have terminated or become inaccessible during iteration
                    pass


def main():
    with ManagedProcess(
        command=[
            "dynamo",
            "run",
            "in=http",
            "out=vllm",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ],
        display_output=True,
        terminate_existing=True,
        health_check_ports=[8080],
        health_check_urls=["http://localhost:8080/v1/models"],
        timeout=10,
    ):
        time.sleep(60)
        pass


if __name__ == "__main__":
    main()
