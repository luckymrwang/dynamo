#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Simple SLA Planner Load Test Script
#
# This script sends concurrent requests to trigger SLA planner scaling.
# The SLA planner monitors metrics every 60 seconds and scales workers based on:
# - Time to First Token (TTFT)
# - Inter-Token Latency (ITL)
# - Request patterns
#
# Usage:
#   ./sla_planner_simple_load_test.sh [FRONTEND_URL] [CONCURRENT_REQUESTS] [DURATION_SECONDS]
#
# Example:
#   ./sla_planner_simple_load_test.sh http://localhost:8000 10 180

set -e

# Configuration
FRONTEND_URL="${1:-http://localhost:8000}"
CONCURRENT_REQUESTS="${2:-8}"
DURATION="${3:-180}"
LOG_DIR="./load_test_logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}    SLA Planner Load Test Script      ${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "Frontend URL: ${GREEN}$FRONTEND_URL${NC}"
echo -e "Concurrent requests: ${GREEN}$CONCURRENT_REQUESTS${NC}"
echo -e "Duration: ${GREEN}${DURATION}s${NC}"
echo -e "Log directory: ${GREEN}$LOG_DIR${NC}"
echo ""

# Test payload - IMPORTANT: streaming must be enabled for SLA planner metrics
read -r -d '' PAYLOAD << 'EOF' || true
{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
  "messages": [
    {
      "role": "user",
      "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
  ],
  "max_tokens": 30,
  "stream": true
}
EOF

# Function to check frontend health
check_health() {
    echo -e "${BLUE}Checking frontend health...${NC}"
    if curl -s -f "$FRONTEND_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend is accessible${NC}"
        return 0
    elif curl -s -f "$FRONTEND_URL/v1/models" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend is accessible (via models endpoint)${NC}"
        return 0
    else
        echo -e "${RED}✗ Frontend is not accessible at $FRONTEND_URL${NC}"
        echo -e "${YELLOW}Please ensure the frontend is running and accessible${NC}"
        return 1
    fi
}

# Function to send a single request
send_request() {
    local request_id=$1
    local log_file="$LOG_DIR/request_${request_id}.log"
    local start_time=$(date +%s.%N)

    # Send streaming request and measure TTFT
    {
        echo "Request $request_id started at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Start time: $start_time"

        # Use curl to send streaming request and capture timing
        curl -s -w "HTTP_CODE:%{http_code}\nTIME_TOTAL:%{time_total}\nTIME_CONNECT:%{time_connect}\nTIME_STARTTRANSFER:%{time_starttransfer}\n" \
             -H "Content-Type: application/json" \
             -d "$PAYLOAD" \
             "$FRONTEND_URL/v1/chat/completions" \
             2>&1

        echo "Request $request_id completed at $(date '+%Y-%m-%d %H:%M:%S')"
    } > "$log_file" 2>&1 &

    echo $!
}

# Function to monitor active requests
monitor_requests() {
    local pids=("$@")
    local active=0

    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((active++))
        fi
    done

    echo $active
}

# Function to analyze results
analyze_results() {
    echo -e "\n${BLUE}Analyzing results...${NC}"

    local total_requests=0
    local successful_requests=0
    local failed_requests=0
    local total_time=0
    local ttft_times=()

    for log_file in "$LOG_DIR"/request_*.log; do
        if [[ -f "$log_file" ]]; then
            ((total_requests++))

            # Extract metrics from log file
            if grep -q "HTTP_CODE:200" "$log_file"; then
                ((successful_requests++))

                # Extract timing information
                local time_total=$(grep "TIME_TOTAL:" "$log_file" | cut -d: -f2)
                local time_starttransfer=$(grep "TIME_STARTTRANSFER:" "$log_file" | cut -d: -f2)

                if [[ -n "$time_total" && -n "$time_starttransfer" ]]; then
                    total_time=$(echo "$total_time + $time_total" | bc -l 2>/dev/null || echo "$total_time")
                    ttft_times+=("$time_starttransfer")
                fi
            else
                ((failed_requests++))
            fi
        fi
    done

    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}         TEST RESULTS SUMMARY         ${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo -e "Total requests: ${GREEN}$total_requests${NC}"
    echo -e "Successful: ${GREEN}$successful_requests${NC}"
    echo -e "Failed: ${RED}$failed_requests${NC}"

    if [[ $total_requests -gt 0 ]]; then
        local success_rate=$(echo "scale=1; $successful_requests * 100 / $total_requests" | bc -l 2>/dev/null || echo "0")
        echo -e "Success rate: ${GREEN}${success_rate}%${NC}"
    fi

    if [[ $successful_requests -gt 0 && -n "$total_time" ]]; then
        local avg_duration=$(echo "scale=3; $total_time / $successful_requests" | bc -l 2>/dev/null || echo "0")
        echo -e "Average response time: ${GREEN}${avg_duration}s${NC}"
    fi

    # Calculate TTFT statistics if available
    if [[ ${#ttft_times[@]} -gt 0 ]]; then
        local ttft_sum=0
        local ttft_count=0
        for ttft in "${ttft_times[@]}"; do
            if [[ -n "$ttft" && "$ttft" != "0.000000" ]]; then
                ttft_sum=$(echo "$ttft_sum + $ttft" | bc -l 2>/dev/null || echo "$ttft_sum")
                ((ttft_count++))
            fi
        done

        if [[ $ttft_count -gt 0 ]]; then
            local avg_ttft=$(echo "scale=3; $ttft_sum / $ttft_count" | bc -l 2>/dev/null || echo "0")
            echo -e "Average TTFT: ${GREEN}${avg_ttft}s${NC}"
        fi
    fi

    echo -e "\n${YELLOW}SLA Planner Scaling Notes:${NC}"
    echo -e "• The planner adjusts workers every ~60 seconds"
    echo -e "• Monitor with: kubectl logs deployment/vllm-disagg-planner-planner"
    echo -e "• Check scaling: kubectl get pods | grep vllm-disagg-planner"
    echo -e "• High TTFT/ITL should trigger scaling decisions"
    echo -e "• Detailed logs available in: $LOG_DIR"
    echo -e "${BLUE}======================================${NC}"
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}

# Set up signal handling
trap cleanup EXIT INT TERM

# Main execution
main() {
    # Check if frontend is accessible
    if ! check_health; then
        exit 1
    fi

    # Clean previous logs
    rm -f "$LOG_DIR"/request_*.log

    echo -e "\n${BLUE}Starting load test...${NC}"
    echo -e "${YELLOW}NOTE: Using streaming requests (required for SLA planner metrics)${NC}"
    echo -e "${YELLOW}NOTE: SLA planner adjusts every ~60 seconds based on collected metrics${NC}"
    echo ""

    local start_time=$(date +%s)
    local end_time=$((start_time + DURATION))
    local request_counter=0
    local pids=()

    # Main load generation loop
    while [[ $(date +%s) -lt $end_time ]]; do
        # Maintain concurrent requests
        local active_requests=$(monitor_requests "${pids[@]}")

        # Start new requests if needed
        while [[ $active_requests -lt $CONCURRENT_REQUESTS ]]; do
            ((request_counter++))
            local pid=$(send_request $request_counter)
            pids+=($pid)
            ((active_requests++))

            # Brief delay to avoid overwhelming the system
            sleep 0.1
        done

        # Clean up completed processes
        local new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=($pid)
            fi
        done
        pids=("${new_pids[@]}")

        # Status update every 10 seconds
        local current_time=$(date +%s)
        if (( (current_time - start_time) % 10 == 0 )); then
            local elapsed=$((current_time - start_time))
            local remaining=$((end_time - current_time))
            echo -e "${BLUE}[${elapsed}s/${DURATION}s]${NC} Active: ${GREEN}${active_requests}${NC}, Total sent: ${GREEN}${request_counter}${NC}, Remaining: ${YELLOW}${remaining}s${NC}"
        fi

        sleep 1
    done

    # Wait for remaining requests to complete
    echo -e "\n${YELLOW}Waiting for remaining requests to complete...${NC}"
    wait

    # Analyze results
    analyze_results
}

# Check dependencies
if ! command -v curl >/dev/null 2>&1; then
    echo -e "${RED}Error: curl is required but not installed${NC}"
    exit 1
fi

if ! command -v bc >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: bc is not installed, some calculations may be skipped${NC}"
fi

# Run main function
main "$@"