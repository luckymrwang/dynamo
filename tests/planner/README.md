# SLA Planner Load Test Script

This script generates concurrent requests to test the SLA planner's scaling behavior.
The planner monitors metrics every 60 seconds (default adjustment interval) and scales
prefill/decode workers based on TTFT, ITL, and request patterns.

## Key features
- Sends streaming requests (required for planner metrics)
- Configurable load patterns (burst, sustained, ramp-up)
- Real-time monitoring of responses and metrics
- Logs scaling indicators

## Usage

```bash
python sla_planner_load_test.py --pattern sustained --concurrent-users 10 --duration 300
```
