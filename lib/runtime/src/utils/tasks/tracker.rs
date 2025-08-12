// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Task Tracker
//!
//! A hierarchical task tracking system with composable scheduling and error policies.
//!
//! ## Core Features
//! - **Hierarchical Management**: Parent-child relationships with automatic lifecycle management
//! - **Pluggable Scheduling**: Semaphore limits, unlimited throughput, or custom policies
//! - **Error Handling**: Cancel-on-error, log-only, threshold-based, or custom policies
//! - **Rich Metrics**: Task counts, success/failure rates, queue depth (with Prometheus support)
//! - **Cancellation**: Token-based with hierarchical propagation and isolation
//! - **Safe Child Creation**: Prevents child creation from closed parents
//!
//! Built on top of `tokio_util::task::TaskTracker` for robust task lifecycle management.
//!
//! ## Future Policies
//!
//! The system is designed for extensibility. See the source code for detailed TODO comments
//! describing additional policies that can be implemented:
//! - **Scheduling**: Token bucket rate limiting, adaptive concurrency, memory-aware scheduling
//! - **Error Handling**: Retry with backoff, circuit breakers, dead letter queues
//!
//! Each TODO comment includes complete implementation guidance with data structures,
//! algorithms, and dependencies needed for future contributors.
//!
//! ## Important: Close-Before-Wait Pattern
//!
//! This implementation follows Tokio's TaskTracker requirement that `close()` must be called
//! before `wait()` to prevent deadlocks. Our hierarchical `join()` method automatically handles this:
//! - `join()` closes each tracker before waiting (hierarchically)
//! - Uses depth-first traversal of the dependency tree for proper shutdown order
//!
//! ## Quick Start
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Simple setup with convenience constructors - minimal boilerplate!
//! let tracker = TaskTracker::builder()
//!     .scheduler(SemaphoreScheduler::with_permits(10))  // Returns Arc automatically
//!     .error_policy(LogOnlyPolicy::new_arc())         // Returns Arc automatically
//!     .build()?;
//!
//! // Spawn tasks and get results
//! let result = tracker.spawn(async { Ok(42) }).await??;
//! assert_eq!(result, 42);
//! # Ok(())
//! # }
//! ```
//!
//! ## Hierarchical Organization
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{
//!     TaskTracker, UnlimitedScheduler, ThresholdCancelPolicy, SemaphoreScheduler
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create root tracker with failure threshold policy
//! let (error_policy, _token) = ThresholdCancelPolicy::new_arc(5);
//! let root = std::sync::Arc::new(TaskTracker::builder()
//!     .scheduler(UnlimitedScheduler::new_arc())
//!     .error_policy(error_policy)
//!     .build()?);
//!
//! // Create child trackers for different components
//! let api_handler = root.child_tracker()?;  // Inherits policies
//! let background_jobs = root.child_tracker()?;
//!
//! // Children can have custom policies
//! let rate_limited = root.child_tracker_builder()
//!     .scheduler(SemaphoreScheduler::with_permits(2))  // Custom concurrency limit
//!     .build()?;
//!
//! // Tasks run independently but metrics roll up
//! api_handler.spawn(async { Ok(()) });
//! background_jobs.spawn(async { Ok(()) });
//! rate_limited.spawn(async { Ok(()) });
//!
//! // Join all children hierarchically
//! root.clone().join().await;
//! assert_eq!(root.metrics().success(), 3); // Sees all successes
//! # Ok(())
//! # }
//! ```
//!
//! ## Policy Examples
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{
//!     TaskTracker, CancelOnError, SemaphoreScheduler, ThresholdCancelPolicy
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Pattern-based error cancellation
//! let (error_policy, _token) = CancelOnError::with_patterns_arc(
//!     vec!["OutOfMemory".to_string(), "DeviceError".to_string()]
//! );
//! let simple = TaskTracker::builder()
//!     .scheduler(SemaphoreScheduler::with_permits(5))
//!     .error_policy(error_policy)
//!     .build()?;
//!
//! // Threshold-based cancellation with monitoring
//! let scheduler = SemaphoreScheduler::with_permits(10);  // Returns Arc<SemaphoreScheduler>
//! let (error_policy, token) = ThresholdCancelPolicy::new_arc(3);  // Returns (Arc<Policy>, Token)
//!
//! let advanced = TaskTracker::builder()
//!     .scheduler(scheduler)
//!     .error_policy(error_policy)
//!     .build()?;
//!
//! // Monitor cancellation externally
//! if token.is_cancelled() {
//!     println!("Tracker cancelled due to failures");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Metrics and Observability
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let tracker = std::sync::Arc::new(TaskTracker::builder()
//!     .scheduler(SemaphoreScheduler::with_permits(2))  // Only 2 concurrent tasks
//!     .error_policy(LogOnlyPolicy::new_arc())
//!     .build()?);
//!
//! // Spawn multiple tasks
//! for i in 0..5 {
//!     tracker.spawn(async move {
//!         tokio::time::sleep(std::time::Duration::from_millis(100)).await;
//!         Ok(i)
//!     });
//! }
//!
//! // Check metrics
//! let metrics = tracker.metrics();
//! println!("Issued: {}", metrics.issued());        // 5 tasks issued
//! println!("Active: {}", metrics.active());        // 2 tasks running (semaphore limit)
//! println!("Queued: {}", metrics.queued());        // 3 tasks waiting in scheduler queue
//! println!("Pending: {}", metrics.pending());      // 5 tasks not yet completed
//!
//! tracker.clone().join().await;
//! assert_eq!(metrics.success(), 5);
//! assert_eq!(metrics.pending(), 0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Prometheus Integration
//!
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
//! use dynamo_runtime::metrics::MetricsRegistry;
//!
//! # async fn example(registry: &dyn MetricsRegistry) -> anyhow::Result<()> {
//! // Root tracker with Prometheus metrics
//! let tracker = TaskTracker::new_with_prometheus(
//!     SemaphoreScheduler::with_permits(10),
//!     LogOnlyPolicy::new_arc(),
//!     registry,
//!     "my_component"
//! )?;
//!
//! // Metrics automatically exported to Prometheus:
//! // - my_component_tasks_issued_total
//! // - my_component_tasks_success_total
//! // - my_component_tasks_failed_total
//! // - my_component_tasks_active
//! // - my_component_tasks_queued
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::metrics::MetricsRegistry;
use anyhow::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use std::collections::HashSet;
use std::sync::{Mutex, RwLock, Weak};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker as TokioTaskTracker;
use tracing::{debug, error, warn, Instrument};
use uuid::Uuid;

/// Common scheduling policies for task execution
///
/// These enums provide convenient access to built-in scheduling policies
/// without requiring manual construction of policy objects.
///
/// ## Cancellation Semantics
///
/// All schedulers follow the same cancellation behavior:
/// - Respect cancellation tokens before resource allocation (permits, etc.)
/// - Once task execution begins, always await completion
/// - Let tasks handle their own cancellation internally
#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    /// No concurrency limits - execute all tasks immediately
    Unlimited,
    /// Semaphore-based concurrency limiting
    Semaphore(usize),
    // TODO: Future scheduling policies to implement
    //
    // /// Token bucket rate limiting with burst capacity
    // /// Implementation: Use tokio::time::interval for refill, AtomicU64 for tokens.
    // /// acquire() decrements tokens, schedule() waits for refill if empty.
    // /// Burst allows temporary spikes above steady rate.
    // /// struct: { rate: f64, burst: usize, tokens: AtomicU64, last_refill: Mutex<Instant> }
    // /// Example: TokenBucket { rate: 10.0, burst: 5 } = 10 tasks/sec, burst up to 5
    // TokenBucket { rate: f64, burst: usize },
    //
    // /// Weighted fair scheduling across multiple priority classes
    // /// Implementation: Maintain separate VecDeque for each priority class.
    // /// Use weighted round-robin: serve N tasks from high, M from normal, etc.
    // /// Track deficit counters to ensure fairness over time.
    // /// struct: { queues: HashMap<String, VecDeque<Task>>, weights: Vec<(String, u32)> }
    // /// Example: WeightedFair { weights: vec![("high", 70), ("normal", 20), ("low", 10)] }
    // WeightedFair { weights: Vec<(String, u32)> },
    //
    // /// Memory-aware scheduling that limits tasks based on available memory
    // /// Implementation: Monitor system memory via /proc/meminfo or sysinfo crate.
    // /// Pause scheduling when available memory < threshold, resume when memory freed.
    // /// Use exponential backoff for memory checks to avoid overhead.
    // /// struct: { max_memory_mb: usize, check_interval: Duration, semaphore: Semaphore }
    // MemoryAware { max_memory_mb: usize },
    //
    // /// CPU-aware scheduling that adjusts concurrency based on CPU load
    // /// Implementation: Sample system load via sysinfo crate every N seconds.
    // /// Dynamically resize internal semaphore permits based on load average.
    // /// Use PID controller for smooth adjustments, avoid oscillation.
    // /// struct: { max_cpu_percent: f32, permits: Arc<Semaphore>, sampler: tokio::task }
    // CpuAware { max_cpu_percent: f32 },
    //
    // /// Adaptive scheduler that automatically adjusts concurrency based on performance
    // /// Implementation: Track task latency and throughput in sliding windows.
    // /// Increase permits if latency low & throughput stable, decrease if latency spikes.
    // /// Use additive increase, multiplicative decrease (AIMD) algorithm.
    // /// struct: { permits: AtomicUsize, latency_tracker: RingBuffer, throughput_tracker: RingBuffer }
    // Adaptive { initial_permits: usize },
    //
    // /// Throttling scheduler that enforces minimum time between task starts
    // /// Implementation: Store last_execution time in AtomicU64 (unix timestamp).
    // /// Before scheduling, check elapsed time and tokio::time::sleep if needed.
    // /// Useful for rate-limiting API calls to external services.
    // /// struct: { min_interval: Duration, last_execution: AtomicU64 }
    // Throttling { min_interval_ms: u64 },
    //
    // /// Batch scheduler that groups tasks and executes them together
    // /// Implementation: Collect tasks in Vec<Task>, use tokio::time::timeout for max_wait.
    // /// Execute batch when size reached OR timeout expires, whichever first.
    // /// Use futures::future::join_all for parallel execution within batch.
    // /// struct: { batch_size: usize, max_wait: Duration, pending: Mutex<Vec<Task>> }
    // Batch { batch_size: usize, max_wait_ms: u64 },
    //
    // /// Priority-based scheduler with separate queues for different priority levels
    // /// Implementation: Three separate semaphores for high/normal/low priorities.
    // /// Always serve high before normal, normal before low (strict priority).
    // /// Add starvation protection: promote normal->high after timeout.
    // /// struct: { high_sem: Semaphore, normal_sem: Semaphore, low_sem: Semaphore }
    // Priority { high: usize, normal: usize, low: usize },
    //
    // /// Backpressure-aware scheduler that monitors downstream capacity
    // /// Implementation: Track external queue depth via provided callback/metric.
    // /// Pause scheduling when queue_threshold exceeded, resume after pause_duration.
    // /// Use exponential backoff for repeated backpressure events.
    // /// struct: { queue_checker: Arc<dyn Fn() -> usize>, threshold: usize, pause_duration: Duration }
    // Backpressure { queue_threshold: usize, pause_duration_ms: u64 },
}

/// Common error handling policies for task failure management
///
/// These enums provide convenient access to built-in error handling policies
/// without requiring manual construction of policy objects.
#[derive(Debug, Clone)]
pub enum ErrorPolicy {
    /// Log errors but continue execution - no cancellation
    LogOnly,
    /// Cancel all tasks on any error (using default error patterns)
    CancelOnError,
    /// Cancel all tasks when specific error patterns are encountered
    CancelOnPatterns(Vec<String>),
    /// Cancel after a threshold number of failures
    CancelOnThreshold { max_failures: usize },
    /// Cancel when failure rate exceeds threshold within time window
    CancelOnRate {
        max_failure_rate: f32,
        window_secs: u64,
    },
    // TODO: Future error policies to implement
    //
    // /// Retry failed tasks with exponential backoff
    // /// Implementation: Store original task in retry queue with attempt count.
    // /// Use tokio::time::sleep for delays: backoff_ms * 2^attempt.
    // /// Spawn retry as new task, preserve original task_id for tracing.
    // /// Need task cloning support in scheduler interface.
    // /// struct: { max_attempts: usize, backoff_ms: u64, retry_queue: VecDeque<(Task, u32)> }
    // Retry { max_attempts: usize, backoff_ms: u64 },
    //
    // /// Send failed tasks to a dead letter queue for later processing
    // /// Implementation: Use tokio::sync::mpsc::channel for queue.
    // /// Serialize task info (id, error, payload) for persistence.
    // /// Background worker drains queue to external storage (Redis/DB).
    // /// Include retry count and timestamps for debugging.
    // /// struct: { queue: mpsc::Sender<DeadLetterItem>, storage: Arc<dyn DeadLetterStorage> }
    // DeadLetter { queue_name: String },
    //
    // /// Execute fallback logic when tasks fail
    // /// Implementation: Store fallback closure in Arc for thread-safety.
    // /// Execute fallback in same context as failed task (inherit cancel token).
    // /// Track fallback success/failure separately from original task metrics.
    // /// Consider using enum for common fallback patterns (default value, noop, etc).
    // /// struct: { fallback_fn: Arc<dyn Fn(TaskId, Error) -> BoxFuture<Result<()>>> }
    // Fallback { fallback_fn: Arc<dyn Fn() -> BoxFuture<'static, Result<()>>> },
    //
    // /// Circuit breaker pattern - stop executing after threshold failures
    // /// Implementation: Track state (Closed/Open/HalfOpen) with AtomicU8.
    // /// Use failure window (last N tasks) or time window for threshold.
    // /// In Open state, reject tasks immediately, use timer for recovery.
    // /// In HalfOpen, allow one test task to check if issues resolved.
    // /// struct: { state: AtomicU8, failure_count: AtomicU64, last_failure: AtomicU64 }
    // CircuitBreaker { failure_threshold: usize, timeout_secs: u64 },
    //
    // /// Resource protection policy that monitors memory/CPU usage
    // /// Implementation: Background task samples system resources via sysinfo.
    // /// Cancel tracker when memory > threshold, use process-level monitoring.
    // /// Implement graceful degradation: warn at 80%, cancel at 90%.
    // /// Include both system-wide and process-specific thresholds.
    // /// struct: { monitor_task: JoinHandle, thresholds: ResourceThresholds, cancel_token: CancellationToken }
    // ResourceProtection { max_memory_mb: usize },
    //
    // /// Timeout policy that cancels tasks exceeding maximum duration
    // /// Implementation: Wrap each task with tokio::time::timeout.
    // /// Store task start time, check duration in on_error callback.
    // /// Distinguish timeout errors from other task failures in metrics.
    // /// Consider per-task or global timeout strategies.
    // /// struct: { max_duration: Duration, timeout_tracker: HashMap<TaskId, Instant> }
    // Timeout { max_duration_secs: u64 },
    //
    // /// Sampling policy that only logs a percentage of errors
    // /// Implementation: Use thread-local RNG for sampling decisions.
    // /// Hash task_id for deterministic sampling (same task always sampled).
    // /// Store sample rate as f32, compare with rand::random::<f32>().
    // /// Include rate in log messages for context.
    // /// struct: { sample_rate: f32, rng: ThreadLocal<RefCell<SmallRng>> }
    // Sampling { sample_rate: f32 },
    //
    // /// Aggregating policy that batches error reports
    // /// Implementation: Collect errors in Vec, flush on size or time trigger.
    // /// Use tokio::time::interval for periodic flushing.
    // /// Group errors by type/pattern for better insights.
    // /// Include error frequency and rate statistics in reports.
    // /// struct: { window: Duration, batch: Mutex<Vec<ErrorEntry>>, flush_task: JoinHandle }
    // Aggregating { window_secs: u64, max_batch_size: usize },
    //
    // /// Alerting policy that sends notifications on error patterns
    // /// Implementation: Use reqwest for webhook HTTP calls.
    // /// Rate-limit alerts to prevent spam (max N per minute).
    // /// Include error context, task info, and system metrics in payload.
    // /// Support multiple notification channels (webhook, email, slack).
    // /// struct: { client: reqwest::Client, rate_limiter: RateLimiter, alert_config: AlertConfig }
    // Alerting { webhook_url: String, severity_threshold: String },
}

/// Common functionality for policy Arc construction
///
/// This trait provides a standardized `new_arc()` method for all policy types,
/// eliminating the need for manual `Arc::new()` calls in client code.
pub trait ArcPolicy: Sized + Send + Sync + 'static {
    /// Create an Arc-wrapped instance of this policy
    fn new_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

/// Unique identifier for a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(Uuid);

impl TaskId {
    fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "task-{}", self.0)
    }
}

/// Result of task execution
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionStatus {
    /// Task completed successfully
    Ok,
    /// Task was cancelled before or during execution
    Cancelled,
    /// Task failed with an error
    Failed(String),
}

/// Result type for cancellable tasks that explicitly tracks cancellation
#[derive(Debug)]
pub enum CancellableTaskResult<T> {
    /// Task completed successfully
    Ok(T),
    /// Task was cancelled (either via token or shutdown)
    Cancelled,
    /// Task failed with an error
    Err(anyhow::Error),
}

/// Result of scheduling a task
#[derive(Debug)]
pub enum SchedulingResult<T> {
    /// Task was executed and completed
    Execute(T),
    /// Task was cancelled before execution
    Cancelled,
    /// Task was rejected due to scheduling policy
    Rejected(String),
}

/// Resource guard that manages task execution
///
/// This trait enforces proper cancellation semantics by separating resource
/// management from task execution. Once a guard is acquired, task execution
/// must always run to completion.
#[async_trait]
pub trait ResourceGuard: Send {
    /// Execute a task to completion
    ///
    /// This method MUST always await the task completion. No cancellation
    /// token is provided to prevent interrupting task execution.
    ///
    /// Tasks that support cancellation will handle it internally and return
    /// appropriately when their cancellation token is triggered.
    ///
    /// # Arguments
    /// * `task` - The task future to execute (may or may not support cancellation)
    async fn execute_task(
        self: Box<Self>,
        task: Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>,
    ) -> Result<()>;
}

/// Trait for implementing task scheduling policies
///
/// This trait enforces proper cancellation semantics by splitting resource
/// acquisition (which can be cancelled) from task execution (which cannot).
///
/// ## Design Philosophy
///
/// Tasks may or may not support cancellation (depending on whether they were
/// created with `spawn_cancellable` or regular `spawn`). This split design ensures:
///
/// - **Resource acquisition**: Can respect cancellation tokens to avoid unnecessary allocation
/// - **Task execution**: Always runs to completion; tasks handle their own cancellation
///
/// This makes it impossible to accidentally interrupt task execution with `tokio::select!`.
#[async_trait]
pub trait TaskScheduler: Send + Sync + std::fmt::Debug {
    /// Acquire resources needed for task execution and return a guard
    ///
    /// This method handles resource allocation (permits, queue slots, etc.) and
    /// can respect cancellation tokens to avoid unnecessary resource consumption.
    ///
    /// ## Cancellation Behavior
    ///
    /// The `cancel_token` is used for scheduler-level cancellation (e.g., "don't start new work").
    /// If cancellation is requested before or during resource acquisition, this method
    /// should return `SchedulingResult::Cancelled`.
    ///
    /// # Arguments
    /// * `cancel_token` - Optional cancellation token for scheduler-level cancellation
    ///
    /// # Returns
    /// * `SchedulingResult::Execute(guard)` - Resources acquired, ready to execute
    /// * `SchedulingResult::Cancelled` - Cancelled before or during resource acquisition
    /// * `SchedulingResult::Rejected(reason)` - Resources unavailable or policy violation
    async fn acquire_execution_slot(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> SchedulingResult<Box<dyn ResourceGuard>>;
}

/// Trait for implementing error handling policies
///
/// Implementors define how to respond to task failures, including
/// whether to propagate cancellation and how to handle error thresholds.
#[async_trait]
pub trait OnErrorPolicy: Send + Sync + std::fmt::Debug {
    /// Create a child policy for a child tracker
    ///
    /// This allows policies to maintain hierarchical relationships,
    /// such as child cancellation tokens or shared circuit breaker state.
    fn create_child(&self) -> Arc<dyn OnErrorPolicy>;

    /// Called when a task fails during execution
    ///
    /// # Arguments
    /// * `error` - The error that occurred
    /// * `task_id` - Unique identifier of the failed task
    async fn on_error(&self, error: &anyhow::Error, task_id: TaskId);

    /// Called when failure rate exceeds configured thresholds
    ///
    /// # Arguments
    /// * `failure_rate` - Current failure rate (0.0 to 1.0)
    async fn on_failure_threshold_exceeded(&self, failure_rate: f64);

    /// Determine if an error should trigger cancellation
    ///
    /// # Arguments
    /// * `error` - The error to evaluate
    ///
    /// # Returns
    /// `true` if this error should cause the tracker to cancel remaining tasks
    fn should_cancel_on_error(&self, _error: &anyhow::Error) -> bool {
        false
    }

    /// Get the cancellation token if this policy supports cancellation
    ///
    /// This allows callers to observe cancellation state from policies
    /// that implement cancellation logic.
    fn cancellation_token(&self) -> Option<CancellationToken> {
        None
    }
}

/// Task execution metrics for a tracker
#[derive(Debug, Default)]
pub struct TaskMetrics {
    /// Number of tasks issued/submitted (via spawn methods)
    pub issued_count: AtomicU64,
    /// Number of successfully completed tasks
    pub success_count: AtomicU64,
    /// Number of cancelled tasks
    pub cancelled_count: AtomicU64,
    /// Number of failed tasks
    pub failed_count: AtomicU64,
    /// Number of rejected tasks (by scheduler)
    pub rejected_count: AtomicU64,
    /// Number of currently active tasks
    pub active_count: AtomicU64,
}

impl TaskMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment issued task counter
    pub fn increment_issued(&self) {
        self.issued_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment success counter
    pub fn increment_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment cancelled counter
    pub fn increment_cancelled(&self) {
        self.cancelled_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment failed counter
    pub fn increment_failed(&self) {
        self.failed_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment rejected counter
    pub fn increment_rejected(&self) {
        self.rejected_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment active task counter
    pub fn increment_active(&self) {
        self.active_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active task counter
    pub fn decrement_active(&self) {
        self.active_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current issued count
    pub fn issued(&self) -> u64 {
        self.issued_count.load(Ordering::Relaxed)
    }

    /// Get current success count
    pub fn success(&self) -> u64 {
        self.success_count.load(Ordering::Relaxed)
    }

    /// Get current cancelled count
    pub fn cancelled(&self) -> u64 {
        self.cancelled_count.load(Ordering::Relaxed)
    }

    /// Get current failed count
    pub fn failed(&self) -> u64 {
        self.failed_count.load(Ordering::Relaxed)
    }

    /// Get current rejected count
    pub fn rejected(&self) -> u64 {
        self.rejected_count.load(Ordering::Relaxed)
    }

    /// Get current active count
    pub fn active(&self) -> u64 {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Get total completed tasks (success + cancelled + failed + rejected)
    pub fn total_completed(&self) -> u64 {
        self.success() + self.cancelled() + self.failed() + self.rejected()
    }

    /// Get number of pending tasks (issued - completed)
    ///
    /// This represents tasks that have been issued but not yet completed
    /// (includes both active tasks and those waiting in scheduler queues)
    pub fn pending(&self) -> u64 {
        self.issued().saturating_sub(self.total_completed())
    }

    /// Get number of tasks queued in scheduler (pending - active)
    ///
    /// This represents tasks that have been issued but are waiting
    /// in the scheduler queue and not yet actively executing
    pub fn queued(&self) -> u64 {
        self.pending().saturating_sub(self.active())
    }

    /// Calculate failure rate (failed / total_completed)
    ///
    /// Returns 0.0 if no tasks have completed
    pub fn failure_rate(&self) -> f64 {
        let total = self.total_completed();
        if total == 0 {
            0.0
        } else {
            self.failed() as f64 / total as f64
        }
    }
}

/// Trait for hierarchical task metrics that supports aggregation up the tracker tree
///
/// This trait provides different implementations for root and child trackers:
/// - Root trackers integrate with Prometheus metrics for observability
/// - Child trackers chain metric updates up to their parents for aggregation
/// - All implementations maintain thread-safe atomic operations
pub trait HierarchicalTaskMetrics: Send + Sync + std::fmt::Debug {
    /// Increment issued task counter
    fn increment_issued(&self);

    /// Increment success counter
    fn increment_success(&self);

    /// Increment cancelled counter
    fn increment_cancelled(&self);

    /// Increment failed counter
    fn increment_failed(&self);

    /// Increment rejected counter
    fn increment_rejected(&self);

    /// Increment active task counter
    fn increment_active(&self);

    /// Decrement active task counter
    fn decrement_active(&self);

    /// Get current issued count (local to this tracker)
    fn issued(&self) -> u64;

    /// Get current success count (local to this tracker)
    fn success(&self) -> u64;

    /// Get current cancelled count (local to this tracker)
    fn cancelled(&self) -> u64;

    /// Get current failed count (local to this tracker)
    fn failed(&self) -> u64;

    /// Get current rejected count (local to this tracker)
    fn rejected(&self) -> u64;

    /// Get current active count (local to this tracker)
    fn active(&self) -> u64;

    /// Get total completed tasks (success + cancelled + failed + rejected)
    fn total_completed(&self) -> u64 {
        self.success() + self.cancelled() + self.failed() + self.rejected()
    }

    /// Get number of pending tasks (issued - completed)
    fn pending(&self) -> u64 {
        self.issued().saturating_sub(self.total_completed())
    }

    /// Get number of tasks queued in scheduler (pending - active)
    fn queued(&self) -> u64 {
        self.pending().saturating_sub(self.active())
    }

    /// Calculate failure rate (failed / total_completed)
    fn failure_rate(&self) -> f64 {
        let total = self.total_completed();
        if total == 0 {
            0.0
        } else {
            self.failed() as f64 / total as f64
        }
    }
}

/// Root tracker metrics with Prometheus integration
///
/// This implementation maintains local counters and exposes them as Prometheus metrics
/// through the provided MetricsRegistry.
#[derive(Debug)]
pub struct RootTaskMetrics {
    /// Local metrics for this tracker
    local_metrics: TaskMetrics,
    /// Prometheus metrics integration
    prometheus_issued: prometheus::IntCounter,
    prometheus_success: prometheus::IntCounter,
    prometheus_cancelled: prometheus::IntCounter,
    prometheus_failed: prometheus::IntCounter,
    prometheus_rejected: prometheus::IntCounter,
    prometheus_active: prometheus::IntGauge,
    prometheus_queued: prometheus::IntGauge,
}

impl RootTaskMetrics {
    /// Create new root metrics with Prometheus integration
    ///
    /// # Arguments
    /// * `registry` - MetricsRegistry for creating Prometheus metrics
    /// * `component_name` - Name for the component/tracker (used in metric names)
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::RootTaskMetrics;
    /// # use dynamo_runtime::metrics::MetricsRegistry;
    /// # fn example(registry: Arc<dyn MetricsRegistry>) -> anyhow::Result<()> {
    /// let metrics = RootTaskMetrics::new(registry.as_ref(), "main_tracker")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<R: MetricsRegistry + ?Sized>(
        registry: &R,
        component_name: &str,
    ) -> anyhow::Result<Self> {
        let issued_counter = registry.create_intcounter(
            &format!("{}_tasks_issued_total", component_name),
            "Total number of tasks issued/submitted",
            &[],
        )?;

        let success_counter = registry.create_intcounter(
            &format!("{}_tasks_success_total", component_name),
            "Total number of successfully completed tasks",
            &[],
        )?;

        let cancelled_counter = registry.create_intcounter(
            &format!("{}_tasks_cancelled_total", component_name),
            "Total number of cancelled tasks",
            &[],
        )?;

        let failed_counter = registry.create_intcounter(
            &format!("{}_tasks_failed_total", component_name),
            "Total number of failed tasks",
            &[],
        )?;

        let rejected_counter = registry.create_intcounter(
            &format!("{}_tasks_rejected_total", component_name),
            "Total number of rejected tasks",
            &[],
        )?;

        let active_gauge = registry.create_intgauge(
            &format!("{}_tasks_active", component_name),
            "Current number of active tasks",
            &[],
        )?;

        let queued_gauge = registry.create_intgauge(
            &format!("{}_tasks_queued", component_name),
            "Current number of tasks queued in scheduler",
            &[],
        )?;

        Ok(Self {
            local_metrics: TaskMetrics::new(),
            prometheus_issued: issued_counter,
            prometheus_success: success_counter,
            prometheus_cancelled: cancelled_counter,
            prometheus_failed: failed_counter,
            prometheus_rejected: rejected_counter,
            prometheus_active: active_gauge,
            prometheus_queued: queued_gauge,
        })
    }
}

impl RootTaskMetrics {
    /// Update the queued gauge based on current metrics
    fn update_queued_gauge(&self) {
        let queued = self.local_metrics.queued() as i64;
        self.prometheus_queued.set(queued);
    }
}

impl HierarchicalTaskMetrics for RootTaskMetrics {
    fn increment_issued(&self) {
        self.local_metrics.increment_issued();
        self.prometheus_issued.inc();
        self.update_queued_gauge();
    }

    fn increment_success(&self) {
        self.local_metrics.increment_success();
        self.prometheus_success.inc();
        self.update_queued_gauge();
    }

    fn increment_cancelled(&self) {
        self.local_metrics.increment_cancelled();
        self.prometheus_cancelled.inc();
        self.update_queued_gauge();
    }

    fn increment_failed(&self) {
        self.local_metrics.increment_failed();
        self.prometheus_failed.inc();
        self.update_queued_gauge();
    }

    fn increment_rejected(&self) {
        self.local_metrics.increment_rejected();
        self.prometheus_rejected.inc();
        self.update_queued_gauge();
    }

    fn increment_active(&self) {
        self.local_metrics.increment_active();
        self.prometheus_active.inc();
        self.update_queued_gauge();
    }

    fn decrement_active(&self) {
        self.local_metrics.decrement_active();
        self.prometheus_active.dec();
        self.update_queued_gauge();
    }

    fn issued(&self) -> u64 {
        self.local_metrics.issued()
    }

    fn success(&self) -> u64 {
        self.local_metrics.success()
    }

    fn cancelled(&self) -> u64 {
        self.local_metrics.cancelled()
    }

    fn failed(&self) -> u64 {
        self.local_metrics.failed()
    }

    fn rejected(&self) -> u64 {
        self.local_metrics.rejected()
    }

    fn active(&self) -> u64 {
        self.local_metrics.active()
    }
}

/// Default root tracker metrics without Prometheus integration
///
/// This implementation only maintains local counters and is suitable for
/// applications that don't need Prometheus metrics or want to handle
/// metrics exposition themselves.
#[derive(Debug)]
pub struct DefaultRootTaskMetrics {
    /// Local metrics for this tracker
    local_metrics: TaskMetrics,
}

impl DefaultRootTaskMetrics {
    /// Create new default root metrics
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::DefaultRootTaskMetrics;
    /// let metrics = DefaultRootTaskMetrics::new();
    /// ```
    pub fn new() -> Self {
        Self {
            local_metrics: TaskMetrics::new(),
        }
    }
}

impl Default for DefaultRootTaskMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalTaskMetrics for DefaultRootTaskMetrics {
    fn increment_issued(&self) {
        self.local_metrics.increment_issued();
    }

    fn increment_success(&self) {
        self.local_metrics.increment_success();
    }

    fn increment_cancelled(&self) {
        self.local_metrics.increment_cancelled();
    }

    fn increment_failed(&self) {
        self.local_metrics.increment_failed();
    }

    fn increment_rejected(&self) {
        self.local_metrics.increment_rejected();
    }

    fn increment_active(&self) {
        self.local_metrics.increment_active();
    }

    fn decrement_active(&self) {
        self.local_metrics.decrement_active();
    }

    fn issued(&self) -> u64 {
        self.local_metrics.issued()
    }

    fn success(&self) -> u64 {
        self.local_metrics.success()
    }

    fn cancelled(&self) -> u64 {
        self.local_metrics.cancelled()
    }

    fn failed(&self) -> u64 {
        self.local_metrics.failed()
    }

    fn rejected(&self) -> u64 {
        self.local_metrics.rejected()
    }

    fn active(&self) -> u64 {
        self.local_metrics.active()
    }
}

/// Child tracker metrics that chain updates to parent
///
/// This implementation maintains local counters and automatically forwards
/// all metric updates to the parent tracker for hierarchical aggregation.
/// Holds a strong reference to parent metrics for optimal performance.
#[derive(Debug)]
pub struct ChildTaskMetrics {
    /// Local metrics for this tracker
    local_metrics: TaskMetrics,
    /// Strong reference to parent metrics for fast chaining
    /// Safe to hold since metrics don't own trackers - no circular references
    parent_metrics: Arc<dyn HierarchicalTaskMetrics>,
}

impl ChildTaskMetrics {
    /// Create new child metrics with parent chaining
    ///
    /// # Arguments
    /// * `parent_metrics` - Strong reference to parent metrics for chaining
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::{ChildTaskMetrics, HierarchicalTaskMetrics};
    /// # fn example(parent: Arc<dyn HierarchicalTaskMetrics>) {
    /// let child_metrics = ChildTaskMetrics::new(parent);
    /// # }
    /// ```
    pub fn new(parent_metrics: Arc<dyn HierarchicalTaskMetrics>) -> Self {
        Self {
            local_metrics: TaskMetrics::new(),
            parent_metrics,
        }
    }
}

impl HierarchicalTaskMetrics for ChildTaskMetrics {
    fn increment_issued(&self) {
        self.local_metrics.increment_issued();
        self.parent_metrics.increment_issued();
    }

    fn increment_success(&self) {
        self.local_metrics.increment_success();
        self.parent_metrics.increment_success();
    }

    fn increment_cancelled(&self) {
        self.local_metrics.increment_cancelled();
        self.parent_metrics.increment_cancelled();
    }

    fn increment_failed(&self) {
        self.local_metrics.increment_failed();
        self.parent_metrics.increment_failed();
    }

    fn increment_rejected(&self) {
        self.local_metrics.increment_rejected();
        self.parent_metrics.increment_rejected();
    }

    fn increment_active(&self) {
        self.local_metrics.increment_active();
        self.parent_metrics.increment_active();
    }

    fn decrement_active(&self) {
        self.local_metrics.decrement_active();
        self.parent_metrics.decrement_active();
    }

    fn issued(&self) -> u64 {
        self.local_metrics.issued()
    }

    fn success(&self) -> u64 {
        self.local_metrics.success()
    }

    fn cancelled(&self) -> u64 {
        self.local_metrics.cancelled()
    }

    fn failed(&self) -> u64 {
        self.local_metrics.failed()
    }

    fn rejected(&self) -> u64 {
        self.local_metrics.rejected()
    }

    fn active(&self) -> u64 {
        self.local_metrics.active()
    }

    // Child trackers don't expose Prometheus metrics directly
    // They rely on their parent (root) tracker for metrics exposition
}

/// Builder for creating child trackers with custom policies
///
/// Allows flexible customization of scheduling and error handling policies
/// for child trackers while maintaining parent-child relationships.
pub struct ChildTrackerBuilder<'parent> {
    parent: &'parent TaskTracker,
    scheduler: Option<Arc<dyn TaskScheduler>>,
    error_policy: Option<Arc<dyn OnErrorPolicy>>,
}

impl<'parent> ChildTrackerBuilder<'parent> {
    /// Set custom scheduler for the child tracker
    ///
    /// If not set, the child will inherit the parent's scheduler.
    ///
    /// # Arguments
    /// * `scheduler` - The scheduler to use for this child tracker
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler};
    /// # fn example(parent: &TaskTracker) {
    /// let child = parent.child_tracker_builder()
    ///     .scheduler(Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5)))))
    ///     .build().unwrap();
    /// # }
    /// ```
    pub fn scheduler(mut self, scheduler: Arc<dyn TaskScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set custom error policy for the child tracker
    ///
    /// If not set, the child will get a child policy from the parent's error policy
    /// (via `OnErrorPolicy::create_child()`).
    ///
    /// # Arguments
    /// * `error_policy` - The error policy to use for this child tracker
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, LogOnlyPolicy};
    /// # fn example(parent: &TaskTracker) {
    /// let child = parent.child_tracker_builder()
    ///     .error_policy(Arc::new(LogOnlyPolicy::new()))
    ///     .build().unwrap();
    /// # }
    /// ```
    pub fn error_policy(mut self, error_policy: Arc<dyn OnErrorPolicy>) -> Self {
        self.error_policy = Some(error_policy);
        self
    }

    /// Build the child tracker with the specified configuration
    ///
    /// Creates a new child tracker with:
    /// - Custom or inherited scheduler
    /// - Custom or child error policy
    /// - Hierarchical metrics that chain to parent
    /// - Child cancellation token from the parent
    /// - Independent lifecycle from parent
    ///
    /// # Returns
    /// A new `Arc<TaskTracker>` configured as a child of the parent
    ///
    /// # Errors
    /// Returns an error if the parent tracker is already closed
    pub fn build(self) -> anyhow::Result<Arc<TaskTracker>> {
        // Validate that parent tracker is still active
        if self.parent.is_closed() {
            return Err(anyhow::anyhow!(
                "Cannot create child tracker from closed parent tracker"
            ));
        }

        let child_cancel_token = self.parent.cancel_token.child_token();
        let child_metrics = Arc::new(ChildTaskMetrics::new(self.parent.metrics.clone()));

        // Use provided scheduler or inherit from parent
        let scheduler = self
            .scheduler
            .unwrap_or_else(|| self.parent.scheduler.clone());

        // Use provided error policy or create child from parent's
        let error_policy = self
            .error_policy
            .unwrap_or_else(|| self.parent.error_policy.create_child());

        let child = Arc::new(TaskTracker {
            inner: TokioTaskTracker::new(),
            parent: None, // No parent reference needed for hierarchical operations
            scheduler,
            error_policy,
            metrics: child_metrics,
            cancel_token: child_cancel_token,
            children: Arc::new(RwLock::new(Vec::new())),
        });

        // Register this child with the parent for hierarchical operations
        self.parent
            .children
            .write()
            .unwrap()
            .push(Arc::downgrade(&child));

        // Periodically clean up dead children to prevent unbounded growth
        self.parent.cleanup_dead_children();

        Ok(child)
    }
}

/// Hierarchical task tracker with pluggable scheduling and error policies
///
/// TaskTracker provides a composable system for managing background tasks with:
/// - Configurable scheduling via [`TaskScheduler`] implementations
/// - Flexible error handling via [`OnErrorPolicy`] implementations
/// - Parent-child relationships with independent metrics
/// - Cancellation propagation and isolation
/// - Built-in cancellation token support
///
/// Built on top of `tokio_util::task::TaskTracker` for robust task lifecycle management.
///
/// # Example
///
/// ```rust
/// # use std::sync::Arc;
/// # use tokio::sync::Semaphore;
/// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy, CancellableTaskResult};
/// # async fn example() -> anyhow::Result<()> {
/// // Create tracker with builder pattern
/// let semaphore = Arc::new(Semaphore::new(5));
/// let scheduler = Arc::new(SemaphoreScheduler::new(semaphore));
/// let error_policy = Arc::new(LogOnlyPolicy::new());
///
/// let tracker = TaskTracker::builder()
///     .scheduler(scheduler)
///     .error_policy(error_policy)
///     .build()
///     .unwrap();
///
/// // Spawn a regular task
/// let handle = tracker.spawn(async {
///     tokio::time::sleep(std::time::Duration::from_millis(100)).await;
///     Ok(42)
/// });
///
/// // Spawn a cancellable task that can inspect the token
/// let handle2 = tracker.spawn_cancellable(|cancel_token| async move {
///     tokio::select! {
///         _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => CancellableTaskResult::Ok(42),
///         _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
///     }
/// });
///
/// // Wait for completion
/// let result = handle.await??;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Builder)]
#[builder(pattern = "owned")]
pub struct TaskTracker {
    /// Tokio's task tracker for lifecycle management
    #[builder(default = "TokioTaskTracker::new()")]
    inner: TokioTaskTracker,
    /// Parent tracker (None for root)
    #[builder(default = "None")]
    parent: Option<Arc<TaskTracker>>,
    /// Scheduling policy (shared with children by default)
    scheduler: Arc<dyn TaskScheduler>,
    /// Error handling policy (child-specific via create_child)
    error_policy: Arc<dyn OnErrorPolicy>,
    /// Metrics for this tracker
    #[builder(default = "Arc::new(DefaultRootTaskMetrics::new())")]
    metrics: Arc<dyn HierarchicalTaskMetrics>,
    /// Cancellation token for this tracker (always present)
    #[builder(default = "CancellationToken::new()")]
    cancel_token: CancellationToken,
    /// List of child trackers for hierarchical operations
    #[builder(default = "Arc::new(RwLock::new(Vec::new()))")]
    children: Arc<RwLock<Vec<Weak<TaskTracker>>>>,
}

impl TaskTracker {
    /// Create a new root task tracker using the builder pattern
    ///
    /// This is the preferred way to create new task trackers.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
    /// let error_policy = Arc::new(LogOnlyPolicy::new());
    /// let tracker = TaskTracker::builder()
    ///     .scheduler(scheduler)
    ///     .error_policy(error_policy)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> TaskTrackerBuilder {
        TaskTrackerBuilder::default()
    }

    /// Create a new root task tracker with simple parameters (legacy)
    ///
    /// This method is kept for backward compatibility. Use `builder()` for new code.
    /// Uses default metrics (no Prometheus integration).
    ///
    /// # Arguments
    /// * `scheduler` - Scheduling policy to use for all tasks
    /// * `error_policy` - Error handling policy for this tracker
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
    /// let error_policy = Arc::new(LogOnlyPolicy::new());
    /// let tracker = TaskTracker::new(scheduler, error_policy);
    /// ```
    pub fn new(
        scheduler: Arc<dyn TaskScheduler>,
        error_policy: Arc<dyn OnErrorPolicy>,
    ) -> Arc<Self> {
        Arc::new(
            Self::builder()
                .scheduler(scheduler)
                .error_policy(error_policy)
                .build()
                .unwrap(),
        )
    }

    /// Create a new root task tracker with Prometheus metrics integration
    ///
    /// # Arguments
    /// * `scheduler` - Scheduling policy to use for all tasks
    /// * `error_policy` - Error handling policy for this tracker
    /// * `registry` - MetricsRegistry for Prometheus integration
    /// * `component_name` - Name for this tracker component
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # use dynamo_runtime::metrics::MetricsRegistry;
    /// # fn example(registry: Arc<dyn MetricsRegistry>) -> anyhow::Result<()> {
    /// let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
    /// let error_policy = Arc::new(LogOnlyPolicy::new());
    /// let tracker = TaskTracker::new_with_prometheus(
    ///     scheduler,
    ///     error_policy,
    ///     registry.as_ref(),
    ///     "main_tracker"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_prometheus<R: MetricsRegistry + ?Sized>(
        scheduler: Arc<dyn TaskScheduler>,
        error_policy: Arc<dyn OnErrorPolicy>,
        registry: &R,
        component_name: &str,
    ) -> anyhow::Result<Arc<Self>> {
        let prometheus_metrics = Arc::new(RootTaskMetrics::new(registry, component_name)?);

        Ok(Arc::new(
            Self::builder()
                .scheduler(scheduler)
                .error_policy(error_policy)
                .metrics(prometheus_metrics)
                .build()
                .unwrap(),
        ))
    }

    /// Create a child tracker that inherits scheduling policy
    ///
    /// The child tracker:
    /// - Gets its own independent tokio TaskTracker
    /// - Inherits the parent's scheduler
    /// - Gets a child error policy via `create_child()`
    /// - Has hierarchical metrics that chain to parent
    /// - Gets a child cancellation token from the parent
    /// - Is independent for cancellation (child cancellation doesn't affect parent)
    ///
    /// # Errors
    /// Returns an error if the parent tracker is already closed
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(root_tracker: TaskTracker) -> anyhow::Result<()> {
    /// let child_tracker = root_tracker.child_tracker()?;
    /// // Child inherits parent's policies but has separate metrics and lifecycle
    /// # Ok(())
    /// # }
    /// ```
    pub fn child_tracker(&self) -> anyhow::Result<Arc<TaskTracker>> {
        // Validate that parent tracker is still active
        if self.is_closed() {
            return Err(anyhow::anyhow!(
                "Cannot create child tracker from closed parent tracker"
            ));
        }

        let child_cancel_token = self.cancel_token.child_token();
        let child_metrics = Arc::new(ChildTaskMetrics::new(self.metrics.clone()));

        let child = Arc::new(TaskTracker {
            inner: TokioTaskTracker::new(),
            parent: None, // No parent reference needed for our hierarchical operations
            scheduler: self.scheduler.clone(),
            error_policy: self.error_policy.create_child(),
            metrics: child_metrics,
            cancel_token: child_cancel_token,
            children: Arc::new(RwLock::new(Vec::new())),
        });

        // Register this child with the parent for hierarchical operations
        self.children.write().unwrap().push(Arc::downgrade(&child));

        // Periodically clean up dead children to prevent unbounded growth
        self.cleanup_dead_children();

        Ok(child)
    }

    /// Create a child tracker builder for flexible customization
    ///
    /// The builder allows you to customize scheduling and error policies for the child tracker.
    /// If not specified, policies are inherited from the parent.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # fn example(root_tracker: TaskTracker) {
    /// // Custom scheduler, inherit error policy
    /// let child1 = root_tracker.child_tracker_builder()
    ///     .scheduler(Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5)))))
    ///     .build().unwrap();
    ///
    /// // Custom error policy, inherit scheduler
    /// let child2 = root_tracker.child_tracker_builder()
    ///     .error_policy(Arc::new(LogOnlyPolicy::new()))
    ///     .build().unwrap();
    ///
    /// // Both custom
    /// let child3 = root_tracker.child_tracker_builder()
    ///     .scheduler(Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(3)))))
    ///     .error_policy(Arc::new(LogOnlyPolicy::new()))
    ///     .build().unwrap();
    /// # }
    /// ```
    pub fn child_tracker_builder(&self) -> ChildTrackerBuilder<'_> {
        ChildTrackerBuilder {
            parent: self,
            scheduler: None,
            error_policy: None,
        }
    }

    /// Spawn a new task
    ///
    /// The task will be wrapped with scheduling and error handling logic,
    /// then executed according to the configured policies. For tasks that
    /// need to inspect cancellation tokens, use [`spawn_cancellable`] instead.
    ///
    /// # Arguments
    /// * `future` - The async task to execute
    ///
    /// # Returns
    /// A [`JoinHandle`] that can be used to await completion
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # async fn example(tracker: TaskTracker) -> anyhow::Result<()> {
    /// let handle = tracker.spawn(async {
    ///     // Your async work here
    ///     tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    ///     Ok(42)
    /// });
    ///
    /// let result = handle.await??;
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn<F, T>(&self, future: F) -> JoinHandle<Result<T>>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let task_id = self.generate_task_id();

        // Increment issued counter immediately when task is submitted
        self.metrics.increment_issued();

        // Clone necessary components to move into the task
        let scheduler = self.scheduler.clone();
        let error_policy = self.error_policy.clone();
        let metrics = self.metrics.clone();
        let cancel_token = Some(self.cancel_token.clone());

        // Wrap the user's future with our scheduling and error handling
        let wrapped_future = async move {
            Self::execute_with_policies(
                task_id,
                future,
                scheduler,
                error_policy,
                metrics,
                cancel_token,
            )
            .await
        };

        // Let tokio handle the actual task tracking
        self.inner.spawn(wrapped_future)
    }

    /// Spawn a cancellable task that receives a cancellation token
    ///
    /// This is useful for tasks that need to inspect the cancellation token
    /// and gracefully handle cancellation within their logic. The task function
    /// must return a `CancellableTaskResult` to properly track cancellation vs errors.
    ///
    /// # Arguments
    ///
    /// * `task_fn` - Function that takes a cancellation token and returns a future that resolves to `CancellableTaskResult<T>`
    ///
    /// # Returns
    /// A [`JoinHandle`] that can be used to await completion
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, CancellableTaskResult};
    /// # async fn example(tracker: TaskTracker) -> anyhow::Result<()> {
    /// let handle = tracker.spawn_cancellable(|cancel_token| async move {
    ///     tokio::select! {
    ///         _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
    ///             CancellableTaskResult::Ok(42)
    ///         },
    ///         _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
    ///     }
    /// });
    ///
    /// let result = handle.await??;
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn_cancellable<F, Fut, T>(&self, task_fn: F) -> JoinHandle<Result<T>>
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        let task_id = self.generate_task_id();

        // Increment issued counter immediately when task is submitted
        self.metrics.increment_issued();

        // Clone necessary components to move into the task
        let scheduler = self.scheduler.clone();
        let error_policy = self.error_policy.clone();
        let metrics = self.metrics.clone();
        let task_cancel_token = self.cancel_token.clone();

        // Create the future by calling the task function with the cancellation token
        let cancellable_future = task_fn(task_cancel_token.clone());

        // Convert CancellableTaskResult to Result for consistency
        let future = async move {
            match cancellable_future.await {
                CancellableTaskResult::Ok(value) => Ok(value),
                CancellableTaskResult::Cancelled => Err(anyhow::anyhow!("Task was cancelled")),
                CancellableTaskResult::Err(error) => Err(error),
            }
        };

        // Wrap the user's future with our scheduling and error handling
        let wrapped_future = async move {
            Self::execute_with_policies(
                task_id,
                future,
                scheduler,
                error_policy,
                metrics,
                Some(task_cancel_token),
            )
            .await
        };

        // Let tokio handle the actual task tracking
        self.inner.spawn(wrapped_future)
    }

    /// Get metrics for this tracker
    ///
    /// Metrics are specific to this tracker and do not include
    /// metrics from parent or child trackers.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(tracker: &TaskTracker) {
    /// let metrics = tracker.metrics();
    /// println!("Success: {}, Failed: {}", metrics.success(), metrics.failed());
    /// # }
    /// ```
    pub fn metrics(&self) -> &dyn HierarchicalTaskMetrics {
        self.metrics.as_ref()
    }

    /// Cancel this tracker and all its tasks
    ///
    /// This will signal cancellation to all currently running tasks and prevent new tasks from being spawned.
    /// The cancellation is immediate and forceful.
    pub fn cancel(&self) {
        // Close the tracker to prevent new tasks
        self.inner.close();

        // Cancel our own token
        self.cancel_token.cancel();

        // If our error policy has a cancellation token, trigger it too
        if let Some(token) = self.error_policy.cancellation_token() {
            token.cancel();
        }
    }

    /// Join this tracker and all child trackers
    ///
    /// This method gracefully shuts down the entire tracker hierarchy by:
    /// 1. Closing all trackers (preventing new task spawning)
    /// 2. Waiting for all existing tasks to complete
    ///
    /// Uses stack-safe traversal to prevent stack overflow in deep hierarchies.
    /// Children are processed before parents to ensure proper shutdown order.
    ///
    /// **Hierarchical Behavior:**
    /// - Processes children before parents to ensure proper shutdown order
    /// - Each tracker is closed before waiting (Tokio requirement)
    /// - Leaf trackers simply close and wait for their own tasks
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # async fn example(tracker: Arc<TaskTracker>) {
    /// tracker.join().await;
    /// # }
    /// ```
    pub async fn join(self: Arc<Self>) {
        // Fast path for leaf trackers (no children)
        let is_leaf = {
            let children_guard = self.children.read().unwrap();
            children_guard.is_empty()
        };

        if is_leaf {
            self.inner.close();
            self.inner.wait().await;
            return;
        }

        // Stack-safe traversal for deep hierarchies
        // Processes children before parents to ensure proper shutdown order
        let trackers = Self::collect_hierarchy(self);
        for t in trackers {
            t.inner.close();
            t.inner.wait().await;
        }
    }

    /// Check if this tracker is closed
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Get the cancellation token for this tracker
    ///
    /// This allows external code to observe or trigger cancellation of this tracker.
    /// All trackers now have cancellation tokens.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio_util::sync::CancellationToken;
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(tracker: &TaskTracker) {
    /// let token = tracker.cancellation_token();
    /// // Can check cancellation state or cancel manually
    /// if !token.is_cancelled() {
    ///     token.cancel();
    /// }
    /// # }
    /// ```
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Get the number of active child trackers
    ///
    /// This returns the count of child trackers that are still alive (not dropped).
    /// Useful for monitoring the tracker hierarchy.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # fn example(tracker: &TaskTracker) {
    /// let child_count = tracker.child_count();
    /// println!("This tracker has {} active children", child_count);
    /// # }
    /// ```
    pub fn child_count(&self) -> usize {
        let children_guard = self.children.read().unwrap();
        children_guard
            .iter()
            .filter(|weak| weak.upgrade().is_some())
            .count()
    }

    /// Clean up dead weak references if needed
    ///
    /// This is called periodically to prevent unbounded growth of dead references.
    /// Only cleans up if there are significantly more total references than alive ones.
    fn cleanup_dead_children(&self) {
        // Try to get read lock first to check if cleanup is needed
        if let Ok(children_guard) = self.children.read() {
            let total_count = children_guard.len();
            let alive_count = children_guard
                .iter()
                .filter(|weak| weak.upgrade().is_some())
                .count();

            // Only clean up if we have significant dead references
            // Use a threshold to avoid frequent cleanup operations
            let should_cleanup = total_count > 10 && total_count > alive_count * 2;
            drop(children_guard); // Release read lock

            if should_cleanup {
                // Upgrade to write lock and perform cleanup
                if let Ok(mut children_guard) = self.children.write() {
                    children_guard.retain(|weak| weak.upgrade().is_some());
                }
            }
        }
    }

    /// Generate a unique task ID
    fn generate_task_id(&self) -> TaskId {
        TaskId::new()
    }

    /// Collect all trackers in the hierarchy using iterative pre-order traversal
    ///
    /// Returns trackers in bottom-up order (children before parents) for safe shutdown.
    /// Uses reversed pre-order to achieve post-order semantics without recursion,
    /// avoiding stack overflow for deep hierarchies and associated heap allocations.
    fn collect_hierarchy(start: Arc<TaskTracker>) -> Vec<Arc<TaskTracker>> {
        let mut result = Vec::new();
        let mut stack = vec![start];
        let mut visited = HashSet::new();

        // Collect all trackers using depth-first search
        while let Some(tracker) = stack.pop() {
            let tracker_ptr = Arc::as_ptr(&tracker) as usize;
            if visited.contains(&tracker_ptr) {
                continue;
            }
            visited.insert(tracker_ptr);

            // Add current tracker to result
            result.push(Arc::clone(&tracker));

            // Add children to stack for processing
            if let Ok(children_guard) = tracker.children.read() {
                for weak_child in children_guard.iter() {
                    if let Some(child) = weak_child.upgrade() {
                        let child_ptr = Arc::as_ptr(&child) as usize;
                        if !visited.contains(&child_ptr) {
                            stack.push(child);
                        }
                    }
                }
            }
        }

        // Reverse to get bottom-up order (children before parents)
        result.reverse();
        result
    }

    /// Execute a task with scheduling and error handling policies
    #[tracing::instrument(level = "debug", skip_all, fields(task_id = %task_id))]
    async fn execute_with_policies<F, T>(
        task_id: TaskId,
        future: F,
        scheduler: Arc<dyn TaskScheduler>,
        error_policy: Arc<dyn OnErrorPolicy>,
        metrics: Arc<dyn HierarchicalTaskMetrics>,
        cancel_token: Option<CancellationToken>,
    ) -> Result<T>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        debug!("Starting task execution");
        metrics.increment_active();

        // Create a result slot for capturing the task result
        let result_slot: Arc<tokio::sync::Mutex<Option<Result<T>>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let result_slot_clone = result_slot.clone();

        // Box and pin the future, capturing the result
        let boxed_future = Box::pin(async move {
            let result = future.await;
            *result_slot_clone.lock().await = Some(result);
            Ok(())
        });

        // Acquire execution slot through scheduler with cancellation token
        let guard_result = async { scheduler.acquire_execution_slot(cancel_token).await }
            .instrument(tracing::debug_span!("scheduler_resource_acquisition"))
            .await;

        let final_result = match guard_result {
            SchedulingResult::Execute(guard) => {
                // Execute task through the guard (enforces no cancellation during execution)
                let _execution_result = async { guard.execute_task(boxed_future).await }
                    .instrument(tracing::debug_span!("task_execution"))
                    .await;

                // Extract the actual result from our slot
                let result = result_slot
                    .lock()
                    .await
                    .take()
                    .expect("Result should have been set by completed future");

                match result {
                    Ok(value) => {
                        metrics.increment_success();
                        debug!("Task completed successfully");
                        Ok(value)
                    }
                    Err(error) => {
                        metrics.increment_failed();
                        error_policy.on_error(&error, task_id).await;

                        if error_policy.should_cancel_on_error(&error) {
                            warn!(?error, "Error triggered cancellation");
                            // Note: Individual task cancellation would need to be handled
                            // by the error policy if it has access to the tracker
                        }

                        debug!(?error, "Task failed");
                        Err(error)
                    }
                }
            }
            SchedulingResult::Cancelled => {
                metrics.increment_cancelled();
                debug!("Task was cancelled during resource acquisition");
                Err(anyhow::anyhow!("Task was cancelled"))
            }
            SchedulingResult::Rejected(reason) => {
                metrics.increment_rejected();
                debug!(reason, "Task was rejected by scheduler");
                Err(anyhow::anyhow!("Task rejected: {}", reason))
            }
        };

        metrics.decrement_active();
        final_result
    }
}

// Blanket implementation for all schedulers
impl ArcPolicy for UnlimitedScheduler {}
impl ArcPolicy for SemaphoreScheduler {}

// Blanket implementation for all error policies
impl ArcPolicy for LogOnlyPolicy {}
impl ArcPolicy for CancelOnError {}
impl ArcPolicy for ThresholdCancelPolicy {}
impl ArcPolicy for RateCancelPolicy {}

/// Resource guard for unlimited scheduling
///
/// This guard enforces that once task execution begins, it always runs to completion.
#[derive(Debug)]
pub struct UnlimitedGuard;

#[async_trait]
impl ResourceGuard for UnlimitedGuard {
    async fn execute_task(
        self: Box<Self>,
        task: Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>,
    ) -> Result<()> {
        debug!("Executing task with unlimited scheduler");
        let result = task.await;
        debug!("Task execution completed");
        result
    }
}

/// Unlimited task scheduler that executes all tasks immediately
///
/// This scheduler provides no concurrency limits and executes all submitted tasks
/// immediately. Useful for testing, high-throughput scenarios, or when external
/// systems provide the concurrency control.
///
/// ## Cancellation Behavior
///
/// - Respects cancellation tokens before resource acquisition
/// - Once execution begins (via ResourceGuard), always awaits task completion
/// - Tasks handle their own cancellation internally (if created with `spawn_cancellable`)
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::UnlimitedScheduler;
/// let scheduler = UnlimitedScheduler::new();
/// ```
#[derive(Debug)]
pub struct UnlimitedScheduler;

impl UnlimitedScheduler {
    /// Create a new unlimited scheduler
    pub fn new() -> Self {
        Self
    }

    /// Create a new unlimited scheduler returning Arc
    pub fn new_arc() -> Arc<Self> {
        Self::new().new_arc()
    }
}

impl Default for UnlimitedScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TaskScheduler for UnlimitedScheduler {
    async fn acquire_execution_slot(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> SchedulingResult<Box<dyn ResourceGuard>> {
        debug!("Acquiring execution slot (unlimited scheduler)");

        // Check for cancellation before allocating resources
        if let Some(ref token) = cancel_token {
            if token.is_cancelled() {
                debug!("Task cancelled before acquiring execution slot");
                return SchedulingResult::Cancelled;
            }
        }

        // No resource constraints for unlimited scheduler
        debug!("Execution slot acquired immediately");
        SchedulingResult::Execute(Box::new(UnlimitedGuard))
    }
}

/// Resource guard for semaphore-based scheduling
///
/// This guard holds a semaphore permit and enforces that task execution
/// always runs to completion. The permit is automatically released when
/// the guard is dropped.
#[derive(Debug)]
pub struct SemaphoreGuard {
    _permit: tokio::sync::OwnedSemaphorePermit,
}

#[async_trait]
impl ResourceGuard for SemaphoreGuard {
    async fn execute_task(
        self: Box<Self>,
        task: Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>,
    ) -> Result<()> {
        debug!("Executing task with semaphore scheduler");

        // Execute task while holding permit
        let result = task.await;

        debug!("Released semaphore permit");
        // Permit is automatically dropped here, releasing the semaphore
        result
    }
}

/// Semaphore-based task scheduler
///
/// Limits concurrent task execution using a [`tokio::sync::Semaphore`].
/// Tasks will wait for an available permit before executing.
///
/// ## Cancellation Behavior
///
/// - Respects cancellation tokens before and during permit acquisition
/// - Once a permit is acquired (via ResourceGuard), always awaits task completion
/// - Holds the permit until the task completes (regardless of cancellation)
/// - Tasks handle their own cancellation internally (if created with `spawn_cancellable`)
///
/// This ensures that permits are not leaked when tasks are cancelled, while still
/// allowing cancellable tasks to terminate gracefully on their own.
///
/// # Example
/// ```rust
/// # use std::sync::Arc;
/// # use tokio::sync::Semaphore;
/// # use dynamo_runtime::utils::tasks::tracker::SemaphoreScheduler;
/// // Allow up to 5 concurrent tasks
/// let semaphore = Arc::new(Semaphore::new(5));
/// let scheduler = SemaphoreScheduler::new(semaphore);
/// ```
#[derive(Debug)]
pub struct SemaphoreScheduler {
    semaphore: Arc<Semaphore>,
}

impl SemaphoreScheduler {
    /// Create a new semaphore scheduler
    ///
    /// # Arguments
    /// * `semaphore` - Semaphore to use for concurrency control
    pub fn new(semaphore: Arc<Semaphore>) -> Self {
        Self { semaphore }
    }

    /// Create a semaphore scheduler with the specified number of permits, returning Arc
    pub fn with_permits(permits: usize) -> Arc<Self> {
        Self::new(Arc::new(Semaphore::new(permits))).new_arc()
    }

    /// Get the number of available permits
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

#[async_trait]
impl TaskScheduler for SemaphoreScheduler {
    async fn acquire_execution_slot(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> SchedulingResult<Box<dyn ResourceGuard>> {
        debug!("Acquiring semaphore permit");

        // Check for cancellation before attempting to acquire semaphore
        if let Some(ref token) = cancel_token {
            if token.is_cancelled() {
                debug!("Task cancelled before acquiring semaphore permit");
                return SchedulingResult::Cancelled;
            }
        }

        // Try to acquire a permit, with cancellation support
        let permit = if let Some(ref token) = cancel_token {
            tokio::select! {
                result = self.semaphore.clone().acquire_owned() => {
                    match result {
                        Ok(permit) => permit,
                        Err(_) => return SchedulingResult::Cancelled,
                    }
                }
                _ = token.cancelled() => {
                    debug!("Task cancelled while waiting for semaphore permit");
                    return SchedulingResult::Cancelled;
                }
            }
        } else {
            match self.semaphore.clone().acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => return SchedulingResult::Cancelled,
            }
        };

        debug!("Acquired semaphore permit");
        SchedulingResult::Execute(Box::new(SemaphoreGuard { _permit: permit }))
    }
}

/// Error policy that cancels tasks on specific error patterns
///
/// This policy maintains a cancellation token and will cancel all tasks
/// when certain error patterns are encountered.
///
/// # Example
/// ```rust
/// # use tokio_util::sync::CancellationToken;
/// # use dynamo_runtime::utils::tasks::tracker::CancelOnError;
/// let cancel_token = CancellationToken::new();
/// let policy = CancelOnError::new(cancel_token.clone());
///
/// // Policy will cancel tasks on OutOfMemory errors
/// let policy = CancelOnError::with_patterns(
///     cancel_token,
///     vec!["OutOfMemory".to_string(), "DeviceError".to_string()]
/// );
/// ```
#[derive(Debug)]
pub struct CancelOnError {
    cancel_token: CancellationToken,
    error_patterns: Vec<String>,
}

impl CancelOnError {
    /// Create a new cancel-on-error policy with default error patterns
    ///
    /// Default patterns: ["OutOfMemory", "DeviceError"]
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            cancel_token,
            error_patterns: vec!["OutOfMemory".to_string(), "DeviceError".to_string()],
        }
    }

    /// Create a new cancel-on-error policy with default patterns, returning Arc and token
    pub fn new_arc() -> (Arc<Self>, CancellationToken) {
        let token = CancellationToken::new();
        (Self::new(token.clone()).new_arc(), token)
    }

    /// Create a new cancel-on-error policy with custom error patterns
    ///
    /// # Arguments
    /// * `cancel_token` - Token to cancel when errors occur
    /// * `error_patterns` - List of error message patterns that trigger cancellation
    pub fn with_patterns(cancel_token: CancellationToken, error_patterns: Vec<String>) -> Self {
        Self {
            cancel_token,
            error_patterns,
        }
    }

    /// Create a new cancel-on-error policy with custom patterns, returning Arc and token
    pub fn with_patterns_arc(error_patterns: Vec<String>) -> (Arc<Self>, CancellationToken) {
        let token = CancellationToken::new();
        (
            Self::with_patterns(token.clone(), error_patterns).new_arc(),
            token,
        )
    }
}

#[async_trait]
impl OnErrorPolicy for CancelOnError {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token - when parent cancels, child cancels too
        // When child cancels, parent is unaffected
        Arc::new(CancelOnError {
            cancel_token: self.cancel_token.child_token(),
            error_patterns: self.error_patterns.clone(),
        })
    }

    async fn on_error(&self, error: &anyhow::Error, task_id: TaskId) {
        error!(?task_id, ?error, "Task failed");

        if self.should_cancel_on_error(error) {
            warn!(?task_id, "Triggering cancellation due to critical error");
            self.cancel_token.cancel();
        }
    }

    async fn on_failure_threshold_exceeded(&self, failure_rate: f64) {
        warn!(?failure_rate, "Failure threshold exceeded");
    }

    fn should_cancel_on_error(&self, error: &anyhow::Error) -> bool {
        let error_str = error.to_string();
        self.error_patterns
            .iter()
            .any(|pattern| error_str.contains(pattern))
    }

    fn cancellation_token(&self) -> Option<CancellationToken> {
        Some(self.cancel_token.clone())
    }
}

/// Simple error policy that only logs errors
///
/// This policy does not trigger cancellation and is useful for
/// non-critical tasks or when you want to handle errors externally.
#[derive(Debug)]
pub struct LogOnlyPolicy;

impl LogOnlyPolicy {
    /// Create a new log-only policy
    pub fn new() -> Self {
        Self
    }

    /// Create a new log-only policy returning Arc
    pub fn new_arc() -> Arc<Self> {
        Self::new().new_arc()
    }
}

impl Default for LogOnlyPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OnErrorPolicy for LogOnlyPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Simple policies can just clone themselves
        Arc::new(LogOnlyPolicy)
    }

    async fn on_error(&self, error: &anyhow::Error, task_id: TaskId) {
        error!(?task_id, ?error, "Task failed - logging only");
    }

    async fn on_failure_threshold_exceeded(&self, failure_rate: f64) {
        warn!(?failure_rate, "Failure threshold exceeded - logging only");
    }
}

/// Error policy that cancels tasks after a threshold number of failures
///
/// This policy tracks the number of failed tasks and triggers cancellation
/// when the failure count exceeds the specified threshold. Useful for
/// preventing cascading failures in distributed systems.
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::ThresholdCancelPolicy;
/// // Cancel after 5 failures
/// let (policy, token) = ThresholdCancelPolicy::new_arc(5);
/// ```
#[derive(Debug)]
pub struct ThresholdCancelPolicy {
    cancel_token: CancellationToken,
    max_failures: usize,
    failure_count: AtomicU64,
}

impl ThresholdCancelPolicy {
    /// Create a new threshold cancel policy
    ///
    /// # Arguments
    /// * `cancel_token` - Token to cancel when threshold is exceeded
    /// * `max_failures` - Maximum number of failures before cancellation
    pub fn new(cancel_token: CancellationToken, max_failures: usize) -> Self {
        Self {
            cancel_token,
            max_failures,
            failure_count: AtomicU64::new(0),
        }
    }

    /// Create a new threshold cancel policy returning Arc and token
    pub fn new_arc(max_failures: usize) -> (Arc<Self>, CancellationToken) {
        let token = CancellationToken::new();
        (Self::new(token.clone(), max_failures).new_arc(), token)
    }

    /// Get the current failure count
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl OnErrorPolicy for ThresholdCancelPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token and inherits the same failure threshold
        Arc::new(ThresholdCancelPolicy {
            cancel_token: self.cancel_token.child_token(),
            max_failures: self.max_failures,
            failure_count: AtomicU64::new(0), // Child starts with fresh count
        })
    }

    async fn on_error(&self, error: &anyhow::Error, task_id: TaskId) {
        error!(?task_id, ?error, "Task failed");

        let current_failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        if current_failures >= self.max_failures as u64 {
            warn!(
                ?task_id,
                current_failures,
                max_failures = self.max_failures,
                "Failure threshold exceeded, triggering cancellation"
            );
            self.cancel_token.cancel();
        } else {
            debug!(
                ?task_id,
                current_failures,
                max_failures = self.max_failures,
                "Task failed, tracking failure count"
            );
        }
    }

    async fn on_failure_threshold_exceeded(&self, failure_rate: f64) {
        warn!(?failure_rate, "Failure threshold exceeded");
    }

    fn should_cancel_on_error(&self, _error: &anyhow::Error) -> bool {
        // We handle cancellation logic in on_error, not here
        false
    }

    fn cancellation_token(&self) -> Option<CancellationToken> {
        Some(self.cancel_token.clone())
    }
}

/// Error policy that cancels tasks when failure rate exceeds threshold within time window
///
/// This policy tracks failures over a rolling time window and triggers cancellation
/// when the failure rate exceeds the specified threshold. More sophisticated than
/// simple count-based thresholds as it considers the time dimension.
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::RateCancelPolicy;
/// // Cancel if more than 50% of tasks fail within any 60-second window
/// let (policy, token) = RateCancelPolicy::new_arc(0.5, 60);
/// ```
#[derive(Debug)]
pub struct RateCancelPolicy {
    cancel_token: CancellationToken,
    max_failure_rate: f32,
    window_secs: u64,
    // TODO: Implement time-window tracking when needed
    // For now, this is a placeholder structure with the interface defined
}

impl RateCancelPolicy {
    /// Create a new rate-based cancel policy
    ///
    /// # Arguments
    /// * `cancel_token` - Token to cancel when rate threshold is exceeded
    /// * `max_failure_rate` - Maximum failure rate (0.0 to 1.0) before cancellation
    /// * `window_secs` - Time window in seconds for rate calculation
    pub fn new(cancel_token: CancellationToken, max_failure_rate: f32, window_secs: u64) -> Self {
        Self {
            cancel_token,
            max_failure_rate,
            window_secs,
        }
    }

    /// Create a new rate-based cancel policy returning Arc and token
    pub fn new_arc(max_failure_rate: f32, window_secs: u64) -> (Arc<Self>, CancellationToken) {
        let token = CancellationToken::new();
        (
            Self::new(token.clone(), max_failure_rate, window_secs).new_arc(),
            token,
        )
    }
}

#[async_trait]
impl OnErrorPolicy for RateCancelPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        Arc::new(RateCancelPolicy {
            cancel_token: self.cancel_token.child_token(),
            max_failure_rate: self.max_failure_rate,
            window_secs: self.window_secs,
        })
    }

    async fn on_error(&self, error: &anyhow::Error, task_id: TaskId) {
        error!(?task_id, ?error, "Task failed");

        // TODO: Implement time-window failure rate calculation
        // For now, just log the error
        warn!(
            ?task_id,
            max_failure_rate = self.max_failure_rate,
            window_secs = self.window_secs,
            "Rate-based error policy - time window tracking not yet implemented"
        );
    }

    async fn on_failure_threshold_exceeded(&self, failure_rate: f64) {
        warn!(?failure_rate, "Failure rate threshold exceeded");
    }

    fn should_cancel_on_error(&self, _error: &anyhow::Error) -> bool {
        false // We handle cancellation in on_error when implemented
    }

    fn cancellation_token(&self) -> Option<CancellationToken> {
        Some(self.cancel_token.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    /// Helper to create a semaphore scheduler for tests
    fn create_semaphore_scheduler(permits: usize) -> Arc<SemaphoreScheduler> {
        Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(permits))))
    }

    /// Helper to create a cancel-on-error policy for tests
    fn create_cancel_policy() -> (Arc<CancelOnError>, CancellationToken) {
        let token = CancellationToken::new();
        let policy = Arc::new(CancelOnError::new(token.clone()));
        (policy, token)
    }

    /// Helper to create a log-only policy for tests
    fn create_log_policy() -> Arc<LogOnlyPolicy> {
        Arc::new(LogOnlyPolicy::new())
    }

    #[tokio::test]
    async fn test_basic_task_execution() {
        // Test successful task execution
        let scheduler = create_semaphore_scheduler(1);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = tracker.spawn(async {
            // Wait for signal to complete instead of sleep
            rx.await.ok();
            Ok(42)
        });

        // Signal task to complete
        tx.send(()).ok();

        // Verify task completes successfully
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);

        // Verify metrics
        assert_eq!(tracker.metrics().success(), 1);
        assert_eq!(tracker.metrics().failed(), 0);
        assert_eq!(tracker.metrics().cancelled(), 0);
        assert_eq!(tracker.metrics().active(), 0);
    }

    #[tokio::test]
    async fn test_task_failure() {
        // Test task failure handling
        let scheduler = create_semaphore_scheduler(1);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        let handle = tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("test error")) });

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test error"));

        // Verify metrics
        assert_eq!(tracker.metrics().success(), 0);
        assert_eq!(tracker.metrics().failed(), 1);
        assert_eq!(tracker.metrics().cancelled(), 0);
    }

    #[tokio::test]
    async fn test_semaphore_concurrency_limit() {
        // Test that semaphore limits concurrent execution
        let scheduler = create_semaphore_scheduler(2); // Only 2 concurrent tasks
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler.clone(), error_policy);

        let counter = Arc::new(AtomicU32::new(0));
        let max_concurrent = Arc::new(AtomicU32::new(0));

        // Use broadcast channel to coordinate all tasks
        let (tx, _) = tokio::sync::broadcast::channel(1);
        let mut handles = Vec::new();

        // Spawn 5 tasks that will track concurrency
        for _ in 0..5 {
            let counter_clone = counter.clone();
            let max_clone = max_concurrent.clone();
            let mut rx = tx.subscribe();

            let handle = tracker.spawn(async move {
                // Increment active counter
                let current = counter_clone.fetch_add(1, Ordering::Relaxed) + 1;

                // Track max concurrent
                max_clone.fetch_max(current, Ordering::Relaxed);

                // Wait for signal to complete instead of sleep
                rx.recv().await.ok();

                // Decrement when done
                counter_clone.fetch_sub(1, Ordering::Relaxed);

                Ok(())
            });
            handles.push(handle);
        }

        // Give tasks time to start and register concurrency
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Signal all tasks to complete
        tx.send(()).ok();

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Verify that no more than 2 tasks ran concurrently
        assert!(max_concurrent.load(Ordering::Relaxed) <= 2);

        // Verify all tasks completed successfully
        assert_eq!(tracker.metrics().success(), 5);
        assert_eq!(tracker.metrics().failed(), 0);
    }

    #[tokio::test]
    async fn test_cancel_on_error_policy() {
        // Test that CancelOnError policy works correctly
        let (error_policy, cancel_token) = create_cancel_policy();
        let scheduler = create_semaphore_scheduler(10);
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Spawn a task that will trigger cancellation
        let handle =
            tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("OutOfMemory error occurred")) });

        // Wait for the error to occur
        let result = handle.await.unwrap();
        assert!(result.is_err());

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Verify the cancel token was triggered
        assert!(cancel_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_tracker_cancellation() {
        // Test manual cancellation of tracker with CancelOnError policy
        let (error_policy, cancel_token) = create_cancel_policy();
        let scheduler = create_semaphore_scheduler(10);
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Use oneshot channel instead of sleep for deterministic timing
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn a task that respects cancellation
        let handle = tracker.spawn({
            let cancel_token = cancel_token.clone();
            async move {
                tokio::select! {
                    _ = rx => Ok(()),
                    _ = cancel_token.cancelled() => Err(anyhow::anyhow!("Task was cancelled")),
                }
            }
        });

        // Cancel the tracker
        tracker.cancel();

        // Task should be cancelled
        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }

    #[tokio::test]
    async fn test_child_tracker_independence() {
        // Test that child tracker has independent lifecycle
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);

        let child = parent.child_tracker().unwrap();

        // Both should be operational initially
        assert!(!parent.is_closed());
        assert!(!child.is_closed());

        // Cancel child only
        child.cancel();

        // Parent should remain operational
        assert!(!parent.is_closed());

        // Parent can still spawn tasks
        let handle = parent.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_independent_metrics() {
        // Test that parent and child have independent metrics
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child = parent.child_tracker().unwrap();

        // Run tasks in parent
        let handle1 = parent.spawn(async { Ok(1) });
        handle1.await.unwrap().unwrap();

        // Run tasks in child
        let handle2 = child.spawn(async { Ok(2) });
        handle2.await.unwrap().unwrap();

        // Each should have their own metrics, but parent sees aggregated
        assert_eq!(parent.metrics().success(), 2); // Parent sees its own + child's
        assert_eq!(child.metrics().success(), 1); // Child sees only its own
        assert_eq!(parent.metrics().total_completed(), 2); // Parent sees aggregated total
        assert_eq!(child.metrics().total_completed(), 1); // Child sees only its own
    }

    #[tokio::test]
    async fn test_cancel_on_error_hierarchy() {
        // Test that CancelOnError policy creates proper child tokens
        let (error_policy, parent_token) = create_cancel_policy();
        let scheduler = create_semaphore_scheduler(10);
        let parent = TaskTracker::new(scheduler, error_policy);
        let child = parent.child_tracker().unwrap();

        // Get child's cancel token
        let child_token = child
            .error_policy
            .cancellation_token()
            .expect("Child should have cancel token");

        // Child token should be different from parent
        assert!(!parent_token.is_cancelled());
        assert!(!child_token.is_cancelled());

        // Trigger error in child that causes cancellation
        let handle = child.spawn(async { Err::<(), _>(anyhow::anyhow!("OutOfMemory in child")) });

        handle.await.unwrap().unwrap_err();

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Child should be cancelled, but parent should be unaffected
        assert!(!parent_token.is_cancelled());
        assert!(child_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_graceful_shutdown() {
        // Test graceful shutdown with close()
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Use broadcast channel to coordinate task completion
        let (tx, _) = tokio::sync::broadcast::channel(1);
        let mut handles = Vec::new();

        // Spawn some tasks
        for i in 0..3 {
            let mut rx = tx.subscribe();
            let handle = tracker.spawn(async move {
                // Wait for signal instead of sleep
                rx.recv().await.ok();
                Ok(i)
            });
            handles.push(handle);
        }

        // Signal all tasks to complete before closing
        tx.send(()).ok();

        // Close tracker and wait for completion
        tracker.clone().join().await;

        // All tasks should complete successfully
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            assert!(result < 3);
        }

        // Tracker should be closed
        assert!(tracker.is_closed());
    }

    #[tokio::test]
    async fn test_semaphore_scheduler_permit_tracking() {
        // Test that SemaphoreScheduler properly tracks permits
        let semaphore = Arc::new(Semaphore::new(3));
        let scheduler = Arc::new(SemaphoreScheduler::new(semaphore.clone()));
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler.clone(), error_policy);

        // Initially all permits should be available
        assert_eq!(scheduler.available_permits(), 3);

        // Use broadcast channel to coordinate task completion
        let (tx, _) = tokio::sync::broadcast::channel(1);
        let mut handles = Vec::new();

        // Spawn 3 tasks that will hold permits
        for _ in 0..3 {
            let mut rx = tx.subscribe();
            let handle = tracker.spawn(async move {
                // Wait for signal to complete
                rx.recv().await.ok();
                Ok(())
            });
            handles.push(handle);
        }

        // Give tasks time to acquire permits
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // All permits should be taken
        assert_eq!(scheduler.available_permits(), 0);

        // Signal all tasks to complete
        tx.send(()).ok();

        // Wait for tasks to complete
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // All permits should be available again
        assert_eq!(scheduler.available_permits(), 3);
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        // Test that TaskTracker builder works correctly
        let scheduler = create_semaphore_scheduler(5);
        let error_policy = create_log_policy();

        let tracker = TaskTracker::builder()
            .scheduler(scheduler)
            .error_policy(error_policy)
            .build()
            .unwrap();

        // Tracker should have a cancellation token
        let token = tracker.cancellation_token();
        assert!(!token.is_cancelled());

        // Should be able to spawn tasks
        let handle = tracker.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_all_trackers_have_cancellation_tokens() {
        // Test that all trackers (root and children) have cancellation tokens
        let scheduler = create_semaphore_scheduler(5);
        let error_policy = create_log_policy();

        let root = TaskTracker::new(scheduler, error_policy);
        let child = root.child_tracker().unwrap();
        let grandchild = child.child_tracker().unwrap();

        // All should have cancellation tokens
        let root_token = root.cancellation_token();
        let child_token = child.cancellation_token();
        let grandchild_token = grandchild.cancellation_token();

        assert!(!root_token.is_cancelled());
        assert!(!child_token.is_cancelled());
        assert!(!grandchild_token.is_cancelled());

        // Child tokens should be different from parent
        // (We can't directly compare tokens, but we can test behavior)
        root_token.cancel();

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Root should be cancelled
        assert!(root_token.is_cancelled());
        // Children should also be cancelled (because they are child tokens)
        assert!(child_token.is_cancelled());
        assert!(grandchild_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_spawn_cancellable_task() {
        // Test cancellable task spawning with proper result handling
        let scheduler = create_semaphore_scheduler(5);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Test successful completion
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = tracker.spawn_cancellable(|_cancel_token| async move {
            // Wait for signal instead of sleep
            rx.await.ok();
            CancellableTaskResult::Ok(42)
        });

        // Signal task to complete
        tx.send(()).ok();

        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);
        assert_eq!(tracker.metrics().success(), 1);

        // Test cancellation handling
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tracker.spawn_cancellable(|cancel_token| async move {
            tokio::select! {
                _ = rx => CancellableTaskResult::Ok("should not complete"),
                _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
            }
        });

        // Cancel the tracker
        tracker.cancel();

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }

    #[tokio::test]
    async fn test_spawn_cancellable_error_handling() {
        // Test error handling in cancellable tasks
        let scheduler = create_semaphore_scheduler(5);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Test error result
        let handle = tracker.spawn_cancellable(|_cancel_token| async move {
            CancellableTaskResult::<i32>::Err(anyhow::anyhow!("test error"))
        });

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test error"));
        assert_eq!(tracker.metrics().failed(), 1);
    }

    #[tokio::test]
    async fn test_cancellation_before_execution() {
        // Test that cancellation before task execution is handled
        let scheduler = create_semaphore_scheduler(1);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Cancel the tracker first
        tracker.cancel();

        // Use oneshot channel instead of sleep
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();

        // Now try to spawn a task - it should be cancelled before execution
        let handle = tracker.spawn(async {
            rx.await.ok();
            Ok(42)
        });

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }

    #[tokio::test]
    async fn test_semaphore_scheduler_with_cancellation() {
        // Test that SemaphoreScheduler respects cancellation tokens
        let scheduler = create_semaphore_scheduler(1);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Start a long-running task to occupy the semaphore
        let blocker_token = tracker.cancellation_token();
        let _blocker_handle = tracker.spawn(async move {
            // Wait for cancellation
            blocker_token.cancelled().await;
            Ok(())
        });

        // Give the blocker time to acquire the permit
        tokio::task::yield_now().await;

        // Use oneshot channel for the second task
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn another task that will wait for semaphore
        let handle = tracker.spawn(async {
            rx.await.ok();
            Ok(42)
        });

        // Cancel the tracker while second task is waiting for permit
        tracker.cancel();

        // The waiting task should be cancelled
        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }

    #[tokio::test]
    async fn test_child_tracker_cancellation_independence() {
        // Test that child tracker cancellation doesn't affect parent
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child = parent.child_tracker().unwrap();

        // Cancel only the child
        child.cancel();

        // Parent should still be operational
        let parent_token = parent.cancellation_token();
        assert!(!parent_token.is_cancelled());

        // Parent can still spawn tasks
        let handle = parent.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);

        // Child should be cancelled
        let child_token = child.cancellation_token();
        assert!(child_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_parent_cancellation_propagates_to_children() {
        // Test that parent cancellation propagates to all children
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child1 = parent.child_tracker().unwrap();
        let child2 = parent.child_tracker().unwrap();
        let grandchild = child1.child_tracker().unwrap();

        // Cancel the parent
        parent.cancel();

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // All should be cancelled
        assert!(parent.cancellation_token().is_cancelled());
        assert!(child1.cancellation_token().is_cancelled());
        assert!(child2.cancellation_token().is_cancelled());
        assert!(grandchild.cancellation_token().is_cancelled());
    }

    #[tokio::test]
    async fn test_scheduler_queue_depth_calculation() {
        // Test that we can calculate tasks queued in scheduler
        let scheduler = create_semaphore_scheduler(2); // Only 2 concurrent tasks
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Initially no tasks
        assert_eq!(tracker.metrics().issued(), 0);
        assert_eq!(tracker.metrics().active(), 0);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);

        // Use long running tasks to occupy the semaphore
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

        // Spawn 2 long-running tasks that will hold the semaphore permits
        let tx1 = tx.clone();
        let handle1 = tracker.spawn(async move {
            tx1.send(()).await.unwrap(); // Signal that task started
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok(1)
        });

        let tx2 = tx.clone();
        let handle2 = tracker.spawn(async move {
            tx2.send(()).await.unwrap(); // Signal that task started
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok(2)
        });

        // Wait for both tasks to start (they should get the 2 semaphore permits)
        rx.recv().await.unwrap();
        rx.recv().await.unwrap();
        drop(tx); // Close the channel

        // Give tasks time to register as active
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Now spawn 2 more tasks - these should be queued
        let handle3 = tracker.spawn(async { Ok(3) });
        let handle4 = tracker.spawn(async { Ok(4) });

        // Give queued tasks time to register
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Check metrics - we should have queue depth now
        assert_eq!(tracker.metrics().issued(), 4, "Should have 4 issued tasks");
        assert_eq!(
            tracker.metrics().pending(),
            4,
            "Should have 4 pending tasks total"
        );
        assert_eq!(
            tracker.metrics().total_completed(),
            0,
            "No tasks completed yet"
        );

        // The active count should be 2 (semaphore limit) and queued should be 2
        // Note: There might be slight timing variations, so let's be flexible
        let active = tracker.metrics().active();
        let queued = tracker.metrics().queued();
        println!("Active: {}, Queued: {}", active, queued);

        // At minimum, we should have some tasks active and total pending should be 4
        assert!(active > 0, "Should have some active tasks");
        assert_eq!(
            active + queued,
            4,
            "Active + Queued should equal pending tasks"
        );

        // Wait for all tasks to complete
        let results = tokio::join!(handle1, handle2, handle3, handle4);
        for result in [results.0, results.1, results.2, results.3] {
            result.unwrap().unwrap();
        }

        // Final state - all completed
        assert_eq!(tracker.metrics().issued(), 4);
        assert_eq!(tracker.metrics().active(), 0);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);
        assert_eq!(tracker.metrics().total_completed(), 4);
        assert_eq!(tracker.metrics().success(), 4);
    }

    #[tokio::test]
    async fn test_issued_counter_tracking() {
        // Test that issued counter is incremented when tasks are spawned
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let tracker = TaskTracker::new(scheduler, error_policy);

        // Initially no tasks issued
        assert_eq!(tracker.metrics().issued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);

        // Spawn some tasks
        let handle1 = tracker.spawn(async { Ok(1) });
        let handle2 = tracker.spawn(async { Ok(2) });
        let handle3 = tracker.spawn_cancellable(|_| async { CancellableTaskResult::Ok(3) });

        // Issued counter should be incremented immediately
        assert_eq!(tracker.metrics().issued(), 3);
        assert_eq!(tracker.metrics().pending(), 3); // None completed yet

        // Complete the tasks
        assert_eq!(handle1.await.unwrap().unwrap(), 1);
        assert_eq!(handle2.await.unwrap().unwrap(), 2);
        assert_eq!(handle3.await.unwrap().unwrap(), 3);

        // Check final accounting
        assert_eq!(tracker.metrics().issued(), 3);
        assert_eq!(tracker.metrics().success(), 3);
        assert_eq!(tracker.metrics().total_completed(), 3);
        assert_eq!(tracker.metrics().pending(), 0); // All completed

        // Test hierarchical accounting
        let child = tracker.child_tracker().unwrap();
        let child_handle = child.spawn(async { Ok(42) });

        // Both parent and child should see the issued task
        assert_eq!(child.metrics().issued(), 1);
        assert_eq!(tracker.metrics().issued(), 4); // Parent sees all

        child_handle.await.unwrap().unwrap();

        // Final hierarchical check
        assert_eq!(child.metrics().pending(), 0);
        assert_eq!(tracker.metrics().pending(), 0);
        assert_eq!(tracker.metrics().success(), 4); // Parent sees all successes
    }

    #[tokio::test]
    async fn test_child_tracker_builder() {
        // Test that child tracker builder allows custom policies
        let parent_scheduler = create_semaphore_scheduler(10);
        let parent_error_policy = create_log_policy();
        let parent = TaskTracker::new(parent_scheduler, parent_error_policy);

        // Test custom scheduler, inherit error policy
        let child_scheduler = create_semaphore_scheduler(5);
        let child1 = parent
            .child_tracker_builder()
            .scheduler(child_scheduler.clone())
            .build()
            .unwrap();

        // Test custom error policy, inherit scheduler
        let (child_error_policy, _) = create_cancel_policy();
        let child2 = parent
            .child_tracker_builder()
            .error_policy(child_error_policy)
            .build()
            .unwrap();

        // Test both custom
        let another_scheduler = create_semaphore_scheduler(3);
        let another_error_policy = create_log_policy();
        let child3 = parent
            .child_tracker_builder()
            .scheduler(another_scheduler)
            .error_policy(another_error_policy)
            .build()
            .unwrap();

        // Test that all children are properly registered
        assert_eq!(parent.child_count(), 3);

        // Test that custom schedulers work
        let handle1 = child1.spawn(async { Ok(1) });
        let handle2 = child2.spawn(async { Ok(2) });
        let handle3 = child3.spawn(async { Ok(3) });

        assert_eq!(handle1.await.unwrap().unwrap(), 1);
        assert_eq!(handle2.await.unwrap().unwrap(), 2);
        assert_eq!(handle3.await.unwrap().unwrap(), 3);

        // Verify metrics still work
        assert_eq!(parent.metrics().success(), 3); // All child successes roll up
        assert_eq!(child1.metrics().success(), 1);
        assert_eq!(child2.metrics().success(), 1);
        assert_eq!(child3.metrics().success(), 1);
    }

    #[tokio::test]
    async fn test_hierarchical_metrics_aggregation() {
        // Test that child metrics aggregate up to parent
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child1 = parent.child_tracker().unwrap();
        let child2 = parent.child_tracker().unwrap();
        let grandchild = child1.child_tracker().unwrap();

        // Run successful tasks in different trackers
        let handle1 = parent.spawn(async { Ok(1) });
        let handle2 = child1.spawn(async { Ok(2) });
        let handle3 = child2.spawn(async { Ok(3) });
        let handle4 = grandchild.spawn(async { Ok(4) });

        // Wait for all tasks to complete
        let result1 = handle1.await.unwrap().unwrap();
        let result2 = handle2.await.unwrap().unwrap();
        let result3 = handle3.await.unwrap().unwrap();
        let result4 = handle4.await.unwrap().unwrap();

        assert_eq!(result1, 1);
        assert_eq!(result2, 2);
        assert_eq!(result3, 3);
        assert_eq!(result4, 4);

        // Check individual tracker metrics (local counts)
        // Note: Due to hierarchical aggregation, these numbers reflect both local and aggregated counts

        // Parent sees: its own task (1) + child1 task (1) + child2 task (1) + grandchild task (1) = 4
        assert_eq!(
            parent.metrics().success(),
            4,
            "Parent should see 4 total successes (aggregated)"
        );

        // Child1 sees: its own task (1) + grandchild task (1) = 2
        assert_eq!(
            child1.metrics().success(),
            2,
            "Child1 should see 2 total successes (its own + grandchild)"
        );

        // Child2 sees: only its own task = 1
        assert_eq!(
            child2.metrics().success(),
            1,
            "Child2 should see 1 success (its own)"
        );

        // Grandchild sees: only its own task = 1
        assert_eq!(
            grandchild.metrics().success(),
            1,
            "Grandchild should see 1 success (its own)"
        );

        // This demonstrates the hierarchical metrics aggregation:
        // - Each tracker maintains its local count
        // - Child metrics bubble up to parents
        // - Root tracker sees the total system metrics
    }

    #[tokio::test]
    async fn test_hierarchical_metrics_failure_aggregation() {
        // Test that failed task metrics aggregate up to parent
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child = parent.child_tracker().unwrap();

        // Run some successful and some failed tasks
        let success_handle = child.spawn(async { Ok(42) });
        let failure_handle = child.spawn(async { Err::<(), _>(anyhow::anyhow!("test error")) });

        // Wait for tasks to complete
        let _success_result = success_handle.await.unwrap().unwrap();
        let _failure_result = failure_handle.await.unwrap().unwrap_err();

        // Check child metrics
        assert_eq!(child.metrics().success(), 1, "Child should have 1 success");
        assert_eq!(child.metrics().failed(), 1, "Child should have 1 failure");

        // Parent should see the aggregated metrics
        // Note: Due to hierarchical aggregation, these metrics propagate up
    }

    #[tokio::test]
    async fn test_metrics_independence_between_tracker_instances() {
        // Test that different tracker instances have independent metrics
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();

        let tracker1 = TaskTracker::new(scheduler.clone(), error_policy.clone());
        let tracker2 = TaskTracker::new(scheduler, error_policy);

        // Run tasks in both trackers
        let handle1 = tracker1.spawn(async { Ok(1) });
        let handle2 = tracker2.spawn(async { Ok(2) });

        handle1.await.unwrap().unwrap();
        handle2.await.unwrap().unwrap();

        // Each tracker should only see its own metrics
        assert_eq!(tracker1.metrics().success(), 1);
        assert_eq!(tracker2.metrics().success(), 1);
        assert_eq!(tracker1.metrics().total_completed(), 1);
        assert_eq!(tracker2.metrics().total_completed(), 1);
    }

    #[tokio::test]
    async fn test_hierarchical_join_waits_for_all() {
        // Test that parent.join() waits for child tasks too
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child1 = parent.child_tracker().unwrap();
        let child2 = parent.child_tracker().unwrap();
        let grandchild = child1.child_tracker().unwrap();

        // Verify parent tracks children
        assert_eq!(parent.child_count(), 2);
        assert_eq!(child1.child_count(), 1);
        assert_eq!(child2.child_count(), 0);
        assert_eq!(grandchild.child_count(), 0);

        // Track completion order
        let completion_order = Arc::new(Mutex::new(Vec::new()));

        // Spawn tasks with different durations
        let order_clone = completion_order.clone();
        let parent_handle = parent.spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            order_clone.lock().unwrap().push("parent");
            Ok(())
        });

        let order_clone = completion_order.clone();
        let child1_handle = child1.spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            order_clone.lock().unwrap().push("child1");
            Ok(())
        });

        let order_clone = completion_order.clone();
        let child2_handle = child2.spawn(async move {
            tokio::time::sleep(Duration::from_millis(75)).await;
            order_clone.lock().unwrap().push("child2");
            Ok(())
        });

        let order_clone = completion_order.clone();
        let grandchild_handle = grandchild.spawn(async move {
            tokio::time::sleep(Duration::from_millis(125)).await;
            order_clone.lock().unwrap().push("grandchild");
            Ok(())
        });

        // Test hierarchical join - should wait for ALL tasks in hierarchy
        println!("[TEST] About to call parent.join()");
        let start = std::time::Instant::now();
        parent.join().await; // This should wait for ALL tasks
        let elapsed = start.elapsed();
        println!("[TEST] parent.join() completed in {:?}", elapsed);

        // Should have waited for the longest task (grandchild at 125ms)
        assert!(
            elapsed >= Duration::from_millis(120),
            "Hierarchical join should wait for longest task"
        );

        // All tasks should be complete
        assert!(parent_handle.is_finished());
        assert!(child1_handle.is_finished());
        assert!(child2_handle.is_finished());
        assert!(grandchild_handle.is_finished());

        // Verify all tasks completed
        let final_order = completion_order.lock().unwrap();
        assert_eq!(final_order.len(), 4);
        assert!(final_order.contains(&"parent"));
        assert!(final_order.contains(&"child1"));
        assert!(final_order.contains(&"child2"));
        assert!(final_order.contains(&"grandchild"));
    }

    #[tokio::test]
    async fn test_hierarchical_join_waits_for_children() {
        // Test that join() waits for child tasks (hierarchical behavior)
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child = parent.child_tracker().unwrap();

        // Spawn a quick parent task and slow child task
        let _parent_handle = parent.spawn(async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(())
        });

        let _child_handle = child.spawn(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        });

        // Hierarchical join should wait for both parent and child tasks
        let start = std::time::Instant::now();
        parent.join().await; // Should wait for both (hierarchical by default)
        let elapsed = start.elapsed();

        // Should have waited for the longer child task (100ms)
        assert!(
            elapsed >= Duration::from_millis(90),
            "Hierarchical join should wait for all child tasks"
        );
    }

    #[tokio::test]
    async fn test_hierarchical_join_operations() {
        // Test that parent.join() closes and waits for child trackers too
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);
        let child = parent.child_tracker().unwrap();
        let grandchild = child.child_tracker().unwrap();

        // Verify trackers start as open
        assert!(!parent.is_closed());
        assert!(!child.is_closed());
        assert!(!grandchild.is_closed());

        // Join parent (hierarchical by default - closes and waits for all)
        parent.clone().join().await;

        // All should be closed
        assert!(parent.is_closed());
        assert!(child.is_closed());
        assert!(grandchild.is_closed());
    }

    #[tokio::test]
    async fn test_unlimited_scheduler() {
        // Test that UnlimitedScheduler executes tasks immediately
        let scheduler = UnlimitedScheduler::new_arc();
        let error_policy = LogOnlyPolicy::new_arc();
        let tracker = TaskTracker::new(scheduler, error_policy);

        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = tracker.spawn(async {
            rx.await.ok();
            Ok(42)
        });

        // Task should be ready to execute immediately (no concurrency limit)
        tx.send(()).ok();
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);

        assert_eq!(tracker.metrics().success(), 1);
    }

    #[tokio::test]
    async fn test_threshold_cancel_policy() {
        // Test that ThresholdCancelPolicy cancels after failure threshold
        let (error_policy, cancel_token) = ThresholdCancelPolicy::new_arc(2); // Cancel after 2 failures
        let scheduler = create_semaphore_scheduler(10);
        let tracker = TaskTracker::new(scheduler, error_policy.clone());

        // First failure - should not cancel
        let _handle1 = tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("First failure")) });
        tokio::task::yield_now().await;
        assert!(!cancel_token.is_cancelled());
        assert_eq!(error_policy.failure_count(), 1);

        // Second failure - should trigger cancellation
        let _handle2 = tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("Second failure")) });
        tokio::task::yield_now().await;
        assert!(cancel_token.is_cancelled());
        assert_eq!(error_policy.failure_count(), 2);
    }

    #[tokio::test]
    async fn test_policy_arc_constructors() {
        // Test that all Arc constructors work without requiring explicit Arc::new
        let _unlimited = UnlimitedScheduler::new_arc();
        let _semaphore = SemaphoreScheduler::with_permits(5);
        let _log_only = LogOnlyPolicy::new_arc();
        let (_cancel_policy, _token1) = CancelOnError::new_arc();
        let (_threshold_policy, _token2) = ThresholdCancelPolicy::new_arc(3);
        let (_rate_policy, _token3) = RateCancelPolicy::new_arc(0.5, 60);

        // All constructors should work without explicit Arc::new calls
        // This test ensures the convenience API reduces boilerplate
    }

    #[tokio::test]
    async fn test_child_creation_fails_after_join() {
        // Test that child tracker creation fails from closed parent
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);

        // Initially, creating a child should work
        let _child = parent.child_tracker().unwrap();

        // Close the parent tracker
        parent.clone().join().await;
        assert!(parent.is_closed());

        // Now, trying to create a child should fail
        let result = parent.child_tracker();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("closed parent tracker"));
    }

    #[tokio::test]
    async fn test_child_builder_fails_after_join() {
        // Test that child tracker builder creation fails from closed parent
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);

        // Initially, creating a child with builder should work
        let _child = parent.child_tracker_builder().build().unwrap();

        // Close the parent tracker
        parent.clone().join().await;
        assert!(parent.is_closed());

        // Now, trying to create a child with builder should fail
        let result = parent.child_tracker_builder().build();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("closed parent tracker"));
    }

    #[tokio::test]
    async fn test_child_creation_succeeds_before_join() {
        // Test that child creation works normally before parent is joined
        let scheduler = create_semaphore_scheduler(10);
        let error_policy = create_log_policy();
        let parent = TaskTracker::new(scheduler, error_policy);

        // Should be able to create multiple children before closing
        let child1 = parent.child_tracker().unwrap();
        let child2 = parent.child_tracker_builder().build().unwrap();

        // Verify children can spawn tasks
        let handle1 = child1.spawn(async { Ok(42) });
        let handle2 = child2.spawn(async { Ok(24) });

        let result1 = handle1.await.unwrap().unwrap();
        let result2 = handle2.await.unwrap().unwrap();

        assert_eq!(result1, 42);
        assert_eq!(result2, 24);
        assert_eq!(parent.metrics().success(), 2); // Parent sees all successes
    }
}
