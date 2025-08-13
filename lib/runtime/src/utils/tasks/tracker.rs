// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Task Tracker - Hierarchical Task Management System
//!
//! A composable task management system with configurable scheduling and error handling policies.
//! The TaskTracker enables controlled concurrent execution with proper resource management,
//! cancellation semantics, and retry support.
//!
//! ## Architecture Overview
//!
//! The TaskTracker system is built around three core abstractions that compose together:
//!
//! ### 1. **TaskScheduler** - Resource Management
//!
//! Controls when and how tasks acquire execution resources (permits, slots, etc.).
//! Schedulers implement resource acquisition with cancellation support:
//!
//! ```text
//! TaskScheduler::acquire_execution_slot(cancel_token) -> SchedulingResult<ResourceGuard>
//! ```
//!
//! - **Resource Acquisition**: Can be cancelled to avoid unnecessary allocation
//! - **RAII Guards**: Resources are automatically released when guards are dropped
//! - **Pluggable**: Different scheduling policies (unlimited, semaphore, rate-limited, etc.)
//!
//! ### 2. **OnErrorPolicy** - Error Handling
//!
//! Defines how the system responds to task failures:
//!
//! ```text
//! OnErrorPolicy::on_error(error, task_id) -> ErrorResponse
//! ```
//!
//! - **ErrorResponse::Continue**: Log error, continue execution
//! - **ErrorResponse::Cancel**: Cancel tracker and all children
//! - **ErrorResponse::Custom(action)**: Execute custom logic that can return:
//!   - `ActionResult::Continue`: Handle error and continue
//!   - `ActionResult::Cancel`: Cancel tracker
//!   - `ActionResult::ExecuteNext { executor }`: Execute provided task executor
//!
//! ### 3. **Execution Pipeline** - Task Orchestration
//!
//! The execution pipeline coordinates scheduling, execution, and error handling:
//!
//! ```text
//! 1. Acquire resources (scheduler.acquire_execution_slot)
//! 2. Create task future (only after resources acquired)
//! 3. Execute task while holding guard (RAII pattern)
//! 4. Handle errors through policy (with retry support for cancellable tasks)
//! 5. Update metrics and release resources
//! ```
//!
//! ## Key Design Principles
//!
//! ### **Separation of Concerns**
//! - **Scheduling**: When/how to allocate resources
//! - **Execution**: Running tasks with proper resource management
//! - **Error Handling**: Responding to failures with configurable policies
//!
//! ### **Composability**
//! - Schedulers and error policies are independent and can be mixed/matched
//! - Custom policies can be implemented via traits
//! - Execution pipeline handles the coordination automatically
//!
//! ### **Resource Safety**
//! - Resources are acquired before task creation (prevents early execution)
//! - RAII pattern ensures resources are always released
//! - Cancellation is supported during resource acquisition, not during execution
//!
//! ### **Retry Support**
//! - Regular tasks (`spawn`): Cannot be retried (future is consumed)
//! - Cancellable tasks (`spawn_cancellable`): Support retry via `FnMut` closures
//! - Error policies can provide next executors via `ActionResult::ExecuteNext`
//!
//! ## Task Types
//!
//! ### Regular Tasks
//! ```rust
//! let handle = tracker.spawn(async { Ok(42) });
//! ```
//! - Simple futures that run to completion
//! - Cannot be retried (future is consumed on first execution)
//! - Suitable for one-shot operations
//!
//! ### Cancellable Tasks
//! ```rust
//! let handle = tracker.spawn_cancellable(|cancel_token| async move {
//!     // Task can check cancel_token.is_cancelled() or use tokio::select!
//!     CancellableTaskResult::Ok(42)
//! });
//! ```
//! - Receive a `CancellationToken` for cooperative cancellation
//! - Support retry via `FnMut` closures (can be called multiple times)
//! - Return `CancellableTaskResult` to indicate success/cancellation/error
//!
//! ## Hierarchical Structure
//!
//! TaskTrackers form parent-child relationships:
//! - **Metrics**: Child metrics aggregate to parents
//! - **Cancellation**: Parent cancellation propagates to children
//! - **Independence**: Child cancellation doesn't affect parents
//! - **Cleanup**: `join()` waits for all descendants bottom-up
//!
//! ## Metrics and Observability
//!
//! Built-in metrics track task lifecycle:
//! - `issued`: Tasks submitted via spawn methods
//! - `active`: Currently executing tasks
//! - `success/failed/cancelled/rejected`: Final outcomes
//! - `pending`: Issued but not completed (issued - completed)
//! - `queued`: Waiting for resources (pending - active)
//!
//! Optional Prometheus integration available via `RootTaskMetrics`.
//!
//! ## Usage Examples
//!
//! ### Basic Task Execution
//! ```rust
//! use dynamo_runtime::utils::tasks::tracker::*;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let scheduler = SemaphoreScheduler::with_permits(10);
//! let error_policy = LogOnlyPolicy::new();
//! let tracker = TaskTracker::new(scheduler, error_policy)?;
//!
//! let handle = tracker.spawn(async { Ok(42) });
//! let result = handle.await??;
//! assert_eq!(result, 42);
//! # Ok(())
//! # }
//! ```
//!
//! ### Cancellable Tasks with Retry
//! ```rust
//! # use dynamo_runtime::utils::tasks::tracker::*;
//! # use std::sync::Arc;
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! # let scheduler = SemaphoreScheduler::with_permits(10);
//! # let error_policy = LogOnlyPolicy::new();
//! # let tracker = TaskTracker::new(scheduler, error_policy)?;
//! let handle = tracker.spawn_cancellable(|cancel_token| async move {
//!     tokio::select! {
//!         result = do_work() => CancellableTaskResult::Ok(result),
//!         _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
//!     }
//! });
//! # Ok(())
//! # }
//! # async fn do_work() -> i32 { 42 }
//! ```
//!
//! ### Custom Error Policy with Retry
//! ```rust
//! # use dynamo_runtime::utils::tasks::tracker::*;
//! # use std::sync::Arc;
//! # use std::time::Duration;
//! # #[derive(Debug)]
//! struct RetryPolicy {
//!     max_attempts: u32,
//! }
//!
//! impl OnErrorPolicy for RetryPolicy {
//!     fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
//!         Arc::new(RetryPolicy { max_attempts: self.max_attempts })
//!     }
//!
//!     fn on_error(&self, _error: &anyhow::Error, _task_id: TaskId) -> ErrorResponse {
//!         ErrorResponse::Custom(Box::new(RetryAction { max_attempts: self.max_attempts }))
//!     }
//! }
//!
//! # #[derive(Debug)]
//! struct RetryAction { max_attempts: u32 }
//!
//! impl OnErrorAction for RetryAction {
//!     async fn execute(
//!         &self,
//!         _error: &anyhow::Error,
//!         _task_id: TaskId,
//!         attempt_count: u32,
//!         _context: &TaskExecutionContext,
//!     ) -> ActionResult {
//!         if attempt_count < self.max_attempts {
//!             // Create next executor and return it (implementation details in Phase 2-4)
//!             ActionResult::ExecuteNext { executor: /* next_executor */ }
//!         } else {
//!             ActionResult::Continue
//!         }
//!     }
//! }
//! ```
//!
//! ## Future Extensibility
//!
//! The system is designed for extensibility. See the source code for detailed TODO comments
//! describing additional policies that can be implemented:
//! - **Scheduling**: Token bucket rate limiting, adaptive concurrency, memory-aware scheduling
//! - **Error Handling**: Retry with backoff, circuit breakers, dead letter queues
//!
//! Each TODO comment includes complete implementation guidance with data structures,
//! algorithms, and dependencies needed for future contributors.
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
//! let error_policy = ThresholdCancelPolicy::with_threshold(5);
//! let root = std::sync::Arc::new(TaskTracker::builder()
//!     .scheduler(UnlimitedScheduler::new())
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
//! root.join().await;
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
//! let (error_policy, token) = CancelOnError::with_patterns(
//!     vec!["OutOfMemory".to_string(), "DeviceError".to_string()]
//! );
//! let simple = TaskTracker::builder()
//!     .scheduler(SemaphoreScheduler::with_permits(5))
//!     .error_policy(error_policy)
//!     .build()?;
//!
//! // Threshold-based cancellation with monitoring
//! let scheduler = SemaphoreScheduler::with_permits(10);  // Returns Arc<SemaphoreScheduler>
//! let error_policy = ThresholdCancelPolicy::with_threshold(3);  // Returns Arc<Policy>
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
//!     .error_policy(LogOnlyPolicy::new())
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
//! tracker.join().await;
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
//!     LogOnlyPolicy::new(),
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
use std::time::Duration;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker as TokioTaskTracker;
use tracing::{debug, error, warn, Instrument};
use uuid::Uuid;

/// Error type for task execution results
///
/// This enum distinguishes between task cancellation and actual failures,
/// enabling proper metrics tracking and error handling.
#[derive(Error, Debug)]
pub enum TaskError {
    /// Task was cancelled (either via cancellation token or tracker shutdown)
    #[error("Task was cancelled")]
    Cancelled,

    /// Task failed with an error
    #[error(transparent)]
    Failed(#[from] anyhow::Error),

    /// Cannot spawn task on a closed tracker
    #[error("Cannot spawn task on a closed tracker")]
    TrackerClosed,
}

impl TaskError {
    /// Check if this error represents a cancellation
    ///
    /// This is a convenience method for compatibility and readability.
    pub fn is_cancellation(&self) -> bool {
        matches!(self, TaskError::Cancelled)
    }

    /// Check if this error represents a failure
    pub fn is_failure(&self) -> bool {
        matches!(self, TaskError::Failed(_))
    }

    /// Get the underlying anyhow::Error for failures, or a cancellation error for cancellations
    ///
    /// This is provided for compatibility with existing code that expects anyhow::Error.
    pub fn into_anyhow(self) -> anyhow::Error {
        match self {
            TaskError::Failed(err) => err,
            TaskError::Cancelled => anyhow::anyhow!("Task was cancelled"),
            TaskError::TrackerClosed => anyhow::anyhow!("Cannot spawn task on a closed tracker"),
        }
    }
}

/// Trait for tasks that can restart themselves after failure
///
/// This trait allows tasks to define their own restart logic, eliminating the need
/// for complex type erasure and executor management. Tasks implement this trait
/// to provide a clean restart mechanism.
#[async_trait]
pub trait Restartable: Send + Sync + std::fmt::Debug + std::any::Any {
    /// Restart the task after a failure
    ///
    /// This method is called when a task fails and needs to be restarted.
    /// The implementation should create a new attempt at the task execution.
    /// Returns the result in a type-erased Box<dyn Any> for flexibility.
    async fn restart(
        &self,
        cancel_token: CancellationToken,
    ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>>;
}

/// Error type that signals a task can be restarted
///
/// This error type contains a restartable task that can be executed for retry attempts.
/// The task defines its own restart logic through the Restartable trait.
#[derive(Error, Debug)]
#[error("Task failed and can be restarted: {source}")]
pub struct RestartableError {
    /// The underlying error that caused the task to fail
    #[source]
    pub source: anyhow::Error,
    /// The restartable task for retry attempts
    pub restartable: Arc<dyn Restartable + Send + Sync + 'static>,
}

impl RestartableError {
    /// Create a new RestartableError with a restartable task
    ///
    /// The restartable task defines its own restart logic through the Restartable trait.
    pub fn new(
        source: anyhow::Error,
        restartable: Arc<dyn Restartable + Send + Sync + 'static>,
    ) -> Self {
        Self {
            source,
            restartable,
        }
    }

    /// Create a RestartableError and convert it to anyhow::Error
    ///
    /// This is a convenience method for tasks to easily return restartable errors.
    pub fn into_anyhow(
        source: anyhow::Error,
        restartable: Arc<dyn Restartable + Send + Sync + 'static>,
    ) -> anyhow::Error {
        anyhow::Error::new(Self::new(source, restartable))
    }
}

/// Extension trait for extracting RestartableError from anyhow::Error
///
/// This trait provides methods to detect and extract restartable tasks
/// from the type-erased anyhow::Error system.
pub trait RestartableErrorExt {
    /// Extract a restartable task if this error contains one
    ///
    /// Returns the restartable task if the error is a RestartableError,
    /// None otherwise.
    fn extract_restartable(&self) -> Option<Arc<dyn Restartable + Send + Sync + 'static>>;

    /// Check if this error is restartable
    fn is_restartable(&self) -> bool;
}

impl RestartableErrorExt for anyhow::Error {
    fn extract_restartable(&self) -> Option<Arc<dyn Restartable + Send + Sync + 'static>> {
        // Try to downcast to RestartableError
        if let Some(restartable_err) = self.downcast_ref::<RestartableError>() {
            Some(restartable_err.restartable.clone())
        } else {
            None
        }
    }

    fn is_restartable(&self) -> bool {
        self.downcast_ref::<RestartableError>().is_some()
    }
}

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

/// Trait for implementing error handling policies
///
/// Error policies are lightweight, synchronous decision-makers that analyze task failures
/// and return an ErrorResponse telling the TaskTracker what action to take. The TaskTracker
/// handles all the actual work (cancellation, metrics, etc.) based on the policy's response.
///
/// ## Key Design Principles
/// - **Synchronous**: Policies make fast decisions without async operations
/// - **Stateless where possible**: TaskTracker manages cancellation tokens and state
/// - **Composable**: Policies can be combined and nested in hierarchies
/// - **Focused**: Each policy handles one specific error pattern or strategy
pub trait OnErrorPolicy: Send + Sync + std::fmt::Debug {
    /// Create a child policy for a child tracker
    ///
    /// This allows policies to maintain hierarchical relationships,
    /// such as child cancellation tokens or shared circuit breaker state.
    fn create_child(&self) -> Arc<dyn OnErrorPolicy>;

    /// Handle a task failure and return the desired response
    ///
    /// # Arguments
    /// * `error` - The error that occurred
    /// * `task_id` - Unique identifier of the failed task
    ///
    /// # Returns
    /// ErrorResponse indicating how the TaskTracker should handle this failure
    fn on_error(&self, error: &anyhow::Error, task_id: TaskId) -> ErrorResponse;
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

/// Response type for error handling policies
///
/// This enum defines how the TaskTracker should respond to task failures.
/// Currently provides minimal functionality with planned extensions for common patterns.
#[derive(Debug)]
pub enum ErrorResponse {
    /// Continue normal execution, the error counter will be incremented, but task tracker will not be cancelled.
    Continue,

    /// Cancel this tracker and all child trackers
    Cancel,

    /// Execute custom error handling logic with full context access
    Custom(Box<dyn OnErrorAction>),
    // TODO: Future specialized error responses to implement:
    //
    // /// Retry the failed task with configurable strategy
    // /// Implementation: Add RetryStrategy trait with delay(), should_continue(attempt_count),
    // /// release_and_reacquire_resources() methods. TaskTracker handles retry loop with
    // /// attempt counting and resource management. Supports exponential backoff, jitter.
    // /// Usage: ErrorResponse::Retry(Box::new(ExponentialBackoff { max_attempts: 3, base_delay: 100ms }))
    // Retry(Box<dyn RetryStrategy>),
    //
    // /// Execute fallback logic, then follow secondary action
    // /// Implementation: Add FallbackAction trait with execute(error, task_id) -> Result<(), Error>.
    // /// Execute fallback first, then recursively handle the 'then' response based on fallback result.
    // /// Enables patterns like: try fallback, if it works continue, if it fails retry original task.
    // /// Usage: ErrorResponse::Fallback { fallback: Box::new(DefaultValue(42)), then: Box::new(ErrorResponse::Continue) }
    // Fallback { fallback: Box<dyn FallbackAction>, then: Box<ErrorResponse> },
    //
    // /// Restart task with preserved state (for long-running/stateful tasks)
    // /// Implementation: Add TaskState trait for serialize/deserialize state, RestartStrategy trait
    // /// with create_continuation_task(state) -> Future. Task saves checkpoints during execution,
    // /// on error returns StatefulTaskError containing preserved state. Policy can restart from checkpoint.
    // /// Usage: ErrorResponse::RestartWithState { state: checkpointed_state, strategy: Box::new(CheckpointRestart { ... }) }
    // RestartWithState { state: Box<dyn TaskState>, strategy: Box<dyn RestartStrategy> },
}

/// Trait for implementing custom error handling actions
///
/// This provides full access to the task execution context for complex error handling
/// scenarios that don't fit into the built-in response patterns.
#[async_trait]
pub trait OnErrorAction: Send + Sync + std::fmt::Debug {
    /// Execute custom error handling logic
    ///
    /// # Arguments
    /// * `error` - The error that caused the task to fail
    /// * `task_id` - Unique identifier of the failed task
    /// * `attempt_count` - Number of times this task has been attempted (starts at 1)
    /// * `context` - Full execution context with access to scheduler, metrics, etc.
    ///
    /// # Returns
    /// ActionResult indicating what the TaskTracker should do next
    async fn execute(
        &self,
        error: &anyhow::Error,
        task_id: TaskId,
        attempt_count: u32,
        context: &TaskExecutionContext,
    ) -> ActionResult;
}

/// Result of a custom error action execution
#[derive(Debug)]
pub enum ActionResult {
    /// Continue normal execution (error was handled)
    Continue,

    /// Execute the provided restartable task (retry/fallback/transformation)
    ExecuteNext {
        restartable: Arc<dyn Restartable + Send + Sync + 'static>,
    },

    /// Cancel this tracker and all child trackers
    Cancel,
}

/// Execution context provided to custom error actions
///
/// This gives custom actions full access to the task execution environment
/// for implementing complex error handling scenarios.
pub struct TaskExecutionContext {
    /// Scheduler for reacquiring resources or checking state
    pub scheduler: Arc<dyn TaskScheduler>,

    /// Metrics for custom tracking
    pub metrics: Arc<dyn HierarchicalTaskMetrics>,
    // TODO: Future context additions:
    // pub guard: Box<dyn ResourceGuard>,    // Current resource guard (needs Debug impl)
    // pub cancel_token: CancellationToken,  // For implementing custom cancellation
    // pub task_recreation: Box<dyn TaskRecreator>, // For implementing retry/restart
}

/// Result of task execution - unified for both regular and cancellable tasks
#[derive(Debug)]
pub enum TaskExecutionResult<T> {
    /// Task completed successfully
    Success(T),
    /// Task was cancelled (only possible for cancellable tasks)
    Cancelled,
    /// Task failed with an error
    Error(anyhow::Error),
}

/// Trait for executing different types of tasks in a unified way
#[async_trait]
trait TaskExecutor<T>: Send {
    /// Execute the task with the given cancellation token
    async fn execute(&mut self, cancel_token: CancellationToken) -> TaskExecutionResult<T>;

    /// Whether this task type supports retry (can be called multiple times)
    fn supports_retry(&self) -> bool;
}

/// Task executor for regular (non-cancellable) tasks
struct RegularTaskExecutor<F, T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    future: Option<F>,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> RegularTaskExecutor<F, T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    fn new(future: F) -> Self {
        Self {
            future: Some(future),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<F, T> TaskExecutor<T> for RegularTaskExecutor<F, T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    async fn execute(&mut self, _cancel_token: CancellationToken) -> TaskExecutionResult<T> {
        if let Some(future) = self.future.take() {
            match future.await {
                Ok(value) => TaskExecutionResult::Success(value),
                Err(error) => TaskExecutionResult::Error(error),
            }
        } else {
            // This should never happen since regular tasks don't support retry
            TaskExecutionResult::Error(anyhow::anyhow!("Regular task already consumed"))
        }
    }

    fn supports_retry(&self) -> bool {
        false
    }
}

/// Task executor for cancellable tasks
struct CancellableTaskExecutor<F, Fut, T>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
    T: Send + 'static,
{
    task_fn: F,
}

impl<F, Fut, T> CancellableTaskExecutor<F, Fut, T>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
    T: Send + 'static,
{
    fn new(task_fn: F) -> Self {
        Self { task_fn }
    }
}

#[async_trait]
impl<F, Fut, T> TaskExecutor<T> for CancellableTaskExecutor<F, Fut, T>
where
    F: FnMut(CancellationToken) -> Fut + Send + 'static,
    Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
    T: Send + 'static,
{
    async fn execute(&mut self, cancel_token: CancellationToken) -> TaskExecutionResult<T> {
        let future = (self.task_fn)(cancel_token);
        match future.await {
            CancellableTaskResult::Ok(value) => TaskExecutionResult::Success(value),
            CancellableTaskResult::Cancelled => TaskExecutionResult::Cancelled,
            CancellableTaskResult::Err(error) => TaskExecutionResult::Error(error),
        }
    }

    fn supports_retry(&self) -> bool {
        true // Cancellable tasks support retry via RestartableError
    }
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
/// Resource guard for task execution
///
/// This trait represents resources (permits, slots, etc.) acquired from a scheduler
/// that must be held during task execution. The guard automatically releases
/// resources when dropped, implementing proper RAII semantics.
///
/// Guards are returned by `TaskScheduler::acquire_execution_slot()` and must
/// be held in scope while the task executes to ensure resources remain allocated.
pub trait ResourceGuard: Send + 'static {
    // Marker trait - resources are released via Drop on the concrete type
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
    /// * `cancel_token` - [`CancellationToken`] for scheduler-level cancellation
    ///
    /// # Returns
    /// * `SchedulingResult::Execute(guard)` - Resources acquired, ready to execute
    /// * `SchedulingResult::Cancelled` - Cancelled before or during resource acquisition
    /// * `SchedulingResult::Rejected(reason)` - Resources unavailable or policy violation
    async fn acquire_execution_slot(
        &self,
        cancel_token: CancellationToken,
    ) -> SchedulingResult<Box<dyn ResourceGuard>>;
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
    /// Create a new ChildTrackerBuilder
    pub fn new(parent: &'parent TaskTracker) -> Self {
        Self {
            parent,
            scheduler: None,
            error_policy: None,
        }
    }

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
    ///     .scheduler(SemaphoreScheduler::with_permits(5))
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
    ///     .error_policy(LogOnlyPolicy::new())
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
    pub fn build(self) -> anyhow::Result<TaskTracker> {
        // Validate that parent tracker is still active
        if self.parent.is_closed() {
            return Err(anyhow::anyhow!(
                "Cannot create child tracker from closed parent tracker"
            ));
        }

        let parent = self.parent.0.clone();

        let child_cancel_token = parent.cancel_token.child_token();
        let child_metrics = Arc::new(ChildTaskMetrics::new(parent.metrics.clone()));

        // Use provided scheduler or inherit from parent
        let scheduler = self.scheduler.unwrap_or_else(|| parent.scheduler.clone());

        // Use provided error policy or create child from parent's
        let error_policy = self
            .error_policy
            .unwrap_or_else(|| parent.error_policy.create_child());

        let child = Arc::new(TaskTrackerInner {
            tokio_tracker: TokioTaskTracker::new(),
            parent: None, // No parent reference needed for hierarchical operations
            scheduler,
            error_policy,
            metrics: child_metrics,
            cancel_token: child_cancel_token,
            children: RwLock::new(Vec::new()),
        });

        // Register this child with the parent for hierarchical operations
        parent
            .children
            .write()
            .unwrap()
            .push(Arc::downgrade(&child));

        // Periodically clean up dead children to prevent unbounded growth
        parent.cleanup_dead_children();

        Ok(TaskTracker(child))
    }
}

/// Internal data for TaskTracker
///
/// This struct contains all the actual state and functionality of a TaskTracker.
/// TaskTracker itself is just a wrapper around Arc<TaskTrackerInner>.
struct TaskTrackerInner {
    /// Tokio's task tracker for lifecycle management
    tokio_tracker: TokioTaskTracker,
    /// Parent tracker (None for root)
    parent: Option<Arc<TaskTrackerInner>>,
    /// Scheduling policy (shared with children by default)
    scheduler: Arc<dyn TaskScheduler>,
    /// Error handling policy (child-specific via create_child)
    error_policy: Arc<dyn OnErrorPolicy>,
    /// Metrics for this tracker
    metrics: Arc<dyn HierarchicalTaskMetrics>,
    /// Cancellation token for this tracker (always present)
    cancel_token: CancellationToken,
    /// List of child trackers for hierarchical operations
    children: RwLock<Vec<Weak<TaskTrackerInner>>>,
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
/// // Create a task tracker with semaphore-based scheduling
/// let scheduler = SemaphoreScheduler::with_permits(3);
/// let policy = LogOnlyPolicy::new();
/// let root = TaskTracker::builder()
///     .scheduler(scheduler)
///     .error_policy(policy)
///     .build()?;
///
/// // Spawn some tasks
/// let handle1 = root.spawn(async { Ok(1) });
/// let handle2 = root.spawn(async { Ok(2) });
///
/// // Get results and join all tasks
/// let result1 = handle1.await.unwrap().unwrap();
/// let result2 = handle2.await.unwrap().unwrap();
/// assert_eq!(result1, 1);
/// assert_eq!(result2, 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct TaskTracker(Arc<TaskTrackerInner>);

/// Builder for TaskTracker
#[derive(Default)]
pub struct TaskTrackerBuilder {
    scheduler: Option<Arc<dyn TaskScheduler>>,
    error_policy: Option<Arc<dyn OnErrorPolicy>>,
    metrics: Option<Arc<dyn HierarchicalTaskMetrics>>,
    cancel_token: Option<CancellationToken>,
}

impl TaskTrackerBuilder {
    /// Set the scheduler for this TaskTracker
    pub fn scheduler(mut self, scheduler: Arc<dyn TaskScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set the error policy for this TaskTracker
    pub fn error_policy(mut self, error_policy: Arc<dyn OnErrorPolicy>) -> Self {
        self.error_policy = Some(error_policy);
        self
    }

    /// Set custom metrics for this TaskTracker
    pub fn metrics(mut self, metrics: Arc<dyn HierarchicalTaskMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Set the cancellation token for this TaskTracker
    pub fn cancel_token(mut self, cancel_token: CancellationToken) -> Self {
        self.cancel_token = Some(cancel_token);
        self
    }

    /// Build the TaskTracker
    pub fn build(self) -> anyhow::Result<TaskTracker> {
        let scheduler = self
            .scheduler
            .ok_or_else(|| anyhow::anyhow!("TaskTracker requires a scheduler"))?;

        let error_policy = self
            .error_policy
            .ok_or_else(|| anyhow::anyhow!("TaskTracker requires an error policy"))?;

        let metrics = self
            .metrics
            .unwrap_or_else(|| Arc::new(DefaultRootTaskMetrics::new()));

        let cancel_token = self.cancel_token.unwrap_or_default();

        let inner = TaskTrackerInner {
            tokio_tracker: TokioTaskTracker::new(),
            parent: None,
            scheduler,
            error_policy,
            metrics,
            cancel_token,
            children: RwLock::new(Vec::new()),
        };

        Ok(TaskTracker(Arc::new(inner)))
    }
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
    /// # fn main() -> anyhow::Result<()> {
    /// let scheduler = SemaphoreScheduler::with_permits(10);
    /// let error_policy = LogOnlyPolicy::new();
    /// let tracker = TaskTracker::builder()
    ///     .scheduler(scheduler)
    ///     .error_policy(error_policy)
    ///     .build()?;
    /// # Ok(())
    /// # }
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
    /// # fn main() -> anyhow::Result<()> {
    /// let scheduler = SemaphoreScheduler::with_permits(10);
    /// let error_policy = LogOnlyPolicy::new();
    /// let tracker = TaskTracker::new(scheduler, error_policy)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        scheduler: Arc<dyn TaskScheduler>,
        error_policy: Arc<dyn OnErrorPolicy>,
    ) -> anyhow::Result<Self> {
        Self::builder()
            .scheduler(scheduler)
            .error_policy(error_policy)
            .build()
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
    /// let scheduler = SemaphoreScheduler::with_permits(10);
    /// let error_policy = LogOnlyPolicy::new();
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
    ) -> anyhow::Result<Self> {
        let prometheus_metrics = Arc::new(RootTaskMetrics::new(registry, component_name)?);

        Self::builder()
            .scheduler(scheduler)
            .error_policy(error_policy)
            .metrics(prometheus_metrics)
            .build()
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
    pub fn child_tracker(&self) -> anyhow::Result<TaskTracker> {
        Ok(TaskTracker(self.0.child_tracker()?))
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
    ///     .scheduler(SemaphoreScheduler::with_permits(5))
    ///     .build().unwrap();
    ///
    /// // Custom error policy, inherit scheduler
    /// let child2 = root_tracker.child_tracker_builder()
    ///     .error_policy(LogOnlyPolicy::new())
    ///     .build().unwrap();
    ///
    /// // Both custom
    /// let child3 = root_tracker.child_tracker_builder()
    ///     .scheduler(SemaphoreScheduler::with_permits(3))
    ///     .error_policy(LogOnlyPolicy::new())
    ///     .build().unwrap();
    /// # }
    /// ```
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
    /// # Panics
    /// Panics if the tracker has been closed. This indicates a programming error
    /// where tasks are being spawned after the tracker lifecycle has ended.
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
    /// let result = handle.await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn<F, T>(&self, future: F) -> JoinHandle<Result<T, TaskError>>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.0
            .spawn(future)
            .expect("TaskTracker must not be closed when spawning tasks")
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
    /// # Panics
    /// Panics if the tracker has been closed. This indicates a programming error
    /// where tasks are being spawned after the tracker lifecycle has ended.
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
    /// let result = handle.await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn spawn_cancellable<F, Fut, T>(&self, task_fn: F) -> JoinHandle<Result<T, TaskError>>
    where
        F: FnMut(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.0
            .spawn_cancellable(task_fn)
            .expect("TaskTracker must not be closed when spawning tasks")
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
        self.0.metrics.as_ref()
    }

    /// Cancel this tracker and all its tasks
    ///
    /// This will signal cancellation to all currently running tasks and prevent new tasks from being spawned.
    /// The cancellation is immediate and forceful.
    ///
    /// # Example
    /// ```rust
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # async fn example(tracker: TaskTracker) -> anyhow::Result<()> {
    /// // Spawn a long-running task
    /// let handle = tracker.spawn_cancellable(|cancel_token| async move {
    ///     tokio::select! {
    ///         _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
    ///             dynamo_runtime::utils::tasks::tracker::CancellableTaskResult::Ok(42)
    ///         }
    ///         _ = cancel_token.cancelled() => {
    ///             dynamo_runtime::utils::tasks::tracker::CancellableTaskResult::Cancelled
    ///         }
    ///     }
    /// }).await?;
    ///
    /// // Cancel the tracker (and thus the task)
    /// tracker.cancel();
    /// # Ok(())
    /// # }
    /// ```
    pub fn cancel(&self) {
        self.0.cancel();
    }

    /// Check if this tracker is closed
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Get the cancellation token for this tracker
    ///
    /// This allows external code to observe or trigger cancellation of this tracker.
    ///
    /// # Example
    /// ```rust
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
        self.0.cancellation_token()
    }

    /// Get the number of active child trackers
    ///
    /// This counts only child trackers that are still alive (not dropped).
    /// Dropped child trackers are automatically cleaned up.
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
        self.0.child_count()
    }

    /// Create a child tracker builder with custom configuration
    ///
    /// This provides fine-grained control over child tracker creation,
    /// allowing you to override the scheduler or error policy while
    /// maintaining the parent-child relationship.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::sync::Semaphore;
    /// # use dynamo_runtime::utils::tasks::tracker::{TaskTracker, SemaphoreScheduler, LogOnlyPolicy};
    /// # fn example(parent: &TaskTracker) {
    /// // Custom scheduler, inherit error policy
    /// let child1 = parent.child_tracker_builder()
    ///     .scheduler(SemaphoreScheduler::with_permits(5))
    ///     .build().unwrap();
    ///
    /// // Custom error policy, inherit scheduler
    /// let child2 = parent.child_tracker_builder()
    ///     .error_policy(LogOnlyPolicy::new())
    ///     .build().unwrap();
    ///
    /// // Inherit both policies from parent
    /// let child3 = parent.child_tracker_builder()
    ///     .build().unwrap();
    /// # }
    /// ```
    pub fn child_tracker_builder(&self) -> ChildTrackerBuilder<'_> {
        ChildTrackerBuilder::new(self)
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
    /// # use dynamo_runtime::utils::tasks::tracker::TaskTracker;
    /// # async fn example(tracker: TaskTracker) {
    /// tracker.join().await;
    /// # }
    /// ```
    pub async fn join(&self) {
        self.0.join().await
    }
}

impl TaskTrackerInner {
    /// Creates child tracker with inherited scheduler/policy, independent metrics, and hierarchical cancellation
    fn child_tracker(self: &Arc<Self>) -> anyhow::Result<Arc<TaskTrackerInner>> {
        // Validate that parent tracker is still active
        if self.is_closed() {
            return Err(anyhow::anyhow!(
                "Cannot create child tracker from closed parent tracker"
            ));
        }

        let child_cancel_token = self.cancel_token.child_token();
        let child_metrics = Arc::new(ChildTaskMetrics::new(self.metrics.clone()));

        let child = Arc::new(TaskTrackerInner {
            tokio_tracker: TokioTaskTracker::new(),
            parent: Some(self.clone()),
            scheduler: self.scheduler.clone(),
            error_policy: self.error_policy.create_child(),
            metrics: child_metrics,
            cancel_token: child_cancel_token,
            children: RwLock::new(Vec::new()),
        });

        // Register this child with the parent for hierarchical operations
        self.children.write().unwrap().push(Arc::downgrade(&child));

        // Periodically clean up dead children to prevent unbounded growth
        self.cleanup_dead_children();

        Ok(child)
    }

    /// Spawn implementation - validates tracker state, generates task ID, applies policies, and tracks execution
    fn spawn<F, T>(
        self: &Arc<Self>,
        future: F,
    ) -> Result<JoinHandle<Result<T, TaskError>>, TaskError>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Validate tracker is not closed
        if self.tokio_tracker.is_closed() {
            return Err(TaskError::TrackerClosed);
        }

        // Generate a unique task ID
        let task_id = self.generate_task_id();

        // Increment issued counter immediately when task is submitted
        self.metrics.increment_issued();

        // Clone the inner Arc to move into the task
        let inner = self.clone();

        // Wrap the user's future with our scheduling and error handling
        let wrapped_future =
            async move { Self::execute_with_policies(task_id, future, inner).await };

        // Let tokio handle the actual task tracking
        Ok(self.tokio_tracker.spawn(wrapped_future))
    }

    /// Spawn cancellable implementation - validates state, provides cancellation token, handles CancellableTaskResult
    fn spawn_cancellable<F, Fut, T>(
        self: &Arc<Self>,
        task_fn: F,
    ) -> Result<JoinHandle<Result<T, TaskError>>, TaskError>
    where
        F: FnMut(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Validate tracker is not closed
        if self.tokio_tracker.is_closed() {
            return Err(TaskError::TrackerClosed);
        }

        // Generate a unique task ID
        let task_id = self.generate_task_id();

        // Increment issued counter immediately when task is submitted
        self.metrics.increment_issued();

        // Clone the inner Arc to move into the task
        let inner = self.clone();

        // Use the new execution pipeline that defers task creation until after guard acquisition
        let wrapped_future =
            async move { Self::execute_cancellable_with_policies(task_id, task_fn, inner).await };

        // Let tokio handle the actual task tracking
        Ok(self.tokio_tracker.spawn(wrapped_future))
    }

    /// Cancel this tracker and all its tasks - implementation
    fn cancel(&self) {
        // Close the tracker to prevent new tasks
        self.tokio_tracker.close();

        // Cancel our own token
        self.cancel_token.cancel();
    }

    /// Returns true if the underlying tokio tracker is closed
    fn is_closed(&self) -> bool {
        self.tokio_tracker.is_closed()
    }

    /// Generates a unique task ID using TaskId::new()
    fn generate_task_id(&self) -> TaskId {
        TaskId::new()
    }

    /// Removes dead weak references from children list to prevent memory leaks
    fn cleanup_dead_children(&self) {
        let mut children_guard = self.children.write().unwrap();
        children_guard.retain(|weak| weak.upgrade().is_some());
    }

    /// Returns a clone of the cancellation token
    fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Counts active child trackers (filters out dead weak references)
    fn child_count(&self) -> usize {
        let children_guard = self.children.read().unwrap();
        children_guard
            .iter()
            .filter(|weak| weak.upgrade().is_some())
            .count()
    }

    /// Join implementation - closes all trackers in hierarchy then waits for task completion using stack-safe traversal
    async fn join(self: &Arc<Self>) {
        // Fast path for leaf trackers (no children)
        let is_leaf = {
            let children_guard = self.children.read().unwrap();
            children_guard.is_empty()
        };

        if is_leaf {
            self.tokio_tracker.close();
            self.tokio_tracker.wait().await;
            return;
        }

        // Stack-safe traversal for deep hierarchies
        // Processes children before parents to ensure proper shutdown order
        let trackers = self.collect_hierarchy();
        for t in trackers {
            t.tokio_tracker.close();
            t.tokio_tracker.wait().await;
        }
    }

    /// Collects hierarchy using iterative DFS, returns Vec in post-order (children before parents) for safe shutdown
    fn collect_hierarchy(self: &Arc<TaskTrackerInner>) -> Vec<Arc<TaskTrackerInner>> {
        let mut result = Vec::new();
        let mut stack = vec![self.clone()];
        let mut visited = HashSet::new();

        // Collect all trackers using depth-first search
        while let Some(tracker) = stack.pop() {
            let tracker_ptr = Arc::as_ptr(&tracker) as usize;
            if visited.contains(&tracker_ptr) {
                continue;
            }
            visited.insert(tracker_ptr);

            // Add current tracker to result
            result.push(tracker.clone());

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

    /// Execute a regular task with scheduling and error handling policies
    #[tracing::instrument(level = "debug", skip_all, fields(task_id = %task_id))]
    async fn execute_with_policies<F, T>(
        task_id: TaskId,
        future: F,
        inner: Arc<TaskTrackerInner>,
    ) -> Result<T, TaskError>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Wrap regular future in a task executor that doesn't support retry
        let task_executor = RegularTaskExecutor::new(future);
        Self::execute_with_retry_loop(task_id, task_executor, inner).await
    }

    /// Execute a cancellable task with scheduling and error handling policies
    #[tracing::instrument(level = "debug", skip_all, fields(task_id = %task_id))]
    async fn execute_cancellable_with_policies<F, Fut, T>(
        task_id: TaskId,
        task_fn: F,
        inner: Arc<TaskTrackerInner>,
    ) -> Result<T, TaskError>
    where
        F: FnMut(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = CancellableTaskResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Wrap cancellable task function in a task executor that supports retry
        let task_executor = CancellableTaskExecutor::new(task_fn);
        Self::execute_with_retry_loop(task_id, task_executor, inner).await
    }

    /// Core execution loop with retry support - unified for both task types
    #[tracing::instrument(level = "debug", skip_all, fields(task_id = %task_id))]
    async fn execute_with_retry_loop<E, T>(
        task_id: TaskId,
        initial_executor: E,
        inner: Arc<TaskTrackerInner>,
    ) -> Result<T, TaskError>
    where
        E: TaskExecutor<T> + Send + 'static,
        T: Send + 'static,
    {
        debug!("Starting task execution");

        // RAII guard for active counter - increments on creation, decrements on drop
        struct ActiveCountGuard {
            metrics: Arc<dyn HierarchicalTaskMetrics>,
            is_active: bool,
        }

        impl ActiveCountGuard {
            fn new(metrics: Arc<dyn HierarchicalTaskMetrics>) -> Self {
                Self {
                    metrics,
                    is_active: false,
                }
            }

            fn activate(&mut self) {
                if !self.is_active {
                    self.metrics.increment_active();
                    self.is_active = true;
                }
            }
        }

        impl Drop for ActiveCountGuard {
            fn drop(&mut self) {
                if self.is_active {
                    self.metrics.decrement_active();
                }
            }
        }

        // Current executable - either the original TaskExecutor or a Restartable
        enum CurrentExecutable<E>
        where
            E: Send + 'static,
        {
            TaskExecutor(E),
            Restartable(Arc<dyn Restartable + Send + Sync + 'static>),
        }

        let mut current_executable = CurrentExecutable::TaskExecutor(initial_executor);
        let mut attempt_count = 1u32;
        let mut active_guard = ActiveCountGuard::new(inner.metrics.clone());

        loop {
            // Acquire execution slot through scheduler with cancellation token
            let guard_result = async {
                inner
                    .scheduler
                    .acquire_execution_slot(inner.cancel_token.child_token())
                    .await
            }
            .instrument(tracing::debug_span!("scheduler_resource_acquisition"))
            .await;

            match guard_result {
                SchedulingResult::Execute(_guard) => {
                    // Activate the RAII guard only once when we successfully acquire resources
                    active_guard.activate();

                    // Execute the current executable while holding the guard (RAII pattern)
                    let execution_result = async {
                        debug!("Executing task with acquired resources");
                        match &mut current_executable {
                            CurrentExecutable::TaskExecutor(executor) => {
                                executor.execute(inner.cancel_token.child_token()).await
                            }
                            CurrentExecutable::Restartable(restartable) => {
                                // Execute restartable and handle type erasure
                                match restartable.restart(inner.cancel_token.child_token()).await {
                                    TaskExecutionResult::Success(result) => {
                                        // Try to downcast the result to the expected type T
                                        if let Ok(typed_result) = result.downcast::<T>() {
                                            TaskExecutionResult::Success(*typed_result)
                                        } else {
                                            // Type mismatch - this shouldn't happen with proper usage
                                            let type_error = anyhow::anyhow!(
                                                "Restartable task returned wrong type"
                                            );
                                            error!(
                                                ?type_error,
                                                "Type mismatch in restartable task result"
                                            );
                                            TaskExecutionResult::Error(type_error)
                                        }
                                    }
                                    TaskExecutionResult::Cancelled => {
                                        TaskExecutionResult::Cancelled
                                    }
                                    TaskExecutionResult::Error(error) => {
                                        TaskExecutionResult::Error(error)
                                    }
                                }
                            }
                        }
                    }
                    .instrument(tracing::debug_span!("task_execution"))
                    .await;

                    // Active counter will be decremented automatically when active_guard drops

                    match execution_result {
                        TaskExecutionResult::Success(value) => {
                            inner.metrics.increment_success();
                            debug!("Task completed successfully");
                            return Ok(value);
                        }
                        TaskExecutionResult::Cancelled => {
                            inner.metrics.increment_cancelled();
                            debug!("Task was cancelled during execution");
                            return Err(TaskError::Cancelled);
                        }
                        TaskExecutionResult::Error(error) => {
                            debug!(
                                attempt_count,
                                "Task failed - handling error through policy - {error:?}"
                            );

                            // Handle the error through the policy system
                            let action_result =
                                Self::handle_task_error(&error, task_id, attempt_count, &inner)
                                    .await;

                            match action_result {
                                ActionResult::Continue => {
                                    inner.metrics.increment_failed();
                                    debug!("Policy decided to continue - task failed {error:?}");
                                    return Err(TaskError::Failed(error));
                                }
                                ActionResult::Cancel => {
                                    inner.metrics.increment_failed();
                                    warn!("Policy triggered cancellation - {error:?}");
                                    inner.cancel();
                                    return Err(TaskError::Failed(error));
                                }
                                ActionResult::ExecuteNext { restartable } => {
                                    debug!(
                                        "Policy provided next executable - continuing loop - {error:?}"
                                    );

                                    // Update current executable and continue the loop
                                    current_executable =
                                        CurrentExecutable::Restartable(restartable);
                                    attempt_count += 1;
                                    continue; // Continue the main loop with the new executable
                                }
                            }
                        }
                    }
                }
                SchedulingResult::Cancelled => {
                    inner.metrics.increment_cancelled();
                    debug!("Task was cancelled during resource acquisition");
                    return Err(TaskError::Cancelled);
                }
                SchedulingResult::Rejected(reason) => {
                    inner.metrics.increment_rejected();
                    debug!(reason, "Task was rejected by scheduler");
                    return Err(TaskError::Failed(anyhow::anyhow!(
                        "Task rejected: {}",
                        reason
                    )));
                }
            }
        }
    }

    /// Handle task errors through the error policy and return the action to take
    async fn handle_task_error(
        error: &anyhow::Error,
        task_id: TaskId,
        attempt_count: u32,
        inner: &Arc<TaskTrackerInner>,
    ) -> ActionResult {
        // First, check if this is a RestartableError (task-driven retry)
        if let Some(restartable_err) = error.downcast_ref::<RestartableError>() {
            debug!(
                task_id = %task_id,
                attempt_count,
                "Task provided RestartableError with restartable task - {error:?}"
            );

            // Task has provided a restartable implementation for the next attempt
            // Clone the Arc to return it in ActionResult::ExecuteNext
            let restartable = restartable_err.restartable.clone();

            return ActionResult::ExecuteNext { restartable };
        }

        // Not a RestartableError, proceed with normal policy handling
        let response = inner.error_policy.on_error(error, task_id);

        match response {
            ErrorResponse::Continue => ActionResult::Continue,
            ErrorResponse::Cancel => ActionResult::Cancel,
            ErrorResponse::Custom(action) => {
                debug!("Task failed - executing custom action - {error:?}");

                // Create execution context for the action
                let context = TaskExecutionContext {
                    scheduler: inner.scheduler.clone(),
                    metrics: inner.metrics.clone(),
                };

                // Execute the custom action asynchronously
                let action_result = action
                    .execute(error, task_id, attempt_count, &context)
                    .await;
                debug!(?action_result, "Custom action completed");

                action_result
            }
        }
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
/// This guard represents "unlimited" resources - no actual resource constraints.
/// Since there are no resources to manage, this guard is essentially a no-op.
#[derive(Debug)]
pub struct UnlimitedGuard;

impl ResourceGuard for UnlimitedGuard {
    // No resources to manage - marker trait implementation only
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
    /// Create a new unlimited scheduler returning Arc
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl Default for UnlimitedScheduler {
    fn default() -> Self {
        UnlimitedScheduler
    }
}

#[async_trait]
impl TaskScheduler for UnlimitedScheduler {
    async fn acquire_execution_slot(
        &self,
        cancel_token: CancellationToken,
    ) -> SchedulingResult<Box<dyn ResourceGuard>> {
        debug!("Acquiring execution slot (unlimited scheduler)");

        // Check for cancellation before allocating resources
        if cancel_token.is_cancelled() {
            debug!("Task cancelled before acquiring execution slot");
            return SchedulingResult::Cancelled;
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

impl ResourceGuard for SemaphoreGuard {
    // Permit is automatically released when the guard is dropped
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
        Arc::new(Self::new(Arc::new(Semaphore::new(permits))))
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
        cancel_token: CancellationToken,
    ) -> SchedulingResult<Box<dyn ResourceGuard>> {
        debug!("Acquiring semaphore permit");

        // Check for cancellation before attempting to acquire semaphore
        if cancel_token.is_cancelled() {
            debug!("Task cancelled before acquiring semaphore permit");
            return SchedulingResult::Cancelled;
        }

        // Try to acquire a permit, with cancellation support
        let permit = {
            tokio::select! {
                result = self.semaphore.clone().acquire_owned() => {
                    match result {
                        Ok(permit) => permit,
                        Err(_) => return SchedulingResult::Cancelled,
                    }
                }
                _ = cancel_token.cancelled() => {
                    debug!("Task cancelled while waiting for semaphore permit");
                    return SchedulingResult::Cancelled;
                }
            }
        };

        debug!("Acquired semaphore permit");
        SchedulingResult::Execute(Box::new(SemaphoreGuard { _permit: permit }))
    }
}

/// Error policy that triggers cancellation based on error patterns
///
/// This policy analyzes error messages and returns `ErrorResponse::Cancel` when:
/// - No patterns are specified (cancels on any error)
/// - Error message matches one of the specified patterns
///
/// The TaskTracker handles the actual cancellation - this policy just makes the decision.
///
/// # Example
/// ```rust
/// # use dynamo_runtime::utils::tasks::tracker::CancelOnError;
/// // Cancel on any error
/// let policy = CancelOnError::new();
///
/// // Cancel only on specific error patterns
/// let (policy, _token) = CancelOnError::with_patterns(
///     vec!["OutOfMemory".to_string(), "DeviceError".to_string()]
/// );
/// ```
#[derive(Debug)]
pub struct CancelOnError {
    error_patterns: Vec<String>,
}

impl CancelOnError {
    /// Create a new cancel-on-error policy that cancels on any error
    ///
    /// Returns a policy with no error patterns, meaning it will cancel the TaskTracker
    /// on any task failure.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            error_patterns: vec![], // Empty patterns = cancel on any error
        })
    }

    /// Create a new cancel-on-error policy with custom error patterns, returning Arc and token
    ///
    /// # Arguments
    /// * `error_patterns` - List of error message patterns that trigger cancellation
    pub fn with_patterns(error_patterns: Vec<String>) -> (Arc<Self>, CancellationToken) {
        let token = CancellationToken::new();
        let policy = Arc::new(Self { error_patterns });
        (policy, token)
    }
}

#[async_trait]
impl OnErrorPolicy for CancelOnError {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token - when parent cancels, child cancels too
        // When child cancels, parent is unaffected
        Arc::new(CancelOnError {
            error_patterns: self.error_patterns.clone(),
        })
    }

    fn on_error(&self, error: &anyhow::Error, task_id: TaskId) -> ErrorResponse {
        error!(?task_id, "Task failed - {error:?}");

        if self.error_patterns.is_empty() {
            return ErrorResponse::Cancel;
        }

        // Check if this error should trigger cancellation
        let error_str = error.to_string();
        let should_cancel = self
            .error_patterns
            .iter()
            .any(|pattern| error_str.contains(pattern));

        if should_cancel {
            ErrorResponse::Cancel
        } else {
            ErrorResponse::Continue
        }
    }
}

/// Simple error policy that only logs errors
///
/// This policy does not trigger cancellation and is useful for
/// non-critical tasks or when you want to handle errors externally.
#[derive(Debug)]
pub struct LogOnlyPolicy;

impl LogOnlyPolicy {
    /// Create a new log-only policy returning Arc
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl Default for LogOnlyPolicy {
    fn default() -> Self {
        LogOnlyPolicy
    }
}

impl OnErrorPolicy for LogOnlyPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Simple policies can just clone themselves
        Arc::new(LogOnlyPolicy)
    }

    fn on_error(&self, error: &anyhow::Error, task_id: TaskId) -> ErrorResponse {
        error!(?task_id, "Task failed - logging only - {error:?}");
        ErrorResponse::Continue
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
/// let policy = ThresholdCancelPolicy::with_threshold(5);
/// ```
#[derive(Debug)]
pub struct ThresholdCancelPolicy {
    max_failures: usize,
    failure_count: AtomicU64,
}

impl ThresholdCancelPolicy {
    /// Create a new threshold cancel policy with specified failure threshold, returning Arc and token
    ///
    /// # Arguments
    /// * `max_failures` - Maximum number of failures before cancellation
    pub fn with_threshold(max_failures: usize) -> Arc<Self> {
        Arc::new(Self {
            max_failures,
            failure_count: AtomicU64::new(0),
        })
    }

    /// Get the current failure count
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }
}

impl OnErrorPolicy for ThresholdCancelPolicy {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token and inherits the same failure threshold
        Arc::new(ThresholdCancelPolicy {
            max_failures: self.max_failures,
            failure_count: AtomicU64::new(0), // Child starts with fresh count
        })
    }

    fn on_error(&self, error: &anyhow::Error, task_id: TaskId) -> ErrorResponse {
        error!(?task_id, "Task failed - {error:?}");

        let current_failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        if current_failures >= self.max_failures as u64 {
            warn!(
                ?task_id,
                current_failures,
                max_failures = self.max_failures,
                "Failure threshold exceeded, triggering cancellation"
            );
            ErrorResponse::Cancel
        } else {
            debug!(
                ?task_id,
                current_failures,
                max_failures = self.max_failures,
                "Task failed, tracking failure count"
            );
            ErrorResponse::Continue
        }
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
/// let (policy, token) = RateCancelPolicy::builder()
///     .rate(0.5)
///     .window_secs(60)
///     .build();
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
    /// Create a builder for rate-based cancel policy
    pub fn builder() -> RateCancelPolicyBuilder {
        RateCancelPolicyBuilder::new()
    }
}

/// Builder for RateCancelPolicy
pub struct RateCancelPolicyBuilder {
    max_failure_rate: Option<f32>,
    window_secs: Option<u64>,
}

impl RateCancelPolicyBuilder {
    fn new() -> Self {
        Self {
            max_failure_rate: None,
            window_secs: None,
        }
    }

    /// Set the maximum failure rate (0.0 to 1.0) before cancellation
    pub fn rate(mut self, max_failure_rate: f32) -> Self {
        self.max_failure_rate = Some(max_failure_rate);
        self
    }

    /// Set the time window in seconds for rate calculation
    pub fn window_secs(mut self, window_secs: u64) -> Self {
        self.window_secs = Some(window_secs);
        self
    }

    /// Build the policy, returning Arc and cancellation token
    pub fn build(self) -> (Arc<RateCancelPolicy>, CancellationToken) {
        let max_failure_rate = self.max_failure_rate.expect("rate must be set");
        let window_secs = self.window_secs.expect("window_secs must be set");

        let token = CancellationToken::new();
        let policy = Arc::new(RateCancelPolicy {
            cancel_token: token.clone(),
            max_failure_rate,
            window_secs,
        });
        (policy, token)
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

    fn on_error(&self, error: &anyhow::Error, task_id: TaskId) -> ErrorResponse {
        error!(?task_id, "Task failed - {error:?}");

        // TODO: Implement time-window failure rate calculation
        // For now, just log the error and continue
        warn!(
            ?task_id,
            max_failure_rate = self.max_failure_rate,
            window_secs = self.window_secs,
            "Rate-based error policy - time window tracking not yet implemented"
        );

        ErrorResponse::Continue
    }
}

/// Custom action that triggers a cancellation token when executed
///
/// This action demonstrates the ErrorResponse::Custom behavior by capturing
/// an external cancellation token and triggering it when executed.
#[derive(Debug)]
pub struct TriggerCancellationTokenAction {
    cancel_token: CancellationToken,
}

impl TriggerCancellationTokenAction {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self { cancel_token }
    }
}

#[async_trait]
impl OnErrorAction for TriggerCancellationTokenAction {
    async fn execute(
        &self,
        error: &anyhow::Error,
        task_id: TaskId,
        _attempt_count: u32,
        _context: &TaskExecutionContext,
    ) -> ActionResult {
        warn!(
            ?task_id,
            "Executing custom action: triggering cancellation token - {error:?}"
        );

        // Trigger the custom cancellation token
        self.cancel_token.cancel();

        // Return success - the action completed successfully
        ActionResult::Cancel
    }
}

/// Test error policy that triggers a custom cancellation token on any error
///
/// This policy demonstrates the ErrorResponse::Custom behavior by capturing
/// an external cancellation token and triggering it when any error occurs.
/// Used for testing custom error handling actions.
///
/// # Example
/// ```rust
/// # use tokio_util::sync::CancellationToken;
/// # use dynamo_runtime::utils::tasks::tracker::TriggerCancellationTokenOnError;
/// let cancel_token = CancellationToken::new();
/// let policy = TriggerCancellationTokenOnError::new(cancel_token.clone());
///
/// // Policy will trigger the token on any error via ErrorResponse::Custom
/// ```
#[derive(Debug)]
pub struct TriggerCancellationTokenOnError {
    cancel_token: CancellationToken,
}

impl TriggerCancellationTokenOnError {
    /// Create a new policy that triggers the given cancellation token on errors
    pub fn new(cancel_token: CancellationToken) -> Arc<Self> {
        Arc::new(Self { cancel_token })
    }
}

impl OnErrorPolicy for TriggerCancellationTokenOnError {
    fn create_child(&self) -> Arc<dyn OnErrorPolicy> {
        // Child gets a child cancel token
        Arc::new(TriggerCancellationTokenOnError {
            cancel_token: self.cancel_token.clone(),
        })
    }

    fn on_error(&self, error: &anyhow::Error, task_id: TaskId) -> ErrorResponse {
        error!(
            ?task_id,
            "Task failed - triggering custom cancellation token - {error:?}"
        );

        // Create the custom action that will trigger our token
        let action = TriggerCancellationTokenAction::new(self.cancel_token.clone());

        // Return Custom response with our action
        ErrorResponse::Custom(Box::new(action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    // Test fixtures using rstest
    #[fixture]
    fn semaphore_scheduler() -> Arc<SemaphoreScheduler> {
        Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))))
    }

    #[fixture]
    fn unlimited_scheduler() -> Arc<UnlimitedScheduler> {
        UnlimitedScheduler::new()
    }

    #[fixture]
    fn log_policy() -> Arc<LogOnlyPolicy> {
        LogOnlyPolicy::new()
    }

    #[fixture]
    fn cancel_policy() -> Arc<CancelOnError> {
        CancelOnError::new()
    }

    #[fixture]
    fn basic_tracker(
        unlimited_scheduler: Arc<UnlimitedScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) -> TaskTracker {
        TaskTracker::new(unlimited_scheduler, log_policy).unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_basic_task_execution(basic_tracker: TaskTracker) {
        // Test successful task execution
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = basic_tracker.spawn(async {
            // Wait for signal to complete instead of sleep
            rx.await.ok();
            Ok(42)
        });

        // Signal task to complete
        tx.send(()).ok();

        // Verify task completes successfully
        let result = handle
            .await
            .expect("Task should complete")
            .expect("Task should succeed");
        assert_eq!(result, 42);

        // Verify metrics
        assert_eq!(basic_tracker.metrics().success(), 1);
        assert_eq!(basic_tracker.metrics().failed(), 0);
        assert_eq!(basic_tracker.metrics().cancelled(), 0);
        assert_eq!(basic_tracker.metrics().active(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_task_failure(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test task failure handling
        let tracker = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        let handle = tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("test error")) });

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Failed(_)));

        // Verify metrics
        assert_eq!(tracker.metrics().success(), 0);
        assert_eq!(tracker.metrics().failed(), 1);
        assert_eq!(tracker.metrics().cancelled(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_semaphore_concurrency_limit(log_policy: Arc<LogOnlyPolicy>) {
        // Test that semaphore limits concurrent execution
        let limited_scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(2)))); // Only 2 concurrent tasks
        let tracker = TaskTracker::new(limited_scheduler, log_policy).unwrap();

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

    #[rstest]
    #[tokio::test]
    async fn test_cancel_on_error_policy() {
        // Test that CancelOnError policy works correctly
        let error_policy = cancel_policy();
        let scheduler = semaphore_scheduler();
        let tracker = TaskTracker::new(scheduler, error_policy).unwrap();

        // Spawn a task that will trigger cancellation
        let handle =
            tracker.spawn(async { Err::<(), _>(anyhow::anyhow!("OutOfMemory error occurred")) });

        // Wait for the error to occur
        let result = handle.await.unwrap();
        assert!(result.is_err());

        // Give cancellation time to propagate
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Verify the cancel token was triggered
        assert!(tracker.cancellation_token().is_cancelled());
    }

    #[rstest]
    #[tokio::test]
    async fn test_tracker_cancellation() {
        // Test manual cancellation of tracker with CancelOnError policy
        let error_policy = cancel_policy();
        let scheduler = semaphore_scheduler();
        let tracker = TaskTracker::new(scheduler, error_policy).unwrap();
        let cancel_token = tracker.cancellation_token().child_token();

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
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_tracker_independence(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker has independent lifecycle
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

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

    #[rstest]
    #[tokio::test]
    async fn test_independent_metrics(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that parent and child have independent metrics
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
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

    #[rstest]
    #[tokio::test]
    async fn test_cancel_on_error_hierarchy() {
        // Test that child error policy cancellation doesn't affect parent
        let parent_error_policy = cancel_policy();
        let scheduler = semaphore_scheduler();
        let parent = TaskTracker::new(scheduler, parent_error_policy).unwrap();
        let parent_policy_token = parent.cancellation_token().child_token();
        let child = parent.child_tracker().unwrap();

        // Initially nothing should be cancelled
        assert!(!parent_policy_token.is_cancelled());

        // Use explicit synchronization instead of sleep
        let (error_tx, error_rx) = tokio::sync::oneshot::channel();
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

        // Spawn a monitoring task to watch for the parent policy token cancellation
        let parent_token_monitor = parent_policy_token.clone();
        let monitor_handle = tokio::spawn(async move {
            tokio::select! {
                _ = parent_token_monitor.cancelled() => {
                    cancel_tx.send(true).ok();
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    cancel_tx.send(false).ok();
                }
            }
        });

        // Spawn a task in the child that will trigger cancellation
        let handle = child.spawn(async move {
            let result = Err::<(), _>(anyhow::anyhow!("OutOfMemory in child"));
            error_tx.send(()).ok(); // Signal that the error has occurred
            result
        });

        // Wait for the error to occur
        let error_result = handle.await.unwrap();
        assert!(error_result.is_err());

        // Wait for our error signal
        error_rx.await.ok();

        // Check if parent policy token was cancelled within timeout
        let was_cancelled = cancel_rx.await.unwrap_or(false);
        monitor_handle.await.ok();

        // Based on hierarchical design: child errors should NOT affect parent
        // The child gets its own policy with a child token, and child cancellation
        // should not propagate up to the parent policy token
        assert!(
            !was_cancelled,
            "Parent policy token should not be cancelled by child errors"
        );
        assert!(
            !parent_policy_token.is_cancelled(),
            "Parent policy token should remain active"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_graceful_shutdown(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test graceful shutdown with close()
        let tracker = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

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
        tracker.join().await;

        // All tasks should complete successfully
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            assert!(result < 3);
        }

        // Tracker should be closed
        assert!(tracker.is_closed());
    }

    #[rstest]
    #[tokio::test]
    async fn test_semaphore_scheduler_permit_tracking(log_policy: Arc<LogOnlyPolicy>) {
        // Test that SemaphoreScheduler properly tracks permits
        let semaphore = Arc::new(Semaphore::new(3));
        let scheduler = Arc::new(SemaphoreScheduler::new(semaphore.clone()));
        let tracker = TaskTracker::new(scheduler.clone(), log_policy).unwrap();

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

    #[rstest]
    #[tokio::test]
    async fn test_builder_pattern(log_policy: Arc<LogOnlyPolicy>) {
        // Test that TaskTracker builder works correctly
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let error_policy = log_policy;

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

    #[rstest]
    #[tokio::test]
    async fn test_all_trackers_have_cancellation_tokens(log_policy: Arc<LogOnlyPolicy>) {
        // Test that all trackers (root and children) have cancellation tokens
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let root = TaskTracker::new(scheduler, log_policy).unwrap();
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

    #[rstest]
    #[tokio::test]
    async fn test_spawn_cancellable_task(log_policy: Arc<LogOnlyPolicy>) {
        // Test cancellable task spawning with proper result handling
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Test successful completion
        let (tx, rx) = tokio::sync::oneshot::channel();
        let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let handle = tracker.spawn_cancellable(move |_cancel_token| {
            let rx = rx.clone();
            async move {
                // Wait for signal instead of sleep
                if let Some(rx) = rx.lock().await.take() {
                    rx.await.ok();
                }
                CancellableTaskResult::Ok(42)
            }
        });

        // Signal task to complete
        tx.send(()).ok();

        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);
        assert_eq!(tracker.metrics().success(), 1);

        // Test cancellation handling
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let handle = tracker.spawn_cancellable(move |cancel_token| {
            let rx = rx.clone();
            async move {
                tokio::select! {
                    _ = async {
                        if let Some(rx) = rx.lock().await.take() {
                            rx.await.ok();
                        }
                    } => CancellableTaskResult::Ok("should not complete"),
                    _ = cancel_token.cancelled() => CancellableTaskResult::Cancelled,
                }
            }
        });

        // Cancel the tracker
        tracker.cancel();

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancellable_task_metrics_tracking(log_policy: Arc<LogOnlyPolicy>) {
        // Test that properly cancelled tasks increment cancelled metrics, not failed metrics
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Baseline metrics
        assert_eq!(tracker.metrics().cancelled(), 0);
        assert_eq!(tracker.metrics().failed(), 0);
        assert_eq!(tracker.metrics().success(), 0);

        // Test 1: Task that executes and THEN gets cancelled during execution
        let (start_tx, start_rx) = tokio::sync::oneshot::channel::<()>();
        let (_continue_tx, continue_rx) = tokio::sync::oneshot::channel::<()>();

        let start_tx_shared = Arc::new(tokio::sync::Mutex::new(Some(start_tx)));
        let continue_rx_shared = Arc::new(tokio::sync::Mutex::new(Some(continue_rx)));

        let start_tx_for_task = start_tx_shared.clone();
        let continue_rx_for_task = continue_rx_shared.clone();

        let handle = tracker.spawn_cancellable(move |cancel_token| {
            let start_tx = start_tx_for_task.clone();
            let continue_rx = continue_rx_for_task.clone();
            async move {
                // Signal that we've started executing
                if let Some(tx) = start_tx.lock().await.take() {
                    tx.send(()).ok();
                }

                // Wait for either continuation signal or cancellation
                tokio::select! {
                    _ = async {
                        if let Some(rx) = continue_rx.lock().await.take() {
                            rx.await.ok();
                        }
                    } => CancellableTaskResult::Ok("completed normally"),
                    _ = cancel_token.cancelled() => {
                        println!("Task detected cancellation and is returning Cancelled");
                        CancellableTaskResult::Cancelled
                    },
                }
            }
        });

        // Wait for task to start executing
        start_rx.await.ok();

        // Now cancel while the task is running
        println!("Cancelling tracker while task is executing...");
        tracker.cancel();

        // Wait for the task to complete
        let result = handle.await.unwrap();

        // Debug output
        println!("Task result: {:?}", result);
        println!(
            "Cancelled: {}, Failed: {}, Success: {}",
            tracker.metrics().cancelled(),
            tracker.metrics().failed(),
            tracker.metrics().success()
        );

        // The task should be properly cancelled and counted correctly
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));

        // Verify proper metrics: should be counted as cancelled, not failed
        assert_eq!(
            tracker.metrics().cancelled(),
            1,
            "Properly cancelled task should increment cancelled count"
        );
        assert_eq!(
            tracker.metrics().failed(),
            0,
            "Properly cancelled task should NOT increment failed count"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancellable_vs_error_metrics_distinction(log_policy: Arc<LogOnlyPolicy>) {
        // Test that we properly distinguish between cancellation and actual errors
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Test 1: Actual error should increment failed count
        let handle1 = tracker.spawn_cancellable(|_cancel_token| async move {
            CancellableTaskResult::<i32>::Err(anyhow::anyhow!("This is a real error"))
        });

        let result1 = handle1.await.unwrap();
        assert!(result1.is_err());
        assert!(matches!(result1.unwrap_err(), TaskError::Failed(_)));
        assert_eq!(tracker.metrics().failed(), 1);
        assert_eq!(tracker.metrics().cancelled(), 0);

        // Test 2: Cancellation should increment cancelled count
        let handle2 = tracker.spawn_cancellable(|_cancel_token| async move {
            CancellableTaskResult::<i32>::Cancelled
        });

        let result2 = handle2.await.unwrap();
        assert!(result2.is_err());
        assert!(matches!(result2.unwrap_err(), TaskError::Cancelled));
        assert_eq!(tracker.metrics().failed(), 1); // Still 1 from before
        assert_eq!(tracker.metrics().cancelled(), 1); // Now 1 from cancellation
    }

    #[rstest]
    #[tokio::test]
    async fn test_spawn_cancellable_error_handling(log_policy: Arc<LogOnlyPolicy>) {
        // Test error handling in cancellable tasks
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(5))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Test error result
        let handle = tracker.spawn_cancellable(|_cancel_token| async move {
            CancellableTaskResult::<i32>::Err(anyhow::anyhow!("test error"))
        });

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TaskError::Failed(_)));
        assert_eq!(tracker.metrics().failed(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_cancellation_before_execution(log_policy: Arc<LogOnlyPolicy>) {
        // Test that spawning on a cancelled tracker panics (new behavior)
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(1))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Cancel the tracker first
        tracker.cancel();

        // Give cancellation time to propagate to the inner tracker
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Now try to spawn a task - it should panic since tracker is closed
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tracker.spawn(async { Ok(42) })
        }));

        // Should panic with our new API
        assert!(
            panic_result.is_err(),
            "spawn() should panic when tracker is closed"
        );

        // Verify the panic message contains expected text
        if let Err(panic_payload) = panic_result {
            if let Some(panic_msg) = panic_payload.downcast_ref::<String>() {
                assert!(
                    panic_msg.contains("TaskTracker must not be closed"),
                    "Panic message should indicate tracker is closed: {}",
                    panic_msg
                );
            } else if let Some(panic_msg) = panic_payload.downcast_ref::<&str>() {
                assert!(
                    panic_msg.contains("TaskTracker must not be closed"),
                    "Panic message should indicate tracker is closed: {}",
                    panic_msg
                );
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_semaphore_scheduler_with_cancellation(log_policy: Arc<LogOnlyPolicy>) {
        // Test that SemaphoreScheduler respects cancellation tokens
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(1))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

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
        assert!(matches!(result.unwrap_err(), TaskError::Cancelled));
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_tracker_cancellation_independence(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker cancellation doesn't affect parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
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

    #[rstest]
    #[tokio::test]
    async fn test_parent_cancellation_propagates_to_children(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that parent cancellation propagates to all children
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
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

    #[rstest]
    #[tokio::test]
    async fn test_issued_counter_tracking(log_policy: Arc<LogOnlyPolicy>) {
        // Test that issued counter is incremented when tasks are spawned
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(2))));
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

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

    #[rstest]
    #[tokio::test]
    async fn test_child_tracker_builder(log_policy: Arc<LogOnlyPolicy>) {
        // Test that child tracker builder allows custom policies
        let parent_scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
        let parent = TaskTracker::new(parent_scheduler, log_policy).unwrap();

        // Create child with custom error policy
        let child_error_policy = CancelOnError::new();
        let child = parent
            .child_tracker_builder()
            .error_policy(child_error_policy)
            .build()
            .unwrap();

        // Test that child works
        let handle = child.spawn(async { Ok(42) });
        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, 42);

        // Child should have its own metrics
        assert_eq!(child.metrics().success(), 1);
        assert_eq!(parent.metrics().total_completed(), 1); // Parent sees aggregated
    }

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_metrics_aggregation(log_policy: Arc<LogOnlyPolicy>) {
        // Test that child metrics aggregate up to parent
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
        let parent = TaskTracker::new(scheduler, log_policy.clone()).unwrap();

        // Create child1 with default settings
        let child1 = parent.child_tracker().unwrap();

        // Create child2 with custom error policy
        let child_error_policy = CancelOnError::new();
        let child2 = parent
            .child_tracker_builder()
            .error_policy(child_error_policy)
            .build()
            .unwrap();

        // Test both custom schedulers and policies
        let another_scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(3))));
        let another_error_policy = CancelOnError::new();
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

    #[rstest]
    #[tokio::test]
    async fn test_scheduler_queue_depth_calculation(log_policy: Arc<LogOnlyPolicy>) {
        // Test that we can calculate tasks queued in scheduler
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(2)))); // Only 2 concurrent tasks
        let tracker = TaskTracker::new(scheduler, log_policy).unwrap();

        // Initially no tasks
        assert_eq!(tracker.metrics().issued(), 0);
        assert_eq!(tracker.metrics().active(), 0);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);

        // Use a channel to control when tasks complete
        let (complete_tx, _complete_rx) = tokio::sync::broadcast::channel(1);

        // Spawn 2 tasks that will hold semaphore permits
        let handle1 = tracker.spawn({
            let mut rx = complete_tx.subscribe();
            async move {
                // Wait for completion signal
                rx.recv().await.ok();
                Ok(1)
            }
        });
        let handle2 = tracker.spawn({
            let mut rx = complete_tx.subscribe();
            async move {
                // Wait for completion signal
                rx.recv().await.ok();
                Ok(2)
            }
        });

        // Give tasks time to start and acquire permits
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Should have 2 active tasks, 0 queued
        assert_eq!(tracker.metrics().issued(), 2);
        assert_eq!(tracker.metrics().active(), 2);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 2);

        // Spawn a third task - should be queued since semaphore is full
        let handle3 = tracker.spawn(async move { Ok(3) });

        // Give time for task to be queued
        tokio::task::yield_now().await;

        // Should have 2 active, 1 queued
        assert_eq!(tracker.metrics().issued(), 3);
        assert_eq!(tracker.metrics().active(), 2);
        assert_eq!(
            tracker.metrics().queued(),
            tracker.metrics().pending() - tracker.metrics().active()
        );
        assert_eq!(tracker.metrics().pending(), 3);

        // Complete all tasks by sending the signal
        complete_tx.send(()).ok();

        let result1 = handle1.await.unwrap().unwrap();
        let result2 = handle2.await.unwrap().unwrap();
        let result3 = handle3.await.unwrap().unwrap();

        assert_eq!(result1, 1);
        assert_eq!(result2, 2);
        assert_eq!(result3, 3);

        // All tasks should be completed
        assert_eq!(tracker.metrics().success(), 3);
        assert_eq!(tracker.metrics().active(), 0);
        assert_eq!(tracker.metrics().queued(), 0);
        assert_eq!(tracker.metrics().pending(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_metrics_failure_aggregation(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that failed task metrics aggregate up to parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
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

    #[rstest]
    #[tokio::test]
    async fn test_metrics_independence_between_tracker_instances(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that different tracker instances have independent metrics
        let tracker1 = TaskTracker::new(semaphore_scheduler.clone(), log_policy.clone()).unwrap();
        let tracker2 = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

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

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_join_waits_for_all(log_policy: Arc<LogOnlyPolicy>) {
        // Test that parent.join() waits for child tasks too
        let scheduler = Arc::new(SemaphoreScheduler::new(Arc::new(Semaphore::new(10))));
        let parent = TaskTracker::new(scheduler, log_policy).unwrap();
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

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_join_waits_for_children(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that join() waits for child tasks (hierarchical behavior)
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
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

    #[rstest]
    #[tokio::test]
    async fn test_hierarchical_join_operations(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that parent.join() closes and waits for child trackers too
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();
        let child = parent.child_tracker().unwrap();
        let grandchild = child.child_tracker().unwrap();

        // Verify trackers start as open
        assert!(!parent.is_closed());
        assert!(!child.is_closed());
        assert!(!grandchild.is_closed());

        // Join parent (hierarchical by default - closes and waits for all)
        parent.join().await;

        // All should be closed (check child trackers since parent was moved)
        assert!(child.is_closed());
        assert!(grandchild.is_closed());
    }

    #[rstest]
    #[tokio::test]
    async fn test_unlimited_scheduler() {
        // Test that UnlimitedScheduler executes tasks immediately
        let scheduler = UnlimitedScheduler::new();
        let error_policy = LogOnlyPolicy::new();
        let tracker = TaskTracker::new(scheduler, error_policy).unwrap();

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

    #[rstest]
    #[tokio::test]
    async fn test_threshold_cancel_policy(semaphore_scheduler: Arc<SemaphoreScheduler>) {
        // Test that ThresholdCancelPolicy cancels after failure threshold
        let error_policy = ThresholdCancelPolicy::with_threshold(2); // Cancel after 2 failures
        let tracker = TaskTracker::new(semaphore_scheduler, error_policy.clone()).unwrap();
        let cancel_token = tracker.cancellation_token().child_token();

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
    async fn test_policy_constructors() {
        // Test that all constructors follow the new clean API patterns
        let _unlimited = UnlimitedScheduler::new();
        let _semaphore = SemaphoreScheduler::with_permits(5);
        let _log_only = LogOnlyPolicy::new();
        let _cancel_policy = CancelOnError::new();
        let _threshold_policy = ThresholdCancelPolicy::with_threshold(3);
        let _rate_policy = RateCancelPolicy::builder()
            .rate(0.5)
            .window_secs(60)
            .build();

        // All constructors return Arc directly - no more ugly ::new_arc patterns
        // This test ensures the clean API reduces boilerplate
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_creation_fails_after_join(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker creation fails from closed parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        // Initially, creating a child should work
        let _child = parent.child_tracker().unwrap();

        // Close the parent tracker
        let parent_clone = parent.clone();
        parent.join().await;
        assert!(parent_clone.is_closed());

        // Now, trying to create a child should fail
        let result = parent_clone.child_tracker();
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("closed parent tracker"));
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_builder_fails_after_join(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child tracker builder creation fails from closed parent
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

        // Initially, creating a child with builder should work
        let _child = parent.child_tracker_builder().build().unwrap();

        // Close the parent tracker
        let parent_clone = parent.clone();
        parent.join().await;
        assert!(parent_clone.is_closed());

        // Now, trying to create a child with builder should fail
        let result = parent_clone.child_tracker_builder().build();
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("closed parent tracker"));
    }

    #[rstest]
    #[tokio::test]
    async fn test_child_creation_succeeds_before_join(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
        log_policy: Arc<LogOnlyPolicy>,
    ) {
        // Test that child creation works normally before parent is joined
        let parent = TaskTracker::new(semaphore_scheduler, log_policy).unwrap();

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

    #[rstest]
    #[tokio::test]
    async fn test_custom_error_response_with_cancellation_token(
        semaphore_scheduler: Arc<SemaphoreScheduler>,
    ) {
        // Test ErrorResponse::Custom behavior with TriggerCancellationTokenOnError

        // Create a custom cancellation token
        let custom_cancel_token = CancellationToken::new();

        // Create the policy that will trigger our custom token
        let error_policy = TriggerCancellationTokenOnError::new(custom_cancel_token.clone());

        // Create tracker using builder with the custom policy
        let tracker = TaskTracker::builder()
            .scheduler(semaphore_scheduler)
            .error_policy(error_policy)
            .cancel_token(custom_cancel_token.clone())
            .build()
            .unwrap();

        let child = tracker.child_tracker().unwrap();

        // Initially, the custom token should not be cancelled
        assert!(!custom_cancel_token.is_cancelled());

        // Spawn a task that will fail
        let handle = child.spawn(async {
            Err::<(), _>(anyhow::anyhow!("Test error to trigger custom response"))
        });

        // Wait for the task to complete (it will fail)
        let result = handle.await.unwrap();
        assert!(result.is_err());

        // Await a timeout/deadline or the cancellation token to be cancelled
        // The expectation is that the task will fail, and the cancellation token will be triggered
        // Hitting the deadline is a failure
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                panic!("Task should have failed, but hit the deadline");
            }
            _ = custom_cancel_token.cancelled() => {
                // Task should have failed, and the cancellation token should be triggered
            }
        }

        // The custom cancellation token should now be triggered by our policy
        assert!(
            custom_cancel_token.is_cancelled(),
            "Custom cancellation token should be triggered by ErrorResponse::Custom"
        );

        assert!(tracker.cancellation_token().is_cancelled());
        assert!(child.cancellation_token().is_cancelled());

        // Verify the error was counted
        assert_eq!(tracker.metrics().failed(), 1);
    }

    #[test]
    fn test_action_result_variants() {
        // Test that ActionResult variants can be created and pattern matched

        // Test Continue variant
        let continue_result = ActionResult::Continue;
        match continue_result {
            ActionResult::Continue => {} // Expected
            _ => panic!("Expected Continue variant"),
        }

        // Test Cancel variant
        let cancel_result = ActionResult::Cancel;
        match cancel_result {
            ActionResult::Cancel => {} // Expected
            _ => panic!("Expected Cancel variant"),
        }

        // Test ExecuteNext variant with Restartable
        #[derive(Debug)]
        struct TestRestartable;

        #[async_trait]
        impl Restartable for TestRestartable {
            async fn restart(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("test_result".to_string()))
            }
        }

        let test_restartable = Arc::new(TestRestartable);
        let execute_next_result = ActionResult::ExecuteNext {
            restartable: test_restartable,
        };

        match execute_next_result {
            ActionResult::ExecuteNext { restartable } => {
                // Verify we have a valid Restartable
                assert!(format!("{:?}", restartable).contains("TestRestartable"));
            }
            _ => panic!("Expected ExecuteNext variant"),
        }
    }

    #[test]
    fn test_restartable_error_creation() {
        // Test RestartableError creation and conversion to anyhow::Error

        // Create a dummy restartable task for testing
        #[derive(Debug)]
        struct DummyRestartable;

        #[async_trait]
        impl Restartable for DummyRestartable {
            async fn restart(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("restarted_result".to_string()))
            }
        }

        let dummy_restartable = Arc::new(DummyRestartable);
        let source_error = anyhow::anyhow!("Original task failed");

        // Test RestartableError::new
        let restartable_error = RestartableError::new(source_error, dummy_restartable);

        // Verify the error displays correctly
        let error_string = format!("{}", restartable_error);
        assert!(error_string.contains("Task failed and can be restarted"));
        assert!(error_string.contains("Original task failed"));

        // Test conversion to anyhow::Error
        let anyhow_error = anyhow::Error::new(restartable_error);
        assert!(anyhow_error
            .to_string()
            .contains("Task failed and can be restarted"));
    }

    #[test]
    fn test_restartable_error_ext_trait() {
        // Test the RestartableErrorExt trait methods

        // Test with regular anyhow::Error (not restartable)
        let regular_error = anyhow::anyhow!("Regular error");
        assert!(!regular_error.is_restartable());
        let extracted = regular_error.extract_restartable();
        assert!(extracted.is_none());

        // Test with RestartableError
        #[derive(Debug)]
        struct TestRestartable;

        #[async_trait]
        impl Restartable for TestRestartable {
            async fn restart(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("test_result".to_string()))
            }
        }

        let test_restartable = Arc::new(TestRestartable);
        let source_error = anyhow::anyhow!("Source error");
        let restartable_error = RestartableError::new(source_error, test_restartable);

        let anyhow_error = anyhow::Error::new(restartable_error);
        assert!(anyhow_error.is_restartable());

        // Test extraction of restartable task
        let extracted = anyhow_error.extract_restartable();
        assert!(extracted.is_some());
    }

    #[test]
    fn test_restartable_error_into_anyhow_helper() {
        // Test the convenience method for creating restartable errors
        // Note: This test uses a mock TaskExecutor since we don't have real ones yet

        // For now, we'll test the type erasure concept with a simple type
        struct MockExecutor;

        let _source_error = anyhow::anyhow!("Mock task failed");

        // We can't test RestartableError::into_anyhow yet because it requires
        // a real TaskExecutor<T>. This will be tested in Phase 3.
        // For now, just verify the concept works with manual construction.

        #[derive(Debug)]
        struct MockRestartable;

        #[async_trait]
        impl Restartable for MockRestartable {
            async fn restart(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("mock_result".to_string()))
            }
        }

        let mock_restartable = Arc::new(MockRestartable);
        let restartable_error =
            RestartableError::new(anyhow::anyhow!("Mock task failed"), mock_restartable);

        let anyhow_error = anyhow::Error::new(restartable_error);
        assert!(anyhow_error.is_restartable());
    }

    #[test]
    fn test_cancellable_task_executor_supports_retry() {
        // Test that CancellableTaskExecutor supports retry

        let task_fn = |_token: CancellationToken| async move { CancellableTaskResult::Ok(42u32) };

        let executor = CancellableTaskExecutor::new(task_fn);

        // Test that it implements TaskExecutor and supports retry
        assert!(executor.supports_retry());
    }

    #[test]
    fn test_restartable_error_with_task_executor() {
        // Test RestartableError creation with TaskExecutor

        #[derive(Debug)]
        struct TestRestartableTask;

        #[async_trait]
        impl Restartable for TestRestartableTask {
            async fn restart(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("test_result".to_string()))
            }
        }

        let restartable_task = Arc::new(TestRestartableTask);
        let source_error = anyhow::anyhow!("Task failed");

        // Test RestartableError::new with Restartable
        let restartable_error = RestartableError::new(source_error, restartable_task);

        // Verify the error displays correctly
        let error_string = format!("{}", restartable_error);
        assert!(error_string.contains("Task failed and can be restarted"));
        assert!(error_string.contains("Task failed"));

        // Test conversion to anyhow::Error
        let anyhow_error = anyhow::Error::new(restartable_error);
        assert!(anyhow_error.is_restartable());

        // Test extraction (should work now with Restartable trait)
        let extracted = anyhow_error.extract_restartable();
        assert!(extracted.is_some()); // Should successfully extract the Restartable
    }

    #[test]
    fn test_restartable_error_into_anyhow_convenience() {
        // Test the convenience method for creating restartable errors

        #[derive(Debug)]
        struct ConvenienceRestartable;

        #[async_trait]
        impl Restartable for ConvenienceRestartable {
            async fn restart(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new(42u32))
            }
        }

        let restartable_task = Arc::new(ConvenienceRestartable);
        let source_error = anyhow::anyhow!("Computation failed");

        // Test RestartableError::into_anyhow convenience method
        let anyhow_error = RestartableError::into_anyhow(source_error, restartable_task);

        assert!(anyhow_error.is_restartable());
        assert!(anyhow_error
            .to_string()
            .contains("Task failed and can be restarted"));
        assert!(anyhow_error.to_string().contains("Computation failed"));
    }

    #[test]
    fn test_handle_task_error_with_restartable_error() {
        // Test that handle_task_error properly detects RestartableError

        // Create a mock Restartable task
        #[derive(Debug)]
        struct MockRestartableTask;

        #[async_trait]
        impl Restartable for MockRestartableTask {
            async fn restart(
                &self,
                _cancel_token: CancellationToken,
            ) -> TaskExecutionResult<Box<dyn std::any::Any + Send + 'static>> {
                TaskExecutionResult::Success(Box::new("retry_result".to_string()))
            }
        }

        let restartable_task = Arc::new(MockRestartableTask);

        // Create RestartableError
        let source_error = anyhow::anyhow!("Task failed, but can retry");
        let restartable_error = RestartableError::new(source_error, restartable_task);
        let anyhow_error = anyhow::Error::new(restartable_error);

        // Verify it's detected as restartable
        assert!(anyhow_error.is_restartable());

        // Verify we can downcast to RestartableError
        let restartable_ref = anyhow_error.downcast_ref::<RestartableError>();
        assert!(restartable_ref.is_some());

        // Verify the restartable task is present
        let restartable = restartable_ref.unwrap();
        // Note: We can verify the Arc is valid by checking that Arc::strong_count > 0
        assert!(Arc::strong_count(&restartable.restartable) > 0);
    }

    #[test]
    fn test_handle_task_error_with_regular_error() {
        // Test that handle_task_error properly handles regular errors

        let regular_error = anyhow::anyhow!("Regular task failure");

        // Verify it's not detected as restartable
        assert!(!regular_error.is_restartable());

        // Verify we cannot downcast to RestartableError
        let restartable_ref = regular_error.downcast_ref::<RestartableError>();
        assert!(restartable_ref.is_none());
    }
}
