// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;

use crate::{slug::Slug, transports::etcd::Client};
use async_stream::stream;
use async_trait::async_trait;
use etcd_client::{Compare, CompareOp, EventType, PutOptions, Txn, TxnOp, WatchOptions};

use super::{KeyValueBucket, KeyValueStore, StorageError, StorageOutcome};

#[derive(Clone)]
pub struct EtcdStorage {
    client: Client,
}

impl EtcdStorage {
    pub fn new(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl KeyValueStore for EtcdStorage {
    /// A "bucket" in etcd is a path prefix
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        _ttl: Option<Duration>, // TODO ttl not used yet
    ) -> Result<Box<dyn KeyValueBucket>, StorageError> {
        Ok(self.get_bucket(bucket_name).await?.unwrap())
    }

    /// A "bucket" in etcd is a path prefix. This creates an EtcdBucket object without doing
    /// any network calls.
    async fn get_bucket(
        &self,
        bucket_name: &str,
    ) -> Result<Option<Box<dyn KeyValueBucket>>, StorageError> {
        Ok(Some(Box::new(EtcdBucket {
            client: self.client.clone(),
            bucket_name: bucket_name.to_string(),
        })))
    }
}

pub struct EtcdBucket {
    client: Client,
    bucket_name: String,
}

#[async_trait]
impl KeyValueBucket for EtcdBucket {
    async fn insert(
        &self,
        key: String,
        value: String,
        // "version" in etcd speak. revision is a global cluster-wide value
        revision: u64,
    ) -> Result<StorageOutcome, StorageError> {
        let version = revision;
        if version == 0 {
            self.create(&key, &value).await
        } else {
            self.update(&key, &value, version).await
        }
    }

    async fn get(&self, key: &str) -> Result<Option<bytes::Bytes>, StorageError> {
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd get: {k}");

        let mut kvs = self
            .client
            .kv_get(k, None)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        if kvs.is_empty() {
            return Ok(None);
        }
        let (_, val) = kvs.swap_remove(0).into_key_value();
        Ok(Some(val.into()))
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let _ = self
            .client
            .kv_delete(key, None)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        Ok(())
    }

    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = bytes::Bytes> + Send + 'life0>>, StorageError>
    {
        let k = make_key(&self.bucket_name, "");
        tracing::trace!("etcd watch: {k}");
        let (_watcher, mut watch_stream) = self
            .client
            .etcd_client()
            .clone()
            .watch(k.as_bytes(), Some(WatchOptions::new().with_prefix()))
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        let output = stream! {
            while let Ok(Some(resp)) = watch_stream.message().await {
                for e in resp.events() {
                    if matches!(e.event_type(), EventType::Put) && e.kv().is_some() {
                        let b: bytes::Bytes = e.kv().unwrap().value().to_vec().into();
                        yield b;
                    }
                }
            }
        };
        Ok(Box::pin(output))
    }

    async fn entries(&self) -> Result<HashMap<String, bytes::Bytes>, StorageError> {
        let k = make_key(&self.bucket_name, "");
        tracing::trace!("etcd entries: {k}");

        let resp = self
            .client
            .kv_get_prefix(k)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        let out: HashMap<String, bytes::Bytes> = resp
            .into_iter()
            .map(|kv| {
                let (k, v) = kv.into_key_value();
                (String::from_utf8_lossy(&k).to_string(), v.into())
            })
            .collect();

        Ok(out)
    }
}

impl EtcdBucket {
    async fn create(&self, key: &str, value: &str) -> Result<StorageOutcome, StorageError> {
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd create: {k}");

        // Use atomic transaction to check and create in one operation
        let put_options = PutOptions::new();

        // Build transaction that creates key only if it doesn't exist
        let txn = Txn::new()
            .when(vec![Compare::version(k.as_str(), CompareOp::Equal, 0)]) // Atomic check
            .and_then(vec![TxnOp::put(k.as_str(), value, Some(put_options))]) // Only if check passes
            .or_else(vec![
                TxnOp::get(k.as_str(), None), // Key exists, get its info
            ]);

        // Execute the transaction
        let result = self
            .client
            .etcd_client()
            .kv_client()
            .txn(txn)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;

        if result.succeeded() {
            // Key was created successfully
            return Ok(StorageOutcome::Created(1)); // version of new key is always 1
        }

        // Key already existed, get its version
        if let Some(etcd_client::TxnOpResponse::Get(get_resp)) =
            result.op_responses().into_iter().next()
        {
            if let Some(kv) = get_resp.kvs().first() {
                let version = kv.version() as u64;
                return Ok(StorageOutcome::Exists(version));
            }
        }
        // Shouldn't happen, but handle edge case
        Err(StorageError::EtcdError(
            "Unexpected transaction response".to_string(),
        ))
    }

    async fn update(
        &self,
        key: &str,
        value: &str,
        revision: u64,
    ) -> Result<StorageOutcome, StorageError> {
        let version = revision;
        let k = make_key(&self.bucket_name, key);
        tracing::trace!("etcd update: {k}");

        let kvs = self
            .client
            .kv_get(k.clone(), None)
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        if kvs.is_empty() {
            return Err(StorageError::MissingKey(key.to_string()));
        }
        let current_version = kvs.first().unwrap().version() as u64;
        if current_version != version + 1 {
            tracing::warn!(
                current_version,
                attempted_next_version = version,
                key,
                "update: Wrong revision"
            );
            // NATS does a resync_update, overwriting the key anyway and getting the new revision.
            // So we do too in etcd.
        }

        let mut put_resp = self
            .client
            .kv_put_with_options(k, value, Some(PutOptions::new().with_prev_key()))
            .await
            .map_err(|e| StorageError::EtcdError(e.to_string()))?;
        Ok(match put_resp.take_prev_key() {
            // Should this be an error?
            // The key was deleted between our get and put. We re-created it.
            // Version of new key is always 1.
            // <https://etcd.io/docs/v3.5/learning/data_model/>
            None => StorageOutcome::Created(1),
            // Expected case, success
            Some(kv) if kv.version() as u64 == version + 1 => StorageOutcome::Created(version),
            // Should this be an error? Something updated the version between our get and put
            Some(kv) => StorageOutcome::Created(kv.version() as u64 + 1),
        })
    }
}

fn make_key(bucket_name: &str, key: &str) -> String {
    [
        Slug::slugify(bucket_name).to_string(),
        Slug::slugify(key).to_string(),
    ]
    .join("/")
}

#[cfg(feature = "integration")]
#[cfg(test)]
mod concurrent_create_tests {
    use super::*;
    use crate::{distributed::DistributedConfig, DistributedRuntime, Runtime};
    use std::sync::Arc;
    use tokio::sync::Barrier;

    #[test]
    fn test_concurrent_etcd_create_race_condition() {
        let rt = Runtime::from_settings().unwrap();
        let rt_clone = rt.clone();
        let config = DistributedConfig::from_settings(false);

        rt_clone.primary().block_on(async move {
            let drt = DistributedRuntime::new(rt, config).await.unwrap();
            test_concurrent_create(drt).await.unwrap();
        });
    }

    async fn test_concurrent_create(drt: DistributedRuntime) -> Result<(), StorageError> {
        let etcd_client = drt.etcd_client().expect("etcd client should be available");
        let storage = EtcdStorage::new(etcd_client);

        // Create a bucket for testing
        let bucket = Arc::new(tokio::sync::Mutex::new(
            storage
                .get_or_create_bucket("test_concurrent_bucket", None)
                .await?,
        ));

        // Number of concurrent workers
        let num_workers = 10;
        let barrier = Arc::new(Barrier::new(num_workers));

        // Shared test data
        let test_key = format!("concurrent_test_key_{}", uuid::Uuid::new_v4());
        let test_value = "test_value";

        // Spawn multiple tasks that will all try to create the same key simultaneously
        let mut handles = Vec::new();
        let success_count = Arc::new(tokio::sync::Mutex::new(0));
        let exists_count = Arc::new(tokio::sync::Mutex::new(0));

        for worker_id in 0..num_workers {
            let bucket_clone = bucket.clone();
            let barrier_clone = barrier.clone();
            let key_clone = test_key.clone();
            let value_clone = format!("{}_from_worker_{}", test_value, worker_id);
            let success_count_clone = success_count.clone();
            let exists_count_clone = exists_count.clone();

            let handle = tokio::spawn(async move {
                // Wait for all workers to be ready
                barrier_clone.wait().await;

                // All workers try to create the same key at the same time
                let result = bucket_clone
                    .lock()
                    .await
                    .insert(key_clone, value_clone, 0)
                    .await;

                match result {
                    Ok(StorageOutcome::Created(version)) => {
                        println!(
                            "Worker {} successfully created key with version {}",
                            worker_id, version
                        );
                        let mut count = success_count_clone.lock().await;
                        *count += 1;
                        Ok(version)
                    }
                    Ok(StorageOutcome::Exists(version)) => {
                        println!(
                            "Worker {} found key already exists with version {}",
                            worker_id, version
                        );
                        let mut count = exists_count_clone.lock().await;
                        *count += 1;
                        Ok(version)
                    }
                    Err(e) => {
                        println!("Worker {} got error: {:?}", worker_id, e);
                        Err(e)
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            if let Ok(version) = result {
                results.push(version);
            }
        }

        // Verify results
        let final_success_count = *success_count.lock().await;
        let final_exists_count = *exists_count.lock().await;

        println!(
            "Final counts - Created: {}, Exists: {}",
            final_success_count, final_exists_count
        );

        // CRITICAL ASSERTIONS:
        // 1. Exactly ONE worker should have successfully created the key
        assert_eq!(
            final_success_count, 1,
            "Exactly one worker should create the key"
        );

        // 2. All other workers should have gotten "Exists" response
        assert_eq!(
            final_exists_count,
            num_workers - 1,
            "All other workers should see key exists"
        );

        // 3. Total successful operations should equal number of workers
        assert_eq!(
            results.len(),
            num_workers,
            "All workers should complete successfully"
        );

        // 4. Verify the key actually exists in etcd
        let stored_value = bucket.lock().await.get(&test_key).await?;
        assert!(stored_value.is_some(), "Key should exist in etcd");

        // 5. The stored value should be from one of the workers
        let stored_str = String::from_utf8(stored_value.unwrap().to_vec()).unwrap();
        assert!(
            stored_str.starts_with(test_value),
            "Stored value should match expected prefix"
        );

        // Clean up
        bucket.lock().await.delete(&test_key).await?;

        Ok(())
    }
}

// run with: cd /home/ubuntu/dynamo && cargo test --locked test_etcd_system_stats_server -- --nocapture
#[cfg(test)]
mod etcd_metrics_test {
    use super::*;
    use crate::component::INSTANCE_ROOT_PATH;
    use crate::pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    };
    use crate::protocols::annotated::Annotated;
    use crate::stream;
    use crate::{distributed::DistributedConfig, DistributedRuntime, Runtime};
    use reqwest;
    use std::env;
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};
    use uuid::Uuid;

    const TEST_NAMESPACE: &str = "testnamespace";
    const TEST_COMPONENT: &str = "testcomponent";
    const TEST_ENDPOINT: &str = "testendpoint";

    struct TestRequestHandler {}

    impl TestRequestHandler {
        fn new() -> Arc<Self> {
            Arc::new(Self {})
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for TestRequestHandler {
        async fn generate(
            &self,
            input: SingleIn<String>,
        ) -> crate::Result<ManyOut<Annotated<String>>> {
            let (data, ctx) = input.into_parts();

            let chars = data
                .chars()
                .map(|c| Annotated::from_data(c.to_string()))
                .collect::<Vec<_>>();

            let stream = stream::iter(chars);

            Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
        }
    }

    async fn backend(drt: DistributedRuntime) -> crate::Result<()> {
        // Create a namespace, component, service, and endpoint like the backend examples
        let ingress = Ingress::for_engine(TestRequestHandler::new())?;

        let endpoint = drt
            .namespace(TEST_NAMESPACE)?
            .component(TEST_COMPONENT)?
            .service_builder()
            .create()
            .await?
            .endpoint(TEST_ENDPOINT);

        // Start the endpoint with the ingress handler
        endpoint.endpoint_builder().handler(ingress).start().await?;

        Ok(())
    }

    #[test]
    fn test_etcd_system_stats_server() {
        // Test direct etcd client operations. Generate bogus instances on etcd.
        let rt = Runtime::from_settings().unwrap();
        let rt_clone = rt.clone();

        rt_clone.primary().block_on(async move {
            // Create first drt for etcd operations
            let drt1 =
                DistributedRuntime::new(rt_clone.clone(), DistributedConfig::from_settings(false))
                    .await
                    .unwrap();
            let etcd_client = drt1.etcd_client().expect("etcd client should be available");

            // Test kv_create - create 10 deterministic keys
            let mut created_keys = Vec::new();
            for i in 0..10 {
                let key = format!(
                    "{}/dynamo/deleteme_test/generate:{:02}",
                    INSTANCE_ROOT_PATH, i
                );
                let value = format!("deterministic_value_{:02}", i);
                let value_bytes = value.as_bytes().to_vec();

                println!("Testing kv_create with key: '{}', value: '{}'", key, value);
                etcd_client
                    .kv_create(key.clone(), value_bytes, None)
                    .await
                    .map_err(|e| StorageError::EtcdError(e.to_string()))
                    .unwrap();
                println!("Successfully created key via kv_create: {}", key);
                created_keys.push(key);
            }

            // Set environment variables for dynamic port allocation BEFORE creating the runtime
            std::env::set_var("DYN_SYSTEM_ENABLED", "true");
            std::env::set_var("DYN_SYSTEM_PORT", "0");

            // Create second drt for system stats server test
            let drt2 = DistributedRuntime::new(rt_clone, DistributedConfig::from_settings(false))
                .await
                .unwrap();
            test_system_stats_server(drt2).await.unwrap();

            // Clean up created keys
            for key in created_keys {
                etcd_client
                    .kv_delete(key.as_bytes(), None)
                    .await
                    .map_err(|e| StorageError::EtcdError(e.to_string()))
                    .unwrap();
                println!("Deleted key: {}", key);
            }
        });
    }

    async fn test_system_stats_server(drt: DistributedRuntime) -> Result<(), StorageError> {
        // Get the system stats server info to find the actual port
        let metrics_server_info = drt.metrics_server_info();
        let metrics_port = match metrics_server_info {
            Some(info) => {
                println!("System stats server running on: {}", info.address());
                info.port()
            }
            None => {
                panic!("System stats server not started - check DYN_SYSTEM_ENABLED environment variable");
            }
        };

        // Start the backend in a separate task (like system_metrics example)
        let drt_clone = drt.clone();
        let backend_handle = tokio::spawn(async move { backend(drt_clone).await });

        // Give the backend some time to start up
        sleep(Duration::from_millis(500)).await;

        // Now fetch the system stats server endpoint using the dynamic port
        let metrics_url = format!("http://localhost:{}/metrics", metrics_port);
        println!("Fetching metrics from: {}", metrics_url);

        // Make HTTP request to get metrics using reqwest
        let client = reqwest::Client::new();
        let response = client.get(&metrics_url).send().await;

        match response {
            Ok(response) => {
                if response.status().is_success() {
                    let metrics_content = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Failed to read response body".to_string());

                    // println!("=== METRICS CONTENT ===");
                    // println!("{}", metrics_content);
                    // println!("=== END METRICS CONTENT ===");

                    // Verify that metrics endpoint is working
                    assert!(
                        !metrics_content.is_empty(),
                        "Metrics content should not be empty"
                    );

                    // Filter to only show lines containing '_etcd_' in the metric name (not labels)
                    let etcd_metrics: Vec<&str> = metrics_content
                        .lines()
                        .filter(|line| {
                            // Only include lines that contain '_etcd_' in the metric name
                            // (not in labels like dynamo_namespace="test_etcd_namespace")
                            line.contains("_etcd_")
                                && (line.starts_with("# HELP")
                                    || line.starts_with("# TYPE")
                                    || line.starts_with("dynamo_component_etcd_"))
                        })
                        .collect();

                    println!("=== ETCD METRICS ONLY ===");
                    for line in &etcd_metrics {
                        println!("{}", line);
                    }
                    println!("=== END ETCD METRICS ===");

                    // Check for some common metrics that should be present
                    let has_metrics = metrics_content.contains("dynamo_")
                        || metrics_content.contains("process_")
                        || metrics_content.contains("go_");

                    assert!(has_metrics, "Metrics content should contain some metrics");

                    println!("Successfully retrieved metrics from system stats server endpoint!");
                } else {
                    println!("HTTP request failed with status: {}", response.status());
                    panic!("Failed to get metrics: HTTP {}", response.status());
                }
            }
            Err(e) => {
                println!("Failed to connect to system stats server endpoint: {}", e);
                panic!("Failed to connect to system stats server endpoint: {}", e);
            }
        }

        // Cancel the backend task
        backend_handle.abort();

        Ok(())
    }
}
