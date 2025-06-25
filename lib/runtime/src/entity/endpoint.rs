// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use derive_getters::Dissolve;

use crate::entity::client::TransportType;

use super::*;

pub use async_nats::service::endpoint::Stats as EndpointStats;
use crate::entity::client::StoredTransport;

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct EndpointConfig {
    #[builder(private)]
    endpoint: Endpoint,

    /// Endpoint handler
    #[educe(Debug(ignore))]
    handler: Arc<dyn PushWorkHandler>,

    /// Stats handler
    #[educe(Debug(ignore))]
    #[builder(default, private)]
    _stats_handler: Option<EndpointStatsHandler>,
}

impl EndpointConfigBuilder {
    pub(crate) fn from_endpoint(endpoint: Endpoint) -> Self {
        Self::default().endpoint(endpoint)
    }

    pub fn stats_handler<F>(self, handler: F) -> Self
    where
        F: FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static,
    {
        self._stats_handler(Some(Box::new(handler)))
    }

    pub async fn start(self) -> Result<()> {
        let (endpoint, handler, stats_handler) = self.build_internal()?.dissolve();

        // acquire the registry lock
        let registry = endpoint.drt().component_registry.inner.lock().await;

        let identifier = &endpoint.to_descriptor().identifier().to_component().unwrap();

        // get the group
        let group = registry
            .services
            .get(identifier)
            .map(|service| service.group(endpoint.to_descriptor().slug()))
            .ok_or(error!("Service not found"))?;

        // get the stats handler map
        let handler_map = registry
            .stats_handlers
            .get(identifier)
            .cloned()
            .expect("no stats handler registry; this is unexpected");

        drop(registry);

        // insert the stats handler
        if let Some(stats_handler) = stats_handler {
            handler_map
                .lock()
                .unwrap()
                .insert(endpoint.to_descriptor().slug(), stats_handler);
        }

        // creates an endpoint for the service
        let service_endpoint = group
            .endpoint(&endpoint.to_descriptor().slug())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

        let storage = endpoint.storage()?;
        let cancel_token = storage.primary_lease().child_token();

        let push_endpoint = PushEndpoint::builder()
            .service_handler(handler)
            .cancellation_token(cancel_token.clone())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

        let info = StoredTransport{
            key: endpoint.to_descriptor(),
            value: TransportType::NatsTcp(endpoint.to_descriptor().slug().to_string())
        };

        // Start the service in background
        let task = tokio::spawn(push_endpoint.start(service_endpoint));

        // Register in storage after service is starting
        let info = serde_json::to_vec_pretty(&info)?;
        if let Err(e) = storage.create(info.clone(), None).await {
            tracing::error!("Failed to register discoverable service: {:?}", e);
            cancel_token.cancel();
            return Err(error!("Failed to register discoverable service"));
        }

        // Monitor the task and handle result
        let task_result = match task.await {
            Ok(Ok(())) => {
                tracing::debug!("Endpoint service completed successfully");
                Ok(())
            }
            Ok(Err(service_error)) => {
                tracing::error!(
                    error = %service_error,
                    endpoint = %endpoint,
                    "Service failed"
                );
                cancel_token.cancel();
                Err(service_error)
            }
            Err(join_error) => {
                tracing::error!(
                    error = %join_error,
                    endpoint = %endpoint,
                    "Task join failed"
                );
                cancel_token.cancel();
                Err(error!("Task failed to complete"))
            }
        };

        // Always cleanup: remove from storage regardless of success/failure
        if let Err(cleanup_error) = storage.delete(None).await {
            tracing::warn!(
                error = %cleanup_error,
                endpoint = %endpoint,
                action = "cleanup_service_registration",
                "Failed to cleanup service registration"
            );
        }

        task_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::{Component, Endpoint};
    use crate::{Runtime, DistributedRuntime};
    use crate::pipeline::network::PushWorkHandler;
    use crate::pipeline::error::PipelineError;
    use std::sync::Arc;
    use async_trait::async_trait;
    use bytes::Bytes;

    // Mock handler for testing
    #[derive(Clone)]
    struct MockHandler;

    #[async_trait]
    impl PushWorkHandler for MockHandler {
        async fn handle_payload(&self, _payload: Bytes) -> Result<(), PipelineError> {
            Ok(())
        }
    }

    async fn create_test_runtime() -> DistributedRuntime {
        let runtime = Runtime::from_current().unwrap();
        DistributedRuntime::from_settings_without_discovery(runtime).await.unwrap()
    }

    async fn create_test_runtime_with_etcd() -> Result<DistributedRuntime> {
        let runtime = Runtime::from_current()?;
        let mut config = crate::distributed::DistributedConfig::from_settings(false);
        config.etcd_config.etcd_url = vec!["http://localhost:2379".to_string()];
        DistributedRuntime::new(runtime, config).await
    }

    async fn check_nats_available() -> bool {
        // Try to connect to NATS to see if it's available
        match async_nats::connect("nats://localhost:4222").await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    async fn check_etcd_available() -> bool {
        // Try to connect to etcd to see if it's available
        etcd_client::Client::connect(["localhost:2379"], None).await.is_ok()
    }

    #[tokio::test]
    async fn test_service_and_endpoint_integration() {
        // Check if NATS is available
        if !check_nats_available().await {
            eprintln!("Skipping test: NATS not available");
            return;
        }

        // Check if etcd is available
        if !check_etcd_available().await {
            eprintln!("Skipping test: ETCD not available");
            return;
        }

        let drt = match create_test_runtime_with_etcd().await {
            Ok(drt) => drt,
            Err(_) => {
                eprintln!("Skipping test: Could not create runtime with etcd");
                return;
            }
        };

        // Step 1: Create a service
        let component = Component::new("test", "integration", drt.clone()).unwrap();
        let service_builder = crate::entity::service::ServiceConfigBuilder::from_component(component.clone());
        let _created_component = service_builder.create().await.expect("Service creation should succeed");

        // Step 2: Create an endpoint for the service
        let endpoint = Endpoint::new("test", "integration", "testapi", drt.clone()).unwrap();
        let handler: Arc<dyn PushWorkHandler> = Arc::new(MockHandler);

        // Get the storage and its cancellation token before moving endpoint
        let storage = endpoint.storage().unwrap();
        let cancel_token = storage.primary_lease().primary_token();

        let endpoint_builder = EndpointConfigBuilder::from_endpoint(endpoint.clone())
            .handler(handler);

        // Start the endpoint in a background task
        let endpoint_task = tokio::spawn(async move {
            endpoint_builder.start().await
        });

        // Give it a moment to register
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Step 3: Verify the endpoint is registered in storage
        let kvs = storage.get().await.expect("Should get storage data");
        assert!(!kvs.is_empty(), "Endpoint should be registered in storage");

        // Parse and verify the stored transport
        if let Some(kv) = kvs.first() {
            let stored: StoredTransport = serde_json::from_slice(kv.value()).expect("Should deserialize stored transport");
            assert_eq!(stored.key, endpoint.to_descriptor());
            assert!(matches!(stored.value, TransportType::NatsTcp(_)));
        }

        // Trigger graceful shutdown via cancellation token
        cancel_token.cancel();

        // Wait for the endpoint task to complete
        match tokio::time::timeout(tokio::time::Duration::from_secs(5), endpoint_task).await {
            Ok(Ok(Ok(()))) => println!("Endpoint shut down successfully"),
            Ok(Ok(Err(e))) => println!("Endpoint shut down with error: {:?}", e),
            Ok(Err(e)) => println!("Endpoint task panicked: {:?}", e),
            Err(_) => println!("Endpoint shutdown timed out"),
        }

        // Give cleanup a moment to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Verify cleanup happened
        let kvs_after = storage.get().await.expect("Should get storage data");
        assert!(kvs_after.is_empty(), "Endpoint should be removed from storage after shutdown");
    }
}
