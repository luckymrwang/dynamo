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

        // get the group
        let group = registry
            .services
            .get(&endpoint.to_descriptor().identifier().to_component().unwrap())
            .map(|service| service.group(endpoint.to_descriptor().slug()))
            .ok_or(error!("Service not found"))?;

        // get the stats handler map
        let handler_map = registry
            .stats_handlers
            .get(&endpoint.to_descriptor().identifier().to_component().unwrap())
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

        // 1. Register in storage first
        let info = serde_json::to_vec_pretty(&info)?;

        if let Err(e) = storage.create(info.clone(), None).await {
            tracing::error!("Failed to register discoverable service: {:?}", e);
            cancel_token.cancel();
            return Err(error!("Failed to register discoverable service"));
        }

        // 2. Start the service only after successful registration
        let task = tokio::spawn(push_endpoint.start(service_endpoint));

        // 3. Monitor the task and handle result
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
