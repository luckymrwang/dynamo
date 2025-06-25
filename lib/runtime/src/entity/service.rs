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
use std::collections::HashMap;
use std::sync::Mutex;

use super::*;
use crate::transports::nats::Slug;

pub use super::endpoint::EndpointStats;
pub type StatsHandler =
    Box<dyn FnMut(String, EndpointStats) -> serde_json::Value + Send + Sync + 'static>;
pub type EndpointStatsHandler =
    Box<dyn FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static>;

pub const PROJECT_NAME: &str = "Dynamo";

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct ServiceConfig {
    #[builder(private)]
    component: Component,

    /// Description
    #[builder(default)]
    description: Option<String>,
}

impl ServiceConfigBuilder {
    /// Create the [`Component`]'s service and store it in the registry.
    pub async fn create(self) -> Result<Component> {
        let (component, description) = self.build_internal()?.dissolve();

        let version = "0.0.1".to_string();

        // let service_name = component.service_name();
        // log::debug!("component: {component}; creating, service_name: {service_name}");

        let description = description.unwrap_or(format!(
            "{PROJECT_NAME} {component}"));

        let stats_handler_registry: Arc<Mutex<HashMap<Slug, EndpointStatsHandler>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let stats_handler_registry_clone = stats_handler_registry.clone();

        let mut guard = component.drt().component_registry.inner.lock().await;

        if guard.services.contains_key(&component.to_descriptor()) {
            return Err(anyhow::anyhow!("Service already exists"));
        }

        // create service on the secondary runtime
        let builder = component.drt().nats_client.client().service_builder();

        tracing::debug!("Starting service: {}", component.to_descriptor().slug());
        let service_builder = builder
            .description(description)
            .stats_handler(move |name: String, stats| {
                log::trace!("stats_handler: {name}, {stats:?}");
                let mut guard = stats_handler_registry.lock().unwrap();
                match guard.get_mut(&Slug::slugify(&name)) {
                    Some(handler) => handler(stats),
                    None => serde_json::Value::Null,
                }
            });
        tracing::debug!("Got builder");
        let service = service_builder
            .start(component.to_descriptor().slug().to_string(), version)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start service: {e}"))?;

        // insert the service into the registry
        guard.services.insert(component.to_descriptor(), service);

        // insert the stats handler into the registry
        guard
            .stats_handlers
            .insert(component.to_descriptor(), stats_handler_registry_clone);

        // drop the guard to unlock the mutex
        drop(guard);

        Ok(component)
    }
}

impl ServiceConfigBuilder {
    pub(crate) fn from_component(component: Component) -> Self {
        Self::default().component(component)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Component;
    use crate::Runtime;

    async fn create_test_runtime() -> DistributedRuntime {
        let runtime = Runtime::from_current().unwrap();
        DistributedRuntime::from_settings_without_discovery(runtime).await.unwrap()
    }

    async fn check_nats_available() -> bool {
        // Try to connect to NATS to see if it's available
        match async_nats::connect("nats://localhost:4222").await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    #[tokio::test]
    async fn test_service_creation_registers_service() {
        // Check if NATS is available
        if !check_nats_available().await {
            eprintln!("Skipping test: NATS not available");
            return;
        }

        let drt = create_test_runtime().await;
        let component = Component::new("test", "svc", drt).unwrap();
        let builder = ServiceConfigBuilder::from_component(component.clone());
        let created_component = builder.create().await.expect("Service should be created");

        // Check that the service is in the registry
        let registry = created_component.drt().component_registry.inner.lock().await;
        assert!(registry.services.contains_key(&created_component.to_descriptor()));

        // Also check that stats handler registry was created
        assert!(registry.stats_handlers.contains_key(&created_component.to_descriptor()));
    }

    #[tokio::test]
    async fn test_duplicate_service_creation_fails() {
        // Check if NATS is available
        if !check_nats_available().await {
            eprintln!("Skipping test: NATS not available");
            return;
        }

        let drt = create_test_runtime().await;
        let component = Component::new("test", "svc", drt).unwrap();
        let builder = ServiceConfigBuilder::from_component(component.clone());
        builder.create().await.expect("First creation should succeed");

        // Second creation should fail
        let builder2 = ServiceConfigBuilder::from_component(component.clone());
        let result = builder2.create().await;
        assert!(result.is_err(), "Duplicate service creation should fail");

        // Verify the error message contains expected text
        if let Err(e) = result {
            assert!(e.to_string().contains("Service already exists"));
        }
    }
}
