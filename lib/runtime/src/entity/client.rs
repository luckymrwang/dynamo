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

use crate::pipeline::{
    AddressedPushRouter, AddressedRequest, AsyncEngine, Data, ManyOut, PushRouter, RouterMode,
    SingleIn,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::{net::unix::pipe::Receiver, sync::Mutex};

use crate::{
    pipeline::async_trait,
    transports::etcd::{Client as EtcdClient, WatchEvent},
};

use super::*;

/// Each state will be have a nonce associated with it
/// The state will be emitted in a watch channel, so we can observe the
/// critical state transitions.
enum MapState {
    /// The map is empty; value = nonce
    Empty(u64),

    /// The map is not-empty; values are (nonce, count)
    NonEmpty(u64, u64),

    /// The watcher has finished, no more events will be emitted
    Finished,
}

enum EndpointEvent {
    Put(String, i64),
    Delete(String),
}

#[derive(Clone, Debug)]
pub struct Client {
    // This is me
    pub endpoint: Endpoint,
    // These are the remotes I know about
    pub instance_source: Arc<InstanceSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    NatsTcp(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct StoredValue<T> {
    pub key: Instance,
    pub value: T,
}

pub type StoredTransport = StoredValue<TransportType>;

#[derive(Clone, Debug)]
pub enum InstanceSource {
    Static,
    Dynamic(tokio::sync::watch::Receiver<Vec<StoredTransport>>),
}

impl Client {

    pub(crate) async fn new(endpoint: Endpoint) -> Result<Self> {
        if endpoint.to_descriptor().is_static() {
            Ok(Client {
                endpoint,
                instance_source: Arc::new(InstanceSource::Static),
            })
        } else {
            let instance_source =
            Self::get_or_create_dynamic_instance_source(&endpoint).await?;
            Ok(Client {
                endpoint,
                instance_source,
            })
        }

    }

    pub fn instances(&self) -> Vec<StoredTransport> {
        match self.instance_source.as_ref() {
            InstanceSource::Static => vec![],
            InstanceSource::Dynamic(watch_rx) => watch_rx.borrow().clone(),
        }
    }

    pub fn instance_ids(&self) -> Vec<i64> {
        self.instances().into_iter().filter_map(|ep| ep.key.instance_id()).collect()
    }

    /// Wait for at least one Instance to be available for this Endpoint
    pub async fn wait_for_instances(&self) -> Result<Vec<Instance>> {
        let mut instances: Vec<Instance> = vec![];
        if let InstanceSource::Dynamic(mut rx) = self.instance_source.as_ref().clone() {
            // wait for there to be 1 or more endpoints
            loop {
                let stored_transports = rx.borrow_and_update().to_vec();
                instances = stored_transports.into_iter().map(|st| st.key).collect();
                if instances.is_empty() {
                    rx.changed().await?;
                } else {
                    break;
                }
            }
        }
        Ok(instances)
    }

    /// Is this component know at startup and not discovered via etcd?
    pub fn is_static(&self) -> bool {
        self.endpoint.to_descriptor().is_static()
    }

    async fn get_or_create_dynamic_instance_source(
        endpoint: &Endpoint,
    ) -> Result<Arc<InstanceSource>> {
        // Try to get from cache first
        if let Some(cached) = Self::try_get_cached_instance_source(endpoint).await? {
            return Ok(cached);
        }

        // Set up new watcher
        let prefix_watcher = endpoint.discovery_storage()?.watch_prefix().await?;
        let (prefix, _watcher, kv_event_rx) = prefix_watcher.dissolve();

        let (watch_tx, watch_rx) = tokio::sync::watch::channel(Vec::<StoredTransport>::new());

        // Spawn background watcher task
        let drt = endpoint.drt();
        Self::spawn_instance_watcher(drt, prefix, watch_tx, kv_event_rx);

        // Create and cache the new instance source
        let instance_source = Arc::new(InstanceSource::Dynamic(watch_rx));
        Self::cache_instance_source(endpoint, &instance_source).await?;

        Ok(instance_source)
    }

    async fn try_get_cached_instance_source(endpoint: &Endpoint) -> Result<Option<Arc<InstanceSource>>> {
        let drt = endpoint.drt();
        let instance_sources_guard = drt.instance_sources();
        let mut instance_sources = instance_sources_guard.lock().await;

        if let Some(instance_source) = instance_sources.get(&endpoint.to_descriptor().identifier())
            .and_then(|weak| weak.upgrade())
        {
            return Ok(Some(instance_source));
        }

        // Clean up stale entry if it exists but couldn't upgrade
        instance_sources.remove(&endpoint.to_descriptor().identifier());
        Ok(None)
    }

    async fn cache_instance_source(
        endpoint: &Endpoint,
        instance_source: &Arc<InstanceSource>
    ) -> Result<()> {
        let drt = endpoint.drt();
        let instance_sources_guard = drt.instance_sources();
        let mut instance_sources = instance_sources_guard.lock().await;
        instance_sources.insert(endpoint.to_descriptor().identifier(), Arc::downgrade(instance_source));
        Ok(())
    }

    fn spawn_instance_watcher(
        drt: &DistributedRuntime,
        prefix: String,
        watch_tx: tokio::sync::watch::Sender<Vec<StoredTransport>>,
        mut kv_event_rx: tokio::sync::mpsc::Receiver<WatchEvent>,
    ) {
        let secondary = drt.runtime.secondary().clone();

        secondary.spawn(async move {
            tracing::debug!(prefix = %prefix, "Starting endpoint watcher");
            let mut instance_map: HashMap<Instance, StoredTransport> = HashMap::new();

            loop {
                let kv_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        tracing::debug!(prefix = %prefix, "All watchers closed; shutting down endpoint watcher");
                        break;
                    }
                    kv_event = kv_event_rx.recv() => {
                        match kv_event {
                            Some(event) => event,
                            None => {
                                tracing::debug!(prefix = %prefix, "Watch stream closed; shutting down endpoint watcher");
                                break;
                            }
                        }
                    }
                };

                let should_continue = match kv_event {
                    WatchEvent::Put(kv) => Self::handle_put_event(kv, &mut instance_map),
                    WatchEvent::Delete(kv) => Self::handle_delete_event(kv, &mut instance_map, &prefix),
                };

                if !should_continue {
                    break;
                }

                let instances: Vec<StoredTransport> = instance_map.values().cloned().collect();
                if watch_tx.send(instances).is_err() {
                    tracing::debug!(prefix = %prefix, "Unable to send watch updates; shutting down endpoint watcher");
                    break;
                }
            }

            tracing::debug!(prefix = %prefix, "Completed endpoint watcher");
            let _ = watch_tx.send(vec![]);
        });
    }

    fn handle_put_event(
        kv: etcd_client::KeyValue,
        instance_map: &mut HashMap<Instance, StoredTransport>
    ) -> bool {
        let Ok(key) = kv.key_str() else {
            tracing::error!("Unable to parse PUT event key as UTF-8");
            return false;
        };

        let Ok(instance) = Instance::parse(key) else {
            tracing::error!(key = key, "Failed to parse instance from key");
            return true; // Continue processing other events
        };

        let Ok(transport) = serde_json::from_slice::<TransportType>(kv.value()) else {
            tracing::error!(key = key, "Failed to deserialize transport type");
            return true; // Continue processing other events
        };

        instance_map.insert(
            instance.clone(),
            StoredValue { key: instance, value: transport }
        );
        true
    }

    fn handle_delete_event(
        kv: etcd_client::KeyValue,
        instance_map: &mut HashMap<Instance, StoredTransport>,
        prefix: &str
    ) -> bool {
        let Ok(key) = kv.key_str() else {
            tracing::error!(
                prefix = %prefix,
                "Unable to parse DELETE event key as UTF-8; shutting down endpoint watcher"
            );
            return false;
        };

        let Ok(instance) = Instance::parse(key) else {
            tracing::error!(key = key, "Failed to parse instance from delete key");
            return true; // Continue processing other events
        };

        instance_map.remove(&instance);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Runtime, distributed::DistributedConfig};
    use crate::descriptor::{Identifier, Instance};
    use crate::entity::Endpoint;

    async fn create_test_runtime() -> DistributedRuntime {
        let runtime = Runtime::from_current().unwrap();
        DistributedRuntime::from_settings_without_discovery(runtime).await.unwrap()
    }

    async fn create_test_runtime_with_etcd() -> Result<DistributedRuntime> {
        let runtime = Runtime::from_current()?;
        let mut config = DistributedConfig::from_settings(false);
        config.etcd_config.etcd_url = vec!["http://localhost:2379".to_string()];
        DistributedRuntime::new(runtime, config).await
    }

    #[tokio::test]
    async fn test_static_client_creation() {
        let drt = create_test_runtime().await;

        // Create a static endpoint
        let id = Identifier::new_endpoint("test", "service", "api").unwrap();
        let static_instance = Instance::new_static(id).unwrap();
        let endpoint = Endpoint::from_instance(static_instance, drt).unwrap();

        // Create client
        let client = Client::new(endpoint.clone()).await.unwrap();

        // Verify it's static
        assert!(client.is_static());
        assert!(matches!(client.instance_source.as_ref(), InstanceSource::Static));

        // Static clients should return empty instances
        assert_eq!(client.instances().len(), 0);
        assert_eq!(client.instance_ids().len(), 0);
    }

    #[tokio::test]
    async fn test_instance_filtering() {
        let drt = create_test_runtime().await;

        // Create test data
        let instances = vec![
            StoredTransport {
                key: Instance::new(
                    Identifier::new_endpoint("test", "svc", "api").unwrap(),
                    123
                ).unwrap(),
                value: TransportType::NatsTcp("nats://localhost:4222".to_string()),
            },
            StoredTransport {
                key: Instance::new_static(
                    Identifier::new_endpoint("test", "svc", "api").unwrap()
                ).unwrap(),
                value: TransportType::NatsTcp("nats://localhost:4223".to_string()),
            },
        ];

        // Create a dynamic client with test data
        let (_tx, rx) = tokio::sync::watch::channel(instances.clone());
        let instance_source = Arc::new(InstanceSource::Dynamic(rx));

        let endpoint = Endpoint::new("test", "svc", "api", drt).unwrap();
        let client = Client {
            endpoint,
            instance_source,
        };

        // Test instances() method
        let retrieved_instances = client.instances();
        assert_eq!(retrieved_instances.len(), 2);

        // Test instance_ids() - should filter out static instances
        let ids = client.instance_ids();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 123);
    }

    #[tokio::test]
    async fn test_wait_for_instances() {
        let drt = create_test_runtime().await;

        // Create a dynamic endpoint
        let (tx, rx) = tokio::sync::watch::channel(vec![]);
        let instance_source = Arc::new(InstanceSource::Dynamic(rx));

        let endpoint = Endpoint::new("test", "svc", "api", drt).unwrap();
        let client = Client {
            endpoint,
            instance_source,
        };

        // Spawn a task to send instances after a delay
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            let instances = vec![
                StoredTransport {
                    key: Instance::new(
                        Identifier::new_endpoint("test", "svc", "api").unwrap(),
                        456
                    ).unwrap(),
                    value: TransportType::NatsTcp("nats://localhost:4224".to_string()),
                },
            ];
            let _ = tx_clone.send(instances);
        });

        // Wait for instances
        let instances = client.wait_for_instances().await.unwrap();
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].instance_id(), Some(456));
    }

    #[tokio::test]
    async fn test_dynamic_client_with_etcd() {
        // Check if etcd is available
        if etcd_client::Client::connect(["localhost:2379"], None).await.is_err() {
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

        // Create a dynamic endpoint
        let endpoint = Endpoint::new("test", "client", "dynamic", drt.clone()).unwrap();

        // Create client
        let client = Client::new(endpoint.clone()).await.unwrap();

        // Verify it's dynamic
        assert!(!client.is_static());
        assert!(matches!(client.instance_source.as_ref(), InstanceSource::Dynamic(_)));

        // Initially should have no instances
        assert_eq!(client.instances().len(), 0);

        // Register an instance using another client
        let test_instance = Instance::new(
            Identifier::new_endpoint("test", "client", "dynamic").unwrap(),
            789
        ).unwrap();
        let test_endpoint = Endpoint::from_instance(test_instance.clone(), drt.clone()).unwrap();

        // Store transport info
        let transport = TransportType::NatsTcp("nats://localhost:4225".to_string());
        let storage = test_endpoint.storage().unwrap();
        storage.put(serde_json::to_vec(&transport).unwrap(), None).await.unwrap();

        // Wait a bit for the watcher to pick it up
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Check that the client sees the instance
        let instances = client.instances();
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].key.instance_id(), Some(789));

        // Add a second instance
        let test_instance2 = Instance::new(
            Identifier::new_endpoint("test", "client", "dynamic").unwrap(),
            790
        ).unwrap();
        let test_endpoint2 = Endpoint::from_instance(test_instance2.clone(), drt.clone()).unwrap();

        // Store transport info for second instance
        let transport2 = TransportType::NatsTcp("nats://localhost:4226".to_string());
        let storage2 = test_endpoint2.storage().unwrap();
        storage2.put(serde_json::to_vec(&transport2).unwrap(), None).await.unwrap();

        // Wait for the watcher to pick up the second instance
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Check that the client now sees both instances
        let instances = client.instances();
        assert_eq!(instances.len(), 2);

        // Verify both instance IDs are present
        let instance_ids: Vec<i64> = instances.iter()
            .filter_map(|st| st.key.instance_id())
            .collect();
        assert!(instance_ids.contains(&789));
        assert!(instance_ids.contains(&790));

        // Check instance IDs method
        let ids = client.instance_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&789));
        assert!(ids.contains(&790));

        // Clean up both instances
        storage.delete(None).await.unwrap();
        storage2.delete(None).await.unwrap();

        // Wait for deletions to propagate
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Verify both instances were removed
        assert_eq!(client.instances().len(), 0);
    }

    #[tokio::test]
    async fn test_instance_source_caching() {
        // Check if etcd is available
        if etcd_client::Client::connect(["localhost:2379"], None).await.is_err() {
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

        let endpoint1 = Endpoint::new("test", "cache", "endpoint1", drt.clone()).unwrap();
        let endpoint2 = Endpoint::new("test", "cache", "endpoint1", drt.clone()).unwrap(); // Same endpoint

        // Create two clients for the same endpoint
        let client1 = Client::new(endpoint1).await.unwrap();
        let client2 = Client::new(endpoint2).await.unwrap();

        // They should share the same instance source (via Arc)
        assert!(Arc::ptr_eq(&client1.instance_source, &client2.instance_source));
    }
}
