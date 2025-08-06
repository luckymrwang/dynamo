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

use crate::metrics::MetricsRegistry;
use prometheus::IntGauge;
use std::sync::Arc;

// Metrics configuration for etcd transport
#[derive(Clone, Debug)]
pub struct EtcdMetrics {
    pub etcd_block_total: Arc<IntGauge>,
    pub etcd_block_bytes_total: Arc<IntGauge>,
}

impl EtcdMetrics {
    pub fn new(etcd_block_total: Arc<IntGauge>, etcd_block_bytes_total: Arc<IntGauge>) -> Self {
        Self {
            etcd_block_total,
            etcd_block_bytes_total,
        }
    }

    pub fn from_endpoint(
        endpoint: &crate::component::Endpoint,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let etcd_block_total = endpoint.create_intgauge(
            "etcd_block_total",
            "Total number of etcd key-value pairs processed",
            &[],
        )?;

        let etcd_block_bytes_total = endpoint.create_intgauge(
            "etcd_block_bytes_total",
            "Total number of bytes processed in etcd blocks",
            &[],
        )?;

        Ok(Self::new(etcd_block_total, etcd_block_bytes_total))
    }
}
