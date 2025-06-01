// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    extract::{ConnectInfo, MatchedPath, Request},
    http::HeaderMap,
    middleware::Next,
    response::Response,
};
use std::{
    net::SocketAddr,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tower_http::trace::TraceLayer;

/// HTTP request logging middleware that logs timestamps and duration for each request
pub async fn request_logging_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    matched_path: Option<MatchedPath>,
    request: Request,
    next: Next,
) -> Response {
    let start_time = Instant::now();
    let start_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let method = request.method().clone();
    let path = matched_path
        .as_ref()
        .map(|mp| mp.as_str())
        .unwrap_or_else(|| request.uri().path());
    let user_agent = request
        .headers()
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");
    let content_length = request
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    // Generate request ID if not present in headers
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            // This will be overridden by individual handlers, but provides a fallback
            "middleware-generated".to_string()
        });

    tracing::info!(
        request_id = %request_id,
        start_timestamp = %start_timestamp,
        method = %method,
        path = %path,
        client_ip = %addr.ip(),
        user_agent = %user_agent,
        content_length = %content_length,
        "HTTP request received"
    );

    // Process the request
    let response = next.run(request).await;

    let elapsed = start_time.elapsed();
    let end_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let status = response.status();
    let response_content_length = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    tracing::info!(
        request_id = %request_id,
        end_timestamp = %end_timestamp,
        duration_ms = %elapsed.as_millis(),
        status_code = %status.as_u16(),
        response_content_length = %response_content_length,
        "HTTP request completed"
    );

    response
}

/// Creates a TraceLayer for HTTP request/response tracing
pub fn create_trace_layer(
) -> TraceLayer<tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>>
{
    TraceLayer::new_for_http()
}

/// Request context helper for extracting useful information
pub struct RequestContext {
    pub request_id: String,
    pub start_time: Instant,
    pub start_timestamp: u128,
    pub client_ip: Option<String>,
    pub user_agent: Option<String>,
}

impl RequestContext {
    pub fn new() -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            start_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            client_ip: None,
            user_agent: None,
        }
    }

    pub fn from_headers(headers: &HeaderMap) -> Self {
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let user_agent = headers
            .get("user-agent")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let client_ip = headers
            .get("x-forwarded-for")
            .or_else(|| headers.get("x-real-ip"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        Self {
            request_id,
            start_time: Instant::now(),
            start_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            client_ip,
            user_agent,
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    pub fn end_timestamp(&self) -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }

    pub fn log_start(&self, endpoint: &str, model: Option<&str>) {
        match model {
            Some(model) => {
                tracing::info!(
                    request_id = %self.request_id,
                    start_timestamp = %self.start_timestamp,
                    endpoint = %endpoint,
                    model = %model,
                    client_ip = ?self.client_ip,
                    user_agent = ?self.user_agent,
                    "HTTP request started"
                );
            }
            None => {
                tracing::info!(
                    request_id = %self.request_id,
                    start_timestamp = %self.start_timestamp,
                    endpoint = %endpoint,
                    client_ip = ?self.client_ip,
                    user_agent = ?self.user_agent,
                    "HTTP request started"
                );
            }
        }
    }

    pub fn log_success(&self, streaming: bool, additional_info: Option<&str>) {
        let end_timestamp = self.end_timestamp();
        let duration_ms = self.elapsed().as_millis();

        match additional_info {
            Some(info) => {
                tracing::info!(
                    request_id = %self.request_id,
                    end_timestamp = %end_timestamp,
                    duration_ms = %duration_ms,
                    streaming = %streaming,
                    additional_info = %info,
                    "HTTP request completed successfully"
                );
            }
            None => {
                tracing::info!(
                    request_id = %self.request_id,
                    end_timestamp = %end_timestamp,
                    duration_ms = %duration_ms,
                    streaming = %streaming,
                    "HTTP request completed successfully"
                );
            }
        }
    }

    pub fn log_error(&self, error: &str, error_type: &str) {
        let end_timestamp = self.end_timestamp();
        let duration_ms = self.elapsed().as_millis();

        tracing::error!(
            request_id = %self.request_id,
            end_timestamp = %end_timestamp,
            duration_ms = %duration_ms,
            error = %error,
            error_type = %error_type,
            "HTTP request failed"
        );
    }
}
