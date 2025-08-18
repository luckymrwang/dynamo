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

use futures::{Stream, StreamExt};
use std::collections::HashMap;

use super::{NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse};
use crate::protocols::{
    codec::{Message, SseCodecError},
    convert_sse_stream,
    openai::chat_completions::{NvChatChoice, NvChatCompletionResponseMessage},
    Annotated,
};

use dynamo_runtime::engine::DataStream;

/// Aggregates a stream of [`NvCreateChatCompletionStreamResponse`]s into a single
/// [`NvCreateChatCompletionResponse`]. This struct accumulates incremental responses
/// from a streaming OpenAI API call into a complete final response.
pub struct DeltaAggregator {
    /// Unique identifier for the chat completion.
    id: String,
    /// Model name used for the chat completion.
    model: String,
    /// Timestamp (Unix epoch) indicating when the response was created.
    created: u32,
    /// Optional usage statistics for the completion request.
    usage: Option<async_openai::types::CompletionUsage>,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Map of incremental response choices, keyed by index.
    choices: HashMap<u32, DeltaChoice>,
    /// Optional error message if an error occurs during aggregation.
    error: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<async_openai::types::ServiceTierResponse>,
}

/// Represents the accumulated state of a single chat choice during streaming aggregation.
struct DeltaChoice {
    /// The index of the choice in the completion.
    index: u32,
    /// The accumulated text content for the choice.
    text: String,
    /// The role associated with this message (e.g., `system`, `user`, `assistant`).
    role: Option<async_openai::types::Role>,
    /// The reason the completion was finished (if applicable).
    finish_reason: Option<async_openai::types::FinishReason>,
    /// Optional log probabilities for the chat choice.
    logprobs: Option<async_openai::types::ChatChoiceLogprobs>,
    // Optional tool calls for the chat choice.
    tool_calls: Option<Vec<async_openai::types::ChatCompletionMessageToolCall>>,

    /// Optional function call for the reasoning content
    reasoning_content: Option<String>,
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    /// Creates a new, empty [`DeltaAggregator`] instance.
    pub fn new() -> Self {
        Self {
            id: "".to_string(),
            model: "".to_string(),
            created: 0,
            usage: None,
            system_fingerprint: None,
            choices: HashMap::new(),
            error: None,
            service_tier: None,
        }
    }

    /// Aggregates a stream of [`NvCreateChatCompletionStreamResponse`]s into a single
    /// [`NvCreateChatCompletionResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation is successful.
    /// * `Err(String)` if an error occurs during processing.
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                // Attempt to unwrap the delta, capturing any errors.
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none() && delta.data.is_some() {
                    // Extract the data payload from the delta.
                    let delta = delta.data.unwrap();
                    aggregator.id = delta.id;
                    aggregator.model = delta.model;
                    aggregator.created = delta.created;
                    aggregator.service_tier = delta.service_tier;

                    // Aggregate usage statistics if available.
                    if let Some(usage) = delta.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // Aggregate choices incrementally.
                    for choice in delta.choices {
                        let state_choice =
                            aggregator
                                .choices
                                .entry(choice.index)
                                .or_insert(DeltaChoice {
                                    index: choice.index,
                                    text: "".to_string(),
                                    role: choice.delta.role,
                                    finish_reason: None,
                                    logprobs: choice.logprobs,
                                    tool_calls: None,
                                    reasoning_content: None,
                                });

                        // Append content if available.
                        if let Some(content) = &choice.delta.content {
                            state_choice.text.push_str(content);
                        }

                        if let Some(reasoning_content) = &choice.delta.reasoning_content {
                            if state_choice.reasoning_content.is_none() {
                                state_choice.reasoning_content = Some("".to_string());
                            }
                            state_choice
                                .reasoning_content
                                .as_mut()
                                .expect("Reason Content")
                                .push_str(reasoning_content);
                        }

                        // Update finish reason if provided.
                        if let Some(finish_reason) = choice.finish_reason {
                            state_choice.finish_reason = Some(finish_reason);
                        }
                    }
                }
                aggregator
            })
            .await;

        // Return early if an error was encountered.
        let mut aggregator = if let Some(error) = aggregator.error {
            return Err(error);
        } else {
            aggregator
        };

        // After aggregation, inspect each choice's text for tool call syntax
        for choice in aggregator.choices.values_mut() {
            if choice.tool_calls.is_none() {
                if let Ok(Some(tool_call)) =
                    crate::postprocessor::tool_calling::tools::try_tool_call_parse_aggregate(
                        &choice.text,
                        None,
                    )
                {
                    tracing::debug!(
                        tool_call_id = %tool_call.id,
                        function_name = %tool_call.function.name,
                        arguments = %tool_call.function.arguments,
                        "Parsed structured tool call from aggregated content"
                    );

                    choice.tool_calls = Some(vec![tool_call]);
                    choice.text.clear();
                    choice.finish_reason = Some(async_openai::types::FinishReason::ToolCalls);
                }
            }
        }

        // Extract aggregated choices and sort them by index.
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(NvChatChoice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        // Construct the final response object.
        let response = NvCreateChatCompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "chat.completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
            service_tier: aggregator.service_tier,
        };

        // let response = NvCreateChatCompletionResponse { inner };

        Ok(response)
    }
}

#[allow(deprecated)]
impl From<DeltaChoice> for NvChatChoice {
    /// Converts a [`DeltaChoice`] into an [`async_openai::types::ChatChoice`].
    ///
    /// # Note
    /// The `function_call` field is deprecated.
    fn from(delta: DeltaChoice) -> Self {
        NvChatChoice {
            message: NvChatCompletionResponseMessage {
                role: delta.role.expect("delta should have a Role"),
                content: if delta.tool_calls.is_some() {
                    None
                } else {
                    Some(delta.text)
                },
                tool_calls: delta.tool_calls,
                refusal: None,
                function_call: None,
                audio: None,
                reasoning_content: delta.reasoning_content,
            },
            index: delta.index,
            finish_reason: delta.finish_reason,
            logprobs: delta.logprobs,
        }
    }
}

impl NvCreateChatCompletionResponse {
    /// Converts an SSE stream into a [`NvCreateChatCompletionResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of SSE messages containing chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        let stream = convert_sse_stream::<NvCreateChatCompletionStreamResponse>(stream);
        NvCreateChatCompletionResponse::from_annotated_stream(stream).await
    }

    /// Aggregates an annotated stream of chat completion responses into a final response.
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}

#[cfg(test)]
mod tests {

    use crate::protocols::openai::chat_completions::{
        NvChatChoiceStream, NvChatCompletionStreamResponseDelta,
    };

    use super::*;
    use futures::stream;

    #[allow(deprecated)]
    fn create_test_delta(
        index: u32,
        text: &str,
        role: Option<async_openai::types::Role>,
        finish_reason: Option<async_openai::types::FinishReason>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        // ALLOW: function_call is deprecated
        let delta = NvChatCompletionStreamResponseDelta {
            content: Some(text.to_string()),
            function_call: None,
            tool_calls: None,
            role,
            refusal: None,
            reasoning_content: None,
        };
        let choice = NvChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs: None,
        };

        let data = NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b-instruct".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
        };

        Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        // Create an empty stream
        let stream: DataStream<Annotated<NvCreateChatCompletionStreamResponse>> =
            Box::pin(stream::empty());

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify that the response is empty and has default values
        assert_eq!(response.id, "");
        assert_eq!(response.model, "");
        assert_eq!(response.created, 0);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 0);
        assert!(response.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta =
            create_test_delta(0, "Hello,", Some(async_openai::types::Role::User), None);

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.id, "test_id");
        assert_eq!(response.model, "meta/llama-3.1-8b-instruct");
        assert_eq!(response.created, 1234567890);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello,");
        assert!(choice.finish_reason.is_none());
        assert_eq!(choice.message.role, async_openai::types::Role::User);
        assert!(response.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 =
            create_test_delta(0, "Hello,", Some(async_openai::types::Role::User), None);
        let annotated_delta2 = create_test_delta(
            0,
            " world!",
            None,
            Some(async_openai::types::FinishReason::Stop),
        );

        // Create a stream
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello, world!");
        assert_eq!(
            choice.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice.message.role, async_openai::types::Role::User);
    }

    #[allow(deprecated)]
    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        // ALLOW: function_call is deprecated
        let data = NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "test_model".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![
                NvChatChoiceStream {
                    index: 0,
                    delta: NvChatCompletionStreamResponseDelta {
                        role: Some(async_openai::types::Role::Assistant),
                        content: Some("Choice 0".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(async_openai::types::FinishReason::Stop),
                    logprobs: None,
                },
                NvChatChoiceStream {
                    index: 1,
                    delta: NvChatCompletionStreamResponseDelta {
                        role: Some(async_openai::types::Role::Assistant),
                        content: Some("Choice 1".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(async_openai::types::FinishReason::Stop),
                    logprobs: None,
                },
            ],
            object: "chat.completion".to_string(),
        };

        // Wrap it in Annotated and create a stream
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let mut response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 2);
        response.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(choice0.message.content.as_ref().unwrap(), "Choice 0");
        assert_eq!(
            choice0.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice0.message.role, async_openai::types::Role::Assistant);

        let choice1 = &response.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(choice1.message.content.as_ref().unwrap(), "Choice 1");
        assert_eq!(
            choice1.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice1.message.role, async_openai::types::Role::Assistant);
    }
}
