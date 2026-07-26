//! Direct custom endpoint support — calls selected providers directly from the
//! client when Warp's server cannot proxy their streaming protocol reliably.

use futures::stream::{self, BoxStream, StreamExt};
use warp_multi_agent_api as api;

use crate::Error;

const CHATGPT_CODEX_BASE: &str = "https://chatgpt.com/backend-api/codex";

/// Returns true if the request's selected custom model is served by a local
/// HTTP endpoint (e.g. openai-oauth proxy at 127.0.0.1) that Warp's server
/// would reject.
pub fn should_use_direct_local(request: &api::Request) -> bool {
    let Some(settings) = &request.settings else {
        log::warn!("should_use_direct_local: no settings");
        return false;
    };

    // Check if any custom provider in the request uses a local HTTP URL.
    if let Some(providers) = &settings.custom_model_providers {
        for provider in &providers.providers {
            let url = provider.base_url.trim();
            if is_direct_endpoint(url) {
                // Check if the selected model belongs to this provider.
                let selected = selected_model_config(settings);
                if provider.models.iter().any(|m| m.config_key == selected) {
                    return true;
                } else {
                    log::warn!(
                        "should_use_direct_local: direct endpoint {} but config_key '{}' not found in provider models",
                        url,
                        selected,
                    );
                }
            }
        }
    } else {
        log::warn!("should_use_direct_local: no custom_model_providers in settings");
    }
    false
}

fn is_local_http_url(url: &str) -> bool {
    url.starts_with("http://127.")
        || url.starts_with("http://localhost")
        || url.starts_with("http://0.0.0.0")
        || url.starts_with("http://192.168.")
        || url.starts_with("http://10.")
}

fn is_chatgpt_endpoint(url: &str) -> bool {
    let url = url.trim().trim_end_matches('/');
    url == CHATGPT_CODEX_BASE || url.starts_with("https://chatgpt.com/backend-api/codex/")
}

fn is_opencode_endpoint(url: &str) -> bool {
    let url = url.trim().trim_end_matches('/');
    url.starts_with("https://opencode.ai/zen/")
}

fn is_direct_endpoint(url: &str) -> bool {
    is_local_http_url(url) || is_chatgpt_endpoint(url) || is_opencode_endpoint(url)
}

fn selected_model_config(settings: &api::request::Settings) -> String {
    settings
        .model_config
        .as_ref()
        .map(|mc| mc.base.clone())
        .unwrap_or_default()
}

/// Find the local provider's base URL and model slug for the selected model.
fn find_local_endpoint(request: &api::Request) -> Option<(String, String, String)> {
    let settings = request.settings.as_ref()?;
    let providers = settings.custom_model_providers.as_ref()?;
    let selected = selected_model_config(settings);

    for provider in &providers.providers {
        log::info!(
            "find_local_endpoint: checking provider url={} is_direct={}",
            provider.base_url,
            is_direct_endpoint(&provider.base_url),
        );
        if is_direct_endpoint(&provider.base_url) {
            for m in &provider.models {
                log::info!(
                    "find_local_endpoint: model slug={} config_key={} selected={} match={}",
                    m.slug,
                    m.config_key,
                    selected,
                    m.config_key == selected,
                );
            }
            if let Some(model) = provider.models.iter().find(|m| m.config_key == selected) {
                return Some((
                    provider.base_url.clone(),
                    model.slug.clone(),
                    provider.api_key.clone(),
                ));
            }
        }
    }
    log::warn!(
        "find_local_endpoint: no matching direct endpoint found for selected={}",
        selected
    );
    None
}

/// Call the local endpoint directly and stream back ResponseEvents.
pub async fn generate_local_output(
    request: &api::Request,
) -> Result<BoxStream<'static, Result<api::ResponseEvent, Error>>, Error> {
    let (base_url, model_slug, api_key) = find_local_endpoint(request).unwrap_or_else(|| {
        log::warn!("generate_local_output: find_local_endpoint returned None, using defaults");
        (
            CHATGPT_CODEX_BASE.to_string(),
            "gpt-5.6-sol".to_string(),
            String::new(),
        )
    });

    log::info!(
        "generate_local_output: base_url={} model_slug={} has_api_key={}",
        base_url,
        model_slug,
        !api_key.is_empty(),
    );

    let messages = extract_messages(request);
    let is_chatgpt = is_chatgpt_endpoint(&base_url);
    let is_opencode = is_opencode_endpoint(&base_url);
    let chat_request = if is_chatgpt {
        let instructions = messages
            .iter()
            .filter(|message| message.get("role").and_then(|role| role.as_str()) == Some("system"))
            .filter_map(|message| message.get("content").and_then(|content| content.as_str()))
            .collect::<Vec<_>>()
            .join("\n\n");
        let input = messages
            .iter()
            .filter(|message| message.get("role").and_then(|role| role.as_str()) != Some("system"))
            .map(|message| {
                let role = message
                    .get("role")
                    .and_then(|value| value.as_str())
                    .unwrap_or("user");
                serde_json::json!({
                    "role": role,
                    "content": [{
                        "type": if role == "assistant" { "output_text" } else { "input_text" },
                        "text": message.get("content").and_then(|v| v.as_str()).unwrap_or("")
                    }]
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "model": model_slug,
            "instructions": if instructions.is_empty() { "You are a helpful AI assistant." } else { &instructions },
            "input": input,
            "tools": direct_tool_definitions(true),
            "stream": true,
            "store": false,
        })
    } else {
        serde_json::json!({
            "model": model_slug,
            "messages": messages,
            "tools": direct_tool_definitions(false),
            "tool_choice": "auto",
            "stream": true,
        })
    };

    let chat_url = if is_chatgpt {
        format!("{}/responses", base_url.trim_end_matches('/'))
    } else {
        format!("{}/chat/completions", base_url.trim_end_matches('/'))
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| Error::Authentication(anyhow::anyhow!("HTTP client error: {e}")))?;

    let mut request_builder = client
        .post(&chat_url)
        .header("Content-Type", "application/json");
    if is_chatgpt {
        request_builder = request_builder
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Origin", "https://chatgpt.com")
            .header("Referer", "https://chatgpt.com/codex");
    } else if is_opencode {
        request_builder = request_builder.header("Authorization", format!("Bearer {api_key}"));
    } else {
        // Preserve the existing openai-oauth/local-provider contract.
        request_builder = request_builder.header("Authorization", "Bearer openai-oauth");
    }
    let response = request_builder
        .json(&chat_request)
        .send()
        .await
        .map_err(|e| Error::Authentication(anyhow::anyhow!("Request failed: {e}")))?;

    let status = response.status();
    log::info!(
        "generate_local_output: HTTP response status={} url={}",
        status,
        chat_url,
    );
    for (name, value) in response.headers() {
        log::debug!(
            "generate_local_output: response header {}={}",
            name,
            value.to_str().unwrap_or("<binary>"),
        );
    }

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        log::warn!("generate_local_output: error response body: {body}");
        return Err(Error::Authentication(anyhow::anyhow!(
            "Local endpoint error ({status}): {body}"
        )));
    }

    // The history layer treats StreamInit.conversation_id as a server
    // conversation token. A random local UUID causes futile metadata lookups
    // and "No metadata returned" warnings.
    let conversation_id = request
        .metadata
        .as_ref()
        .map(|metadata| metadata.conversation_id.clone())
        .unwrap_or_default();
    let request_id = uuid::Uuid::new_v4().to_string();
    // Existing conversations already contain a server-backed root task. Sending
    // CreateTask for that task attempts to upgrade it a second time and causes
    // UnexpectedUpgrade/ExchangeNotFound errors. Only create a synthetic server
    // task for a brand-new conversation.
    // Preserve the original local-provider protocol. The special reuse path
    // is required only for ChatGPT, whose direct Responses stream has no Warp
    // server to emit the normal task/message actions.
    let has_task_context = request
        .task_context
        .as_ref()
        .is_some_and(|context| !context.tasks.is_empty());
    // Any direct endpoint with an existing task must append to that task. A
    // CreateTask action is only valid for the first request in a new local
    // conversation; emitting it on a follow-up upgrades an already-server
    // task and causes the cascade of ExchangeNotFound errors.
    let needs_task_upgrade = !has_task_context;
    log::info!(
        "generate_local_output: is_chatgpt={} has_task_context={} needs_task_upgrade={}",
        is_chatgpt,
        has_task_context,
        needs_task_upgrade,
    );
    let task_id = if needs_task_upgrade {
        uuid::Uuid::new_v4().to_string()
    } else {
        request
            .task_context
            .as_ref()
            .and_then(|context| {
                context
                    .tasks
                    .iter()
                    .find(|task| {
                        task.dependencies
                            .as_ref()
                            .is_none_or(|dependencies| dependencies.parent_task_id.is_empty())
                    })
                    .or_else(|| context.tasks.first())
            })
            .map(|task| task.id.clone())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
    };
    let message_id = uuid::Uuid::new_v4().to_string();

    let init_event = api::ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                conversation_id,
                request_id,
                run_id: String::new(),
            },
        )),
    };

    let create_task_event =
        needs_task_upgrade.then(|| build_create_task_event(&task_id, &message_id));
    let byte_stream = response.bytes_stream();
    let task_id_clone = task_id.clone();
    let message_id_clone = message_id.clone();

    // Track whether any content was ever emitted, so we can warn on silent streams.
    let chunk_stream = stream::unfold(
        (byte_stream, String::new(), String::new(), Vec::new(), false),
        move |(mut byte_stream, mut buffer, mut output_text, mut tool_calls, mut completed)| {
            let task_id = task_id_clone.clone();
            let message_id = message_id_clone.clone();
            async move {
                if completed {
                    return None;
                }
                loop {
                    if let Some(data) = extract_sse_data(&mut buffer) {
                        log::debug!("chatgpt sse data: {data}");
                        if data == "[DONE]" {
                            completed = true;
                            let event = build_add_message_event(
                                &task_id,
                                &message_id,
                                &output_text,
                                &tool_calls,
                            );
                            return Some((
                                Ok(event),
                                (byte_stream, buffer, output_text, tool_calls, completed),
                            ));
                        }
                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&data) {
                            let event_type =
                                chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            let content = if is_chatgpt {
                                log::debug!(
                                    "chatgpt chunk type={event_type} has_delta={}",
                                    chunk.get("delta").is_some(),
                                );
                                match event_type {
                                    "response.completed"
                                    | "response.incomplete"
                                    | "response.failed" => {
                                        completed = true;
                                        let event = build_add_message_event(
                                            &task_id,
                                            &message_id,
                                            &output_text,
                                            &tool_calls,
                                        );
                                        return Some((
                                            Ok(event),
                                            (
                                                byte_stream,
                                                buffer,
                                                output_text,
                                                tool_calls,
                                                completed,
                                            ),
                                        ));
                                    }
                                    _ => chunk.get("delta").and_then(|v| v.as_str()),
                                }
                            } else {
                                chunk
                                    .pointer("/choices/0/delta/content")
                                    .and_then(|v| v.as_str())
                            };
                            if let Some(tool_call) = extract_tool_call(&chunk, is_chatgpt) {
                                merge_tool_call(&mut tool_calls, tool_call);
                            }
                            if let Some(content) = content {
                                if !content.is_empty() {
                                    output_text.push_str(content);
                                }
                            }
                        } else {
                            log::debug!("chatgpt sse: failed to parse JSON from data: {data}");
                        }
                        continue;
                    }
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            log::debug!("chatgpt raw bytes: {}", String::from_utf8_lossy(&bytes));
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((
                                Err(Error::Authentication(anyhow::anyhow!("Stream error: {e}"))),
                                (byte_stream, buffer, output_text, tool_calls, completed),
                            ));
                        }
                        None => {
                            completed = true;
                            let event = build_add_message_event(
                                &task_id,
                                &message_id,
                                &output_text,
                                &tool_calls,
                            );
                            return Some((
                                Ok(event),
                                (byte_stream, buffer, output_text, tool_calls, completed),
                            ));
                        }
                    }
                }
            }
        },
    );

    let finished_event = api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                )),
                ..Default::default()
            },
        )),
    };

    let full_stream = stream::once(async move { Ok(init_event) })
        .chain(stream::iter(create_task_event.into_iter().map(Ok)))
        .chain(chunk_stream)
        .chain(stream::once(async move { Ok(finished_event) }));

    Ok(full_stream.boxed())
}

fn build_create_task_event(task_id: &str, _message_id: &str) -> api::ResponseEvent {
    let task = api::Task {
        id: task_id.to_string(),
        ..Default::default()
    };
    let client_action = api::ClientAction {
        action: Some(api::client_action::Action::CreateTask(
            api::client_action::CreateTask { task: Some(task) },
        )),
    };
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![client_action],
            },
        )),
    }
}

#[derive(Debug, Clone)]
struct ProviderToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn extract_tool_call(chunk: &serde_json::Value, responses_api: bool) -> Option<ProviderToolCall> {
    if responses_api {
        let event_type = chunk.get("type")?.as_str()?;
        if event_type == "response.function_call_arguments.done" {
            return Some(ProviderToolCall {
                id: chunk.get("call_id")?.as_str()?.to_string(),
                name: chunk.get("name")?.as_str()?.to_string(),
                arguments: chunk.get("arguments")?.as_str()?.to_string(),
            });
        }
        if event_type == "response.output_item.done"
            && chunk.pointer("/item/type").and_then(|v| v.as_str()) == Some("function_call")
        {
            return Some(ProviderToolCall {
                id: chunk.pointer("/item/call_id")?.as_str()?.to_string(),
                name: chunk.pointer("/item/name")?.as_str()?.to_string(),
                arguments: chunk.pointer("/item/arguments")?.as_str()?.to_string(),
            });
        }
        None
    } else {
        let call = chunk.pointer("/choices/0/delta/tool_calls/0")?;
        let function = call.get("function")?;
        Some(ProviderToolCall {
            id: call
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            name: function
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            arguments: function
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })
    }
}

fn merge_tool_call(calls: &mut Vec<ProviderToolCall>, incoming: ProviderToolCall) {
    if incoming.id.is_empty() && incoming.name.is_empty() && incoming.arguments.is_empty() {
        return;
    }
    // OpenAI-compatible streaming sends the first tool chunk with the id/name,
    // then sends argument fragments with neither field populated. Those
    // fragments belong to the most recently opened tool call.
    if incoming.id.is_empty()
        && incoming.name.is_empty()
        && !incoming.arguments.is_empty()
        && let Some(existing) = calls.last_mut()
    {
        existing.arguments.push_str(&incoming.arguments);
        return;
    }
    let key = if incoming.id.is_empty() {
        calls.len().to_string()
    } else {
        incoming.id.clone()
    };
    if let Some(existing) = calls.iter_mut().find(|call| {
        (!key.is_empty() && call.id == key)
            || (call.id.is_empty() && !incoming.name.is_empty() && call.name == incoming.name)
    }) {
        if existing.name.is_empty() {
            existing.name = incoming.name;
        }
        if existing.id.is_empty() {
            existing.id = incoming.id;
        }
        existing.arguments.push_str(&incoming.arguments);
    } else {
        calls.push(incoming);
    }
}

fn direct_tool_definitions(responses_api: bool) -> Vec<serde_json::Value> {
    let definitions = vec![
        (
            "read_files",
            "Read one or more files from the current workspace.",
            serde_json::json!({
                "type": "object",
                "properties": { "files": { "type": "array", "items": { "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] } } },
                "required": ["files"]
            }),
        ),
        (
            "grep",
            "Search the workspace for text or regular expressions.",
            serde_json::json!({
                "type": "object",
                "properties": { "queries": { "type": "array", "items": { "type": "string" } }, "path": { "type": "string" } },
                "required": ["queries", "path"]
            }),
        ),
        (
            "file_glob",
            "Find files matching glob patterns in the workspace.",
            serde_json::json!({
                "type": "object",
                "properties": { "patterns": { "type": "array", "items": { "type": "string" } }, "path": { "type": "string" } },
                "required": ["patterns", "path"]
            }),
        ),
        (
            "run_shell_command",
            "Run a shell command in the workspace. Use only when needed.",
            serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" }, "is_read_only": { "type": "boolean" } },
                "required": ["command"]
            }),
        ),
    ];
    definitions
        .into_iter()
        .map(|(name, description, parameters)| {
            if responses_api {
                serde_json::json!({ "type": "function", "name": name, "description": description, "parameters": parameters })
            } else {
                serde_json::json!({ "type": "function", "function": { "name": name, "description": description, "parameters": parameters } })
            }
        })
        .collect()
}

fn build_add_message_event(
    task_id: &str,
    message_id: &str,
    text: &str,
    tool_calls: &[ProviderToolCall],
) -> api::ResponseEvent {
    let messages = if tool_calls.is_empty() {
        vec![build_agent_message(task_id, message_id, text)]
    } else {
        tool_calls
            .iter()
            .filter_map(|call| build_tool_call_message(task_id, call))
            .collect()
    };
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_string(),
                            messages,
                        },
                    )),
                }],
            },
        )),
    }
}

fn build_tool_call_message(task_id: &str, call: &ProviderToolCall) -> Option<api::Message> {
    let args: serde_json::Value = serde_json::from_str(&call.arguments).ok()?;
    let tool = match call.name.as_str() {
        "read_files" => {
            let files = args
                .get("files")?
                .as_array()?
                .iter()
                .filter_map(|file| {
                    Some(api::message::tool_call::read_files::File {
                        name: file.get("name")?.as_str()?.to_string(),
                        line_ranges: vec![],
                    })
                })
                .collect();
            api::message::tool_call::Tool::ReadFiles(api::message::tool_call::ReadFiles { files })
        }
        "grep" => api::message::tool_call::Tool::Grep(api::message::tool_call::Grep {
            queries: args
                .get("queries")?
                .as_array()?
                .iter()
                .filter_map(|q| q.as_str().map(str::to_string))
                .collect(),
            path: args.get("path")?.as_str()?.to_string(),
        }),
        "file_glob" => {
            api::message::tool_call::Tool::FileGlobV2(api::message::tool_call::FileGlobV2 {
                patterns: args
                    .get("patterns")?
                    .as_array()?
                    .iter()
                    .filter_map(|p| p.as_str().map(str::to_string))
                    .collect(),
                search_dir: args.get("path")?.as_str()?.to_string(),
                max_matches: 0,
                min_depth: 0,
                max_depth: 0,
            })
        }
        "run_shell_command" => api::message::tool_call::Tool::RunShellCommand(
            api::message::tool_call::RunShellCommand {
                command: args.get("command")?.as_str()?.to_string(),
                is_read_only: args
                    .get("is_read_only")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                ..Default::default()
            },
        ),
        _ => return None,
    };
    Some(api::Message {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id: call.id.clone(),
            tool: Some(tool),
        })),
        ..Default::default()
    })
}

fn build_agent_message(task_id: &str, message_id: &str, text: &str) -> api::Message {
    api::Message {
        id: message_id.to_string(),
        task_id: task_id.to_string(),
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: text.to_string(),
            },
        )),
        ..Default::default()
    }
}

fn extract_messages(request: &api::Request) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();

    // History from task context.
    if let Some(tc) = &request.task_context {
        for task in &tc.tasks {
            for msg in &task.messages {
                match &msg.message {
                    Some(api::message::Message::UserQuery(q)) => {
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": q.query
                        }));
                    }
                    Some(api::message::Message::AgentOutput(o)) if !o.text.is_empty() => {
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": o.text
                        }));
                    }
                    Some(api::message::Message::ToolCall(tool_call)) => {
                        let tool_name = tool_call
                            .tool
                            .as_ref()
                            .map(|tool| format!("{tool:?}"))
                            .unwrap_or_default();
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": format!("Requested workspace tool {}: {}", tool_call.tool_call_id, tool_name)
                        }));
                    }
                    Some(api::message::Message::ToolCallResult(result)) => {
                        messages.push(serde_json::json!({
                            // ChatGPT Responses and OpenCode's upstream both
                            // reject the chat-completions `role: tool` form
                            // on their compatibility endpoints. Preserve the
                            // result as explicit user context instead.
                            "role": "user",
                            "content": format!("[Workspace tool result for {}]\n{:?}", result.tool_call_id, result)
                        }));
                    }
                    _ => {}
                }
            }
        }
    }

    // Current user input.
    if let Some(input) = &request.input {
        if let Some(api::request::input::Type::UserInputs(ui)) = &input.r#type {
            for user_input in &ui.inputs {
                if let Some(api::request::input::user_inputs::user_input::Input::UserQuery(q)) =
                    &user_input.input
                {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": q.query
                    }));
                } else if let Some(
                    api::request::input::user_inputs::user_input::Input::ToolCallResult(result),
                ) = &user_input.input
                {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": format!("[Workspace tool result for {}]\n{:?}", result.tool_call_id, result)
                    }));
                }
            }
        }
    }

    if messages.is_empty() {
        messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));
    }

    messages
}

fn extract_sse_data(buffer: &mut String) -> Option<String> {
    loop {
        let boundary = buffer.find("\n\n").or_else(|| buffer.find("\r\n\r\n"));
        if let Some(pos) = boundary {
            let event_block = buffer[..pos].to_string();
            let consume_to = if buffer[pos..].starts_with("\r\n\r\n") {
                pos + 4
            } else {
                pos + 2
            };
            buffer.drain(..consume_to);
            let data: Vec<&str> = event_block
                .lines()
                .filter_map(|l| l.trim().strip_prefix("data:").map(|d| d.trim()))
                .collect();
            if data.is_empty() {
                continue;
            }
            let joined = data.join("\n");
            if joined.is_empty() {
                continue;
            }
            return Some(joined);
        } else {
            return None;
        }
    }
}
