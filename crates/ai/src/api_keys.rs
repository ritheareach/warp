use std::time::{Duration, SystemTime};

#[cfg(not(target_family = "wasm"))]
use futures::channel::oneshot;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use warp_errors::report_error;
use warp_multi_agent_api as api;
use warpui_core::{Entity, ModelContext, SingletonEntity};
use warpui_extras::secure_storage::{self, AppContextExt};

pub use crate::aws_credentials::{AwsCredentials, AwsCredentialsState};
pub use crate::geap_credentials::{
    GEAP_REFRESH_LEAD_TIME, GeapCredentials, GeapCredentialsState, GeapFederation, GeapMintBinding,
    LoadGeapCredentialsError,
};

const SECURE_STORAGE_KEY: &str = "AiApiKeys";

/// Secure-storage key for the connected xAI/Grok subscription's OAuth tokens.
/// Kept separate from [`SECURE_STORAGE_KEY`] because these are OAuth tokens with
/// a refresh lifecycle, not a user-pasted static key.
const GROK_SECURE_STORAGE_KEY: &str = "GrokOAuthTokens";

/// Secure-storage key for the connected ChatGPT subscription's OAuth tokens.
const CHATGPT_SECURE_STORAGE_KEY: &str = "ChatGptOAuthTokens";

/// Emitted when user-provided API keys are updated in-memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeyManagerEvent {
    KeysUpdated,
}

/// User-provided API keys for AI providers.
///
/// These are used for "Bring Your Own API Key" functionality, allowing
/// users to use their own API keys instead of Warp's.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiKeys {
    pub google: Option<String>,
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub open_router: Option<String>,
    pub custom_endpoints: Vec<CustomEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomEndpoint {
    pub name: String,
    pub url: String,
    pub api_key: String,
    pub models: Vec<CustomEndpointModel>,
    pub schema: CustomEndpointSchema,
    /// Whether this is a local server, cloud API, or proxy aggregator.
    pub endpoint_kind: EndpointKind,
    /// Auto-detected provider type from the URL (e.g. "ollama", "anthropic", "groq").
    #[serde(default)]
    pub provider_type: String,
}

/// Whether an endpoint is a local server, cloud API, or proxy aggregator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    /// Auto-detect from URL (default).
    #[default]
    Auto,
    /// Local inference server (Ollama, llama.cpp, LM Studio, vLLM).
    Local,
    /// Cloud API provider (OpenAI, Anthropic, Google, Groq, etc.).
    Api,
    /// Proxy or aggregator (OpenRouter, LiteLLM, etc.).
    Proxy,
}

impl EndpointKind {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Local => "Local",
            Self::Api => "API",
            Self::Proxy => "Proxy",
        }
    }

    pub fn from_display_name(name: &str) -> Option<Self> {
        match name {
            "Auto" => Some(Self::Auto),
            "Local" => Some(Self::Local),
            "API" => Some(Self::Api),
            "Proxy" => Some(Self::Proxy),
            _ => None,
        }
    }

    /// All variants for display in a dropdown.
    pub fn all() -> &'static [Self] {
        &[Self::Auto, Self::Local, Self::Api, Self::Proxy]
    }
}

/// Detect the provider type from a base URL, mirroring Odysseus's `_detect_provider`.
///
/// Matches on hostname rather than substring so look-alike domains are not
/// misclassified. Unknown hosts fall back to `"openai"` (the generic
/// OpenAI-compatible default used by most local and third-party servers).
pub fn detect_provider(url: &str) -> &'static str {
    let host = extract_host(url);
    if is_ollama_native_url(url) {
        return "ollama";
    }
    if host_matches(&host, "anthropic.com") {
        return "anthropic";
    }
    // OpenCode: path-based detection (opencode.ai/zen/go vs opencode.ai/zen)
    if host_matches(&host, "opencode.ai") {
        let path = extract_path(url);
        if path.starts_with("/zen/go") {
            return "opencode-go";
        }
        return "opencode-zen";
    }
    if host_matches(&host, "openrouter.ai") {
        return "openrouter";
    }
    if host_matches(&host, "groq.com") {
        return "groq";
    }
    if host_matches(&host, "nvidia.com") {
        return "nvidia";
    }
    if host_matches(&host, "moonshot.ai") || host_matches(&host, "moonshot.cn") {
        return "moonshot";
    }
    if host_matches(&host, "cerebras.ai") {
        return "cerebras";
    }
    if host_matches(&host, "mistral.ai") {
        return "mistral";
    }
    if host_matches(&host, "googleapis.com")
        || host_matches(&host, "generativelanguage.googleapis.com")
    {
        return "google";
    }
    if host_matches(&host, "together.xyz") || host_matches(&host, "together.ai") {
        return "together";
    }
    if host_matches(&host, "fireworks.ai") {
        return "fireworks";
    }
    if host_matches(&host, "deepseek.com") {
        return "deepseek";
    }
    if host_matches(&host, "x.ai") {
        return "xai";
    }
    if host_matches(&host, "z.ai") {
        return "zai";
    }
    if host_matches(&host, "openai.com") {
        return "openai";
    }
    // Default: treat as generic OpenAI-compatible (covers LM Studio, vLLM,
    // llama.cpp, text-generation-webui, and any unknown third-party server).
    "openai"
}

/// Infer the best [`EndpointKind`] for a URL when the user leaves it as `Auto`.
pub fn classify_endpoint_kind(url: &str) -> EndpointKind {
    let host = extract_host(url);
    // Loopback / private addresses → local.
    if is_local_host(&host) {
        return EndpointKind::Local;
    }
    // Known aggregator proxies.
    if host_matches(&host, "openrouter.ai") {
        return EndpointKind::Proxy;
    }
    // Everything else with a public hostname is a cloud API.
    if !host.is_empty() {
        return EndpointKind::Api;
    }
    EndpointKind::Auto
}

/// Infer the best [`CustomEndpointSchema`] from the detected provider.
pub fn schema_for_provider(provider: &str) -> CustomEndpointSchema {
    match provider {
        "anthropic" => CustomEndpointSchema::AnthropicMessages,
        _ => CustomEndpointSchema::OpenaiChatCompletions,
    }
}

fn extract_host(url: &str) -> String {
    // Minimal URL host extraction without pulling in a full URL crate dependency.
    let stripped = url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let host = stripped
        .split('/')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("");
    host.to_lowercase()
}

fn extract_path(url: &str) -> String {
    let stripped = url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    // Skip the host:port part, return everything after it as the path.
    let after_host = stripped.splitn(2, '/').nth(1).unwrap_or("");
    format!("/{after_host}").to_lowercase()
}

fn host_matches(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

fn is_local_host(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "0.0.0.0" | "::1")
        || host.starts_with("192.168.")
        || host.starts_with("10.")
        || host.starts_with("172.")
}

fn is_ollama_native_url(url: &str) -> bool {
    let host = extract_host(url);
    // Ollama Cloud.
    if host_matches(&host, "ollama.com") {
        return true;
    }
    // Local Ollama: loopback host with default port 11434 OR native /api path.
    if !is_local_host(&host) {
        return false;
    }
    let path = url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .splitn(2, '/')
        .nth(1)
        .unwrap_or("");
    // If the path starts with /v1, it's using the OpenAI-compatible surface.
    if path.starts_with("v1") {
        return false;
    }
    // Native Ollama port or /api path.
    url.contains(":11434") || path.starts_with("api")
}

/// Fetch the list of available model IDs from a `/v1/models` endpoint.
///
/// Works with any OpenAI-compatible server (Ollama /v1, LM Studio, vLLM,
/// Groq, Together, etc.) and with Ollama's native `/api/tags` endpoint.
/// Returns an empty `Vec` on any error (network, auth, parse).
#[cfg(not(target_family = "wasm"))]
pub async fn fetch_models_from_endpoint(base_url: &str, api_key: &str) -> Vec<String> {
    use std::time::Duration;

    let url = models_url_for_endpoint(base_url);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut req = client.get(&url);
    if !api_key.trim().is_empty() {
        req = req.bearer_auth(api_key.trim());
    }

    let response = match req.send().await {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    if !response.status().is_success() {
        return Vec::new();
    }

    let body: serde_json::Value = match response.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    // OpenAI format: { "data": [{"id": "model-name"}, ...] }
    if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
        let mut models: Vec<String> = data
            .iter()
            .filter_map(|m| m.get("id").and_then(|id| id.as_str()))
            .map(|s| s.to_string())
            .collect();
        models.sort();
        return models;
    }

    // Ollama native format: { "models": [{"name": "model:tag"}, ...] }
    if let Some(models) = body.get("models").and_then(|m| m.as_array()) {
        let mut ids: Vec<String> = models
            .iter()
            .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
            .map(|s| s.to_string())
            .collect();
        ids.sort();
        return ids;
    }

    Vec::new()
}

/// Returns the models-list URL for a given base URL.
fn models_url_for_endpoint(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');

    // Ollama native API uses /api/tags; its /v1 OpenAI-compat surface uses /v1/models.
    if is_ollama_native_url(base_url) && !base.ends_with("/v1") {
        return format!("{base}/api/tags");
    }

    // If the base URL already ends with /v1, append /models directly.
    if base.ends_with("/v1") {
        return format!("{base}/models");
    }

    // For Anthropic, Google etc. that don't have a standard /v1/models,
    // still try /v1/models (harmless to try; callers ignore errors).
    format!("{base}/v1/models")
}

/// The request/response protocol used by a custom inference endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CustomEndpointSchema {
    /// OpenAI Chat Completions, retained as the legacy/default protocol.
    #[default]
    OpenaiChatCompletions,
    /// OpenAI Responses.
    OpenaiResponses,
    /// Anthropic Messages.
    AnthropicMessages,
}

impl CustomEndpointSchema {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::OpenaiChatCompletions => "OpenAI Chat Completions",
            Self::OpenaiResponses => "OpenAI Responses",
            Self::AnthropicMessages => "Anthropic Messages",
        }
    }

    pub fn from_display_name(name: &str) -> Option<Self> {
        match name {
            "OpenAI Chat Completions" => Some(Self::OpenaiChatCompletions),
            "OpenAI Responses" => Some(Self::OpenaiResponses),
            "Anthropic Messages" => Some(Self::AnthropicMessages),
            _ => None,
        }
    }
    fn to_proto(self) -> api::request::settings::custom_model_providers::CustomEndpointSchema {
        match self {
            Self::OpenaiChatCompletions => {
                api::request::settings::custom_model_providers::CustomEndpointSchema::OpenaiChatCompletions
            }
            Self::OpenaiResponses => {
                api::request::settings::custom_model_providers::CustomEndpointSchema::OpenaiResponses
            }
            Self::AnthropicMessages => {
                api::request::settings::custom_model_providers::CustomEndpointSchema::AnthropicMessages
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomEndpointModel {
    pub name: String,
    pub alias: Option<String>,
    /// Stable identifier used as `ModelConfig.{base,coding,cli_agent,computer_use_agent}` and
    /// as the `CustomModelProviders.providers[*].models[*].config_key` on the request wire.
    /// Generated as a UUIDv4 at model creation.
    pub config_key: String,
}

impl CustomEndpointModel {
    /// Picker label: prefer the user-provided alias; fall back to the raw model name
    /// so a row is never blank.
    pub fn display_label(&self) -> &str {
        match self.alias.as_deref() {
            Some(alias) if !alias.trim().is_empty() => alias,
            _ => &self.name,
        }
    }
}

impl ApiKeys {
    pub fn has_any_key(&self) -> bool {
        self.openai.is_some()
            || self.anthropic.is_some()
            || self.google.is_some()
            || self.open_router.is_some()
            || self
                .custom_endpoints
                .iter()
                .any(|endpoint| !endpoint.api_key.trim().is_empty())
    }

    /// Number of single-provider API keys currently configured (OpenAI,
    /// Anthropic, Google, OpenRouter). Custom endpoints are counted separately
    /// via `custom_endpoints`.
    pub fn provider_key_count(&self) -> usize {
        [
            &self.openai,
            &self.anthropic,
            &self.google,
            &self.open_router,
        ]
        .into_iter()
        .filter(|key| key.as_deref().is_some_and(|v| !v.trim().is_empty()))
        .count()
    }
}

/// OAuth tokens for a connected xAI / Grok subscription (e.g. SuperGrok).
///
/// Persisted to secure storage under [`GROK_SECURE_STORAGE_KEY`], separate from
/// the BYO [`ApiKeys`] blob because these are OAuth tokens with a refresh
/// lifecycle rather than a user-pasted static key. `crate::grok_subscription`
/// owns refreshing them; this module is the storage and request-injection
/// source of truth that [`ApiKeyManager::api_keys_for_request`] reads from.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GrokTokens {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Absolute time at which `access_token` expires, if the provider told us.
    #[serde(default)]
    pub expires_at: Option<SystemTime>,
    /// When the user originally connected the subscription (i.e. when the
    /// browser OAuth flow completed). Carried over across token refreshes so
    /// it keeps reflecting the initial connection, not the latest refresh;
    /// surfaced in the settings UI as "Connected on ...". `None` for tokens
    /// stored before this field existed.
    #[serde(default)]
    pub connected_at: Option<SystemTime>,
}

impl GrokTokens {
    /// Returns the access token whenever it is non-empty, regardless of
    /// expiry. Possibly-expired tokens are still sent so the server stays the
    /// final authority on token validity (it rejects truly invalid tokens);
    /// `crate::grok_subscription` refreshes (nearly) expired tokens in the
    /// background.
    pub fn access_token_for_request(&self) -> Option<&str> {
        (!self.access_token.trim().is_empty()).then_some(self.access_token.as_str())
    }

    /// Returns `true` when the token is known to expire within `lead_time` and
    /// should be proactively refreshed. Tokens with an unknown expiry never
    /// report as needing a refresh (there's no expiry signal to act on).
    pub fn needs_refresh(&self, lead_time: Duration) -> bool {
        match self.expires_at {
            Some(expires_at) => expires_at <= SystemTime::now() + lead_time,
            None => false,
        }
    }

    /// Returns `true` when the token is known to be at or past its hard expiry.
    /// Unlike [`Self::needs_refresh`] there is no lead time: a token expiring
    /// soon but still valid reports `false`. Tokens with an unknown expiry are
    /// never considered expired.
    pub fn is_expired(&self) -> bool {
        self.needs_refresh(Duration::ZERO)
    }
}

/// Outcome of a Grok OAuth token refresh, delivered to each request blocked
/// waiting on it so the request can either send with the freshly refreshed
/// token or surface the failure instead of sending an expired one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrokRefreshOutcome {
    /// The token was refreshed and the new value stored.
    Refreshed,
    /// The refresh failed; the stored token is unchanged (still expired).
    Failed,
}

/// OAuth tokens for a connected ChatGPT Plus/Pro/Team subscription.
///
/// Persisted under [`CHATGPT_SECURE_STORAGE_KEY`], separate from the BYO
/// [`ApiKeys`] blob.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ChatGptTokens {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Absolute time at which `access_token` expires, if known.
    #[serde(default)]
    pub expires_at: Option<SystemTime>,
    /// When the user originally connected the subscription.
    #[serde(default)]
    pub connected_at: Option<SystemTime>,
}

impl ChatGptTokens {
    /// Returns the access token whenever it is non-empty, regardless of expiry.
    pub fn access_token_for_request(&self) -> Option<&str> {
        (!self.access_token.trim().is_empty()).then_some(self.access_token.as_str())
    }

    /// Returns `true` when the token is known to be at or past its hard expiry.
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(expires_at) => expires_at <= SystemTime::now(),
            None => false,
        }
    }
}

/// Outcome of a ChatGPT OAuth token refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatGptRefreshOutcome {
    Refreshed,
    #[allow(dead_code)]
    Failed,
}

/// Controls how AWS credentials are refreshed by [`ApiKeyManager`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AwsCredentialsRefreshStrategy {
    /// Load credentials from the local AWS credential chain (~/.aws). This is the default.
    #[default]
    LocalChain,
    /// Credentials are managed externally via OIDC/STS.
    /// The task ID is used to scope the STS AssumeRoleWithWebIdentity session.
    /// The role ARN + region are the info used to assume the IAM role via STS.
    OidcManaged {
        task_id: Option<String>,
        role_arn: String,
        region: String,
    },
}

/// A structure that manages API keys for AI providers.
pub struct ApiKeyManager {
    keys: ApiKeys,
    /// OAuth tokens for a connected xAI/Grok subscription, if any.
    grok_tokens: Option<GrokTokens>,
    #[cfg(not(target_family = "wasm"))]
    pub(crate) grok_refresh_allowed: bool,
    #[cfg(not(target_family = "wasm"))]
    pub(crate) grok_refresh_waiters: Option<Vec<oneshot::Sender<GrokRefreshOutcome>>>,
    /// OAuth tokens for a connected ChatGPT Plus/Pro/Team subscription, if any.
    chatgpt_tokens: Option<ChatGptTokens>,
    #[cfg(not(target_family = "wasm"))]
    pub(crate) chatgpt_refresh_allowed: bool,
    #[cfg(not(target_family = "wasm"))]
    pub(crate) chatgpt_refresh_in_flight: bool,
    pub(crate) aws_credentials_state: AwsCredentialsState,
    aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy,
    pub(crate) geap_credentials_state: GeapCredentialsState,
    secure_storage_write_version: u64,
    grok_secure_storage_write_version: u64,
    chatgpt_secure_storage_write_version: u64,
}

pub struct CustomEndpointParams {
    pub name: String,
    pub url: String,
    pub api_key: String,
    pub models: Vec<(String, Option<String>, Option<String>)>,
    pub schema: CustomEndpointSchema,
    pub endpoint_kind: EndpointKind,
    pub provider_type: String,
}

impl ApiKeyManager {
    pub fn new(ctx: &mut ModelContext<Self>) -> Self {
        let keys = Self::load_keys_from_secure_storage(ctx);
        let grok_tokens = Self::load_grok_tokens_from_secure_storage(ctx);
        let chatgpt_tokens = Self::load_chatgpt_tokens_from_secure_storage(ctx);
        Self {
            keys,
            grok_tokens,
            #[cfg(not(target_family = "wasm"))]
            grok_refresh_allowed: false,
            #[cfg(not(target_family = "wasm"))]
            grok_refresh_waiters: None,
            chatgpt_tokens,
            #[cfg(not(target_family = "wasm"))]
            chatgpt_refresh_allowed: false,
            #[cfg(not(target_family = "wasm"))]
            chatgpt_refresh_in_flight: false,
            aws_credentials_state: AwsCredentialsState::Missing,
            aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy::default(),
            geap_credentials_state: GeapCredentialsState::Missing,
            secure_storage_write_version: 0,
            grok_secure_storage_write_version: 0,
            chatgpt_secure_storage_write_version: 0,
        }
    }

    pub fn keys(&self) -> &ApiKeys {
        &self.keys
    }

    /// The currently stored xAI/Grok OAuth tokens, if the user has connected a
    /// Grok subscription.
    pub fn grok_tokens(&self) -> Option<&GrokTokens> {
        self.grok_tokens.as_ref()
    }

    /// Returns `true` when a Grok subscription is connected with a usable OAuth
    /// access token.
    pub fn has_grok_subscription(&self) -> bool {
        self.grok_tokens
            .as_ref()
            .and_then(GrokTokens::access_token_for_request)
            .is_some()
    }

    /// The currently stored ChatGPT OAuth tokens, if the user has connected a
    /// ChatGPT Plus/Pro/Team subscription.
    pub fn chatgpt_tokens(&self) -> Option<&ChatGptTokens> {
        self.chatgpt_tokens.as_ref()
    }

    /// Returns `true` when a ChatGPT subscription is connected with a usable
    /// OAuth access token.
    pub fn has_chatgpt_subscription(&self) -> bool {
        self.chatgpt_tokens
            .as_ref()
            .and_then(ChatGptTokens::access_token_for_request)
            .is_some()
    }

    /// Stores (or clears) the ChatGPT OAuth tokens and persists them.
    pub fn set_chatgpt_tokens(
        &mut self,
        tokens: Option<ChatGptTokens>,
        ctx: &mut ModelContext<Self>,
    ) {
        if self.chatgpt_tokens == tokens {
            return;
        }
        self.chatgpt_tokens = tokens;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_chatgpt_tokens_to_secure_storage(ctx);
    }

    /// Returns `true` when the user has any usable BYO credential: a pasted
    /// provider or custom-endpoint key, or a connected Grok subscription.
    pub fn has_any_key(&self) -> bool {
        self.keys.has_any_key() || self.has_grok_subscription() || self.has_chatgpt_subscription()
    }

    /// Stores (or clears, with `None`) the xAI/Grok OAuth tokens and persists
    /// them to secure storage. No-op when the value is unchanged so we don't
    /// emit spurious events or schedule redundant keychain writes.
    pub fn set_grok_tokens(&mut self, tokens: Option<GrokTokens>, ctx: &mut ModelContext<Self>) {
        if self.grok_tokens == tokens {
            return;
        }
        self.grok_tokens = tokens;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_grok_tokens_to_secure_storage(ctx);
    }

    pub fn set_google_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.google = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_anthropic_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.anthropic = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_openai_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.openai = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_open_router_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.open_router = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn add_custom_endpoint(
        &mut self,
        params: CustomEndpointParams,
        ctx: &mut ModelContext<Self>,
    ) {
        let CustomEndpointParams {
            name,
            url,
            api_key,
            models,
            schema,
            endpoint_kind,
            provider_type,
        } = params;
        self.keys.custom_endpoints.push(CustomEndpoint {
            name,
            url,
            api_key,
            schema,
            endpoint_kind,
            provider_type,
            models: models
                .into_iter()
                .map(|(name, alias, config_key)| CustomEndpointModel {
                    name,
                    alias,
                    config_key: config_key
                        .filter(|k| !k.is_empty())
                        .unwrap_or_else(|| Uuid::new_v4().to_string()),
                })
                .collect(),
        });
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn save_custom_endpoint(
        &mut self,
        index: usize,
        params: CustomEndpointParams,
        ctx: &mut ModelContext<Self>,
    ) {
        if index >= self.keys.custom_endpoints.len() {
            return;
        }
        let CustomEndpointParams {
            name,
            url,
            api_key,
            models,
            schema,
            endpoint_kind,
            provider_type,
        } = params;
        self.keys.custom_endpoints[index] = CustomEndpoint {
            name,
            url,
            api_key,
            schema,
            endpoint_kind,
            provider_type,
            models: models
                .into_iter()
                .map(|(name, alias, config_key)| CustomEndpointModel {
                    name,
                    alias,
                    config_key: config_key
                        .filter(|k| !k.is_empty())
                        .unwrap_or_else(|| Uuid::new_v4().to_string()),
                })
                .collect(),
        };
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn remove_custom_endpoint(&mut self, index: usize, ctx: &mut ModelContext<Self>) {
        if index >= self.keys.custom_endpoints.len() {
            return;
        }
        self.keys.custom_endpoints.remove(index);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn clear_custom_endpoints(&mut self, ctx: &mut ModelContext<Self>) {
        if self.keys.custom_endpoints.is_empty() {
            return;
        }
        self.keys.custom_endpoints.clear();
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_aws_credentials_state(
        &mut self,
        state: AwsCredentialsState,
        ctx: &mut ModelContext<Self>,
    ) {
        self.aws_credentials_state = state;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
    }

    pub fn aws_credentials_state(&self) -> &AwsCredentialsState {
        &self.aws_credentials_state
    }

    pub fn set_geap_credentials_state(
        &mut self,
        state: GeapCredentialsState,
        ctx: &mut ModelContext<Self>,
    ) {
        if self.geap_credentials_state == state {
            return;
        }
        self.geap_credentials_state = state;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
    }

    pub fn geap_credentials_state(&self) -> &GeapCredentialsState {
        &self.geap_credentials_state
    }

    pub fn aws_credentials_refresh_strategy(&self) -> AwsCredentialsRefreshStrategy {
        self.aws_credentials_refresh_strategy.clone()
    }

    pub fn set_aws_credentials_refresh_strategy(
        &mut self,
        strategy: AwsCredentialsRefreshStrategy,
    ) {
        self.aws_credentials_refresh_strategy = strategy;
    }

    /// Builds the `CustomModelProviders` registry that ships with every agent request.
    ///
    /// Emits one [`CustomModelProvider`] per configured [`CustomEndpoint`], each populated with
    /// all of its [`CustomEndpointModel`]s. The per-model `config_key` is what the server uses
    /// to map a `ModelConfig.{base,coding,cli_agent,computer_use_agent}` selection back to a
    /// user-provided endpoint, so it MUST be the same UUID we store locally.
    ///
    /// Returns `None` when custom models should not be included or no endpoint has both a
    /// non-empty URL and API key.
    pub fn custom_model_providers_for_request(
        &self,
        include_custom_models: bool,
    ) -> Option<api::request::settings::CustomModelProviders> {
        let has_chatgpt_subscription = self.keys.custom_endpoints.iter().any(|endpoint| {
            crate::chatgpt_subscription::oauth::is_chatgpt_subscription_base(&endpoint.url)
        });
        if !include_custom_models && !has_chatgpt_subscription {
            return None;
        }

        let providers: Vec<_> = self
            .keys
            .custom_endpoints
            .iter()
            // ChatGPT subscription is a first-party OAuth integration and does
            // not depend on the workspace BYO/custom-inference entitlement.
            .filter(|endpoint| {
                include_custom_models
                    || crate::chatgpt_subscription::oauth::is_chatgpt_subscription_base(
                        &endpoint.url,
                    )
            })
            .filter(|endpoint| !endpoint.url.trim().is_empty() && !endpoint.api_key.is_empty())
            // Only send HTTPS endpoints to the server — HTTP localhost endpoints
            // are called directly by the client (server rejects plain HTTP URLs).
            .filter(|endpoint| {
                let url = endpoint.url.trim();
                url.starts_with("https://") || !url.starts_with("http://")
            })
            .map(|endpoint| {
                let api_key = if crate::chatgpt_subscription::oauth::is_chatgpt_subscription_base(
                    &endpoint.url,
                ) {
                    self.chatgpt_tokens
                        .as_ref()
                        .and_then(ChatGptTokens::access_token_for_request)
                        .unwrap_or(&endpoint.api_key)
                        .to_owned()
                } else {
                    endpoint.api_key.clone()
                };
                api::request::settings::custom_model_providers::CustomModelProvider {
                    base_url: endpoint.url.clone(),
                    api_key,
                    schema: endpoint.schema.to_proto() as i32,
                    models: endpoint
                        .models
                        .iter()
                        .filter(|m| !m.name.trim().is_empty() && !m.config_key.is_empty())
                        .map(
                            |m| api::request::settings::custom_model_providers::CustomModel {
                                slug: m.name.clone(),
                                config_key: m.config_key.clone(),
                            },
                        )
                        .collect(),
                }
            })
            .filter(|provider| !provider.models.is_empty())
            .collect();

        if providers.is_empty() {
            None
        } else {
            Some(api::request::settings::CustomModelProviders { providers })
        }
    }

    /// Returns whether a model config key belongs to the ChatGPT subscription
    /// endpoint. This is used by the client to keep local provider task state
    /// separate from Warp-server conversations.
    pub fn is_chatgpt_subscription_model(&self, model: &str) -> bool {
        self.keys.custom_endpoints.iter().any(|endpoint| {
            crate::chatgpt_subscription::oauth::is_chatgpt_subscription_base(&endpoint.url)
                && endpoint.models.iter().any(|m| m.config_key == model)
        })
    }

    /// Returns whether a model is handled by the client-side direct-provider
    /// bridge rather than Warp's cloud agent server.
    pub fn is_direct_custom_model(&self, model: &str) -> bool {
        self.keys.custom_endpoints.iter().any(|endpoint| {
            let url = endpoint.url.trim();
            let direct = url.starts_with("http://127.")
                || url.starts_with("http://localhost")
                || url.starts_with("http://0.0.0.0")
                || url.starts_with("https://chatgpt.com/backend-api/codex")
                || url.starts_with("https://opencode.ai/zen/");
            direct && endpoint.models.iter().any(|m| m.config_key == model)
        })
    }

    pub fn api_keys_for_request(
        &self,
        include_byo_keys: bool,
        include_aws_bedrock_credentials: bool,
        geap_binding: Option<GeapMintBinding>,
    ) -> Option<api::request::settings::ApiKeys> {
        let anthropic = include_byo_keys
            .then(|| self.keys.anthropic.clone())
            .flatten()
            .unwrap_or_default();
        let openai = include_byo_keys
            .then(|| self.keys.openai.clone())
            .flatten()
            .unwrap_or_default();
        let google = include_byo_keys
            .then(|| self.keys.google.clone())
            .flatten()
            .unwrap_or_default();
        let open_router = include_byo_keys
            .then(|| self.keys.open_router.clone())
            .flatten()
            .unwrap_or_default();

        // The connected Grok subscription's OAuth access token is user-provided
        // auth, just like a pasted BYO API key, so it respects the same BYO
        // policy gate: when BYO keys are disabled (e.g. by workspace policy),
        // the token must not be sent. Possibly-expired tokens ARE sent — the
        // server is the authority on validity.
        let grok_oauth_access_token = include_byo_keys
            .then(|| {
                self.grok_tokens
                    .as_ref()
                    .and_then(GrokTokens::access_token_for_request)
                    .map(str::to_owned)
            })
            .flatten()
            .unwrap_or_default();

        // Also include credentials when running with OIDC-managed Bedrock inference, regardless
        // of the per-user setting flag (which only applies to the local credential chain path).
        let include_aws = include_aws_bedrock_credentials
            || matches!(
                self.aws_credentials_refresh_strategy,
                AwsCredentialsRefreshStrategy::OidcManaged { .. }
            );
        let aws_credentials = include_aws
            .then(|| match self.aws_credentials_state {
                AwsCredentialsState::Loaded {
                    ref credentials, ..
                } => Some(credentials.clone().into()),
                _ => None,
            })
            .flatten();

        // Gemini Enterprise (GEAP) credentials attach only when the caller's
        // gate is on AND the stored token was minted for that same
        // (user, audience, SA) binding.
        let google_cloud_credentials: Option<
            api::request::settings::api_keys::GoogleCloudCredentials,
        > = geap_binding
            .as_ref()
            .and_then(|binding| match self.geap_credentials_state {
                GeapCredentialsState::Loaded {
                    ref credentials,
                    ref minted_for,
                    ..
                } if minted_for == binding => credentials
                    .access_token_for_request()
                    .map(|_| credentials.clone().into()),
                GeapCredentialsState::Refreshing {
                    previous: Some((ref credentials, ref minted_for)),
                } if minted_for == binding => credentials
                    .access_token_for_request()
                    .map(|_| credentials.clone().into()),
                _ => None,
            });

        if anthropic.is_empty()
            && openai.is_empty()
            && google.is_empty()
            && open_router.is_empty()
            && grok_oauth_access_token.is_empty()
            && aws_credentials.is_none()
            && google_cloud_credentials.is_none()
        {
            None
        } else {
            Some(api::request::settings::ApiKeys {
                anthropic,
                openai,
                google,
                open_router,
                grok_oauth_access_token,
                allow_use_of_warp_credits: false,
                aws_credentials,
                google_cloud_credentials,
            })
        }
    }

    fn load_keys_from_secure_storage(ctx: &mut ModelContext<Self>) -> ApiKeys {
        let key_json = match ctx.secure_storage().read_value(SECURE_STORAGE_KEY) {
            Ok(json) => json,
            Err(e) => {
                if !matches!(e, secure_storage::Error::NotFound) {
                    report_error!(
                        anyhow::Error::new(e)
                            .context("Failed to read API keys from secure storage")
                    );
                }
                return ApiKeys::default();
            }
        };

        match serde_json::from_str(&key_json) {
            Ok(keys) => keys,
            Err(e) => {
                report_error!(anyhow::Error::new(e).context("Failed to deserialize API keys"));
                ApiKeys::default()
            }
        }
    }

    fn write_keys_to_secure_storage(&mut self, ctx: &mut ModelContext<Self>) {
        let json = match serde_json::to_string(&self.keys) {
            Ok(json) => json,
            Err(e) => {
                report_error!(anyhow::Error::new(e).context("Failed to serialize API keys"));
                return;
            }
        };
        self.secure_storage_write_version += 1;
        let write_version = self.secure_storage_write_version;

        // Defer the keychain write so it doesn't block the current event
        // processing. The in-memory state is already updated and events
        // already emitted, so the UI updates immediately while the
        // potentially slow platform secure-storage call runs in a
        // subsequent main-thread callback. Skip stale callbacks so older
        // writes cannot complete after and overwrite a newer payload.
        ctx.spawn(async move { json }, move |me, json, ctx| {
            if write_version != me.secure_storage_write_version {
                return;
            }
            if let Err(e) = ctx.secure_storage().write_value(SECURE_STORAGE_KEY, &json) {
                report_error!(
                    anyhow::Error::new(e).context("Failed to write API keys to secure storage")
                );
            }
        });
    }

    fn load_grok_tokens_from_secure_storage(ctx: &mut ModelContext<Self>) -> Option<GrokTokens> {
        let json = match ctx.secure_storage().read_value(GROK_SECURE_STORAGE_KEY) {
            Ok(json) => json,
            Err(e) => {
                if !matches!(e, secure_storage::Error::NotFound) {
                    report_error!(
                        anyhow::Error::new(e)
                            .context("Failed to read Grok tokens from secure storage")
                    );
                }
                return None;
            }
        };

        match serde_json::from_str(&json) {
            Ok(tokens) => Some(tokens),
            Err(e) => {
                report_error!(anyhow::Error::new(e).context("Failed to deserialize Grok tokens"));
                None
            }
        }
    }

    fn write_grok_tokens_to_secure_storage(&mut self, ctx: &mut ModelContext<Self>) {
        // `Some(json)` writes the tokens; `None` removes the stored entry (the
        // user disconnected). Serialize up front so the deferred callback only
        // touches the keychain.
        let payload = match self.grok_tokens.as_ref().map(serde_json::to_string) {
            Some(Ok(json)) => Some(json),
            Some(Err(e)) => {
                report_error!(anyhow::Error::new(e).context("Failed to serialize Grok tokens"));
                return;
            }
            None => None,
        };
        self.grok_secure_storage_write_version += 1;
        let write_version = self.grok_secure_storage_write_version;

        // Defer the keychain write/remove like `write_keys_to_secure_storage`,
        // skipping stale callbacks so an older write can't clobber a newer one.
        ctx.spawn(async move { payload }, move |me, payload, ctx| {
            if write_version != me.grok_secure_storage_write_version {
                return;
            }
            let result = match payload {
                Some(ref json) => ctx
                    .secure_storage()
                    .write_value(GROK_SECURE_STORAGE_KEY, json),
                None => ctx.secure_storage().remove_value(GROK_SECURE_STORAGE_KEY),
            };
            if let Err(e) = result
                && !matches!(e, secure_storage::Error::NotFound)
            {
                report_error!(
                    anyhow::Error::new(e)
                        .context("Failed to persist Grok tokens to secure storage")
                );
            }
        });
    }

    fn load_chatgpt_tokens_from_secure_storage(
        ctx: &mut ModelContext<Self>,
    ) -> Option<ChatGptTokens> {
        let json = match ctx.secure_storage().read_value(CHATGPT_SECURE_STORAGE_KEY) {
            Ok(json) => json,
            Err(e) => {
                if !matches!(e, secure_storage::Error::NotFound) {
                    report_error!(
                        anyhow::Error::new(e)
                            .context("Failed to read ChatGPT tokens from secure storage")
                    );
                }
                return None;
            }
        };
        match serde_json::from_str(&json) {
            Ok(tokens) => Some(tokens),
            Err(e) => {
                report_error!(
                    anyhow::Error::new(e).context("Failed to deserialize ChatGPT tokens")
                );
                None
            }
        }
    }

    fn write_chatgpt_tokens_to_secure_storage(&mut self, ctx: &mut ModelContext<Self>) {
        let payload = match self.chatgpt_tokens.as_ref().map(serde_json::to_string) {
            Some(Ok(json)) => Some(json),
            Some(Err(e)) => {
                report_error!(anyhow::Error::new(e).context("Failed to serialize ChatGPT tokens"));
                return;
            }
            None => None,
        };
        self.chatgpt_secure_storage_write_version += 1;
        let write_version = self.chatgpt_secure_storage_write_version;
        ctx.spawn(async move { payload }, move |me, payload, ctx| {
            if write_version != me.chatgpt_secure_storage_write_version {
                return;
            }
            let result = match payload {
                Some(ref json) => ctx
                    .secure_storage()
                    .write_value(CHATGPT_SECURE_STORAGE_KEY, json),
                None => ctx
                    .secure_storage()
                    .remove_value(CHATGPT_SECURE_STORAGE_KEY),
            };
            if let Err(e) = result
                && !matches!(e, secure_storage::Error::NotFound)
            {
                report_error!(
                    anyhow::Error::new(e)
                        .context("Failed to persist ChatGPT tokens to secure storage")
                );
            }
        });
    }
}

impl Entity for ApiKeyManager {
    type Event = ApiKeyManagerEvent;
}

impl SingletonEntity for ApiKeyManager {}

#[cfg(test)]
#[path = "api_keys_tests.rs"]
mod tests;
