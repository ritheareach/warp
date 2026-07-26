//! OAuth device-authorization flow for connecting a ChatGPT Plus/Pro/Team
//! subscription to Warp Agent Mode.
//!
//! Unlike xAI's PKCE flow, OpenAI uses a **device authorization grant**:
//! 1. Request a device code from OpenAI's auth server.
//! 2. Display a short user-code and verification URL to the user.
//! 3. Open the URL in the browser.
//! 4. Poll until the user completes authorization.
//! 5. Receive an access + refresh token pair.
//!
//! Requests are sent to `chatgpt.com/backend-api/codex`, not
//! `api.openai.com`; this is the same backend that ChatGPT's own Codex
//! feature uses.
//!
//! Token storage, proactive refresh scheduling, and request injection live in
//! [`crate::api_keys::ApiKeyManager`] (via `ChatGptTokens` in `api_keys.rs`)
//! and the [`super`] refresh-orchestration module.

use std::time::Duration;

use anyhow::{Context as _, bail};
use serde::{Deserialize, Serialize};

/// OpenAI OAuth client ID used by the ChatGPT desktop / Codex client.
pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// Device-code request endpoint.
const DEVICE_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";

/// Device-code poll endpoint.
const DEVICE_POLL_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";

/// Token endpoint (for refresh grant).
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// Verification URL shown to the user (where they enter the code).
pub const VERIFICATION_URL: &str = "https://auth.openai.com/codex/device";

/// Base URL for ChatGPT's Codex inference backend.
pub const CHATGPT_BACKEND_BASE: &str = "https://chatgpt.com/backend-api/codex";

/// Default polling interval (seconds) between token poll requests.
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;

/// Response from the device-code request endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_auth_id: String,
    /// The short code to show the user. Some responses use "usercode" instead.
    #[serde(alias = "usercode")]
    pub user_code: String,
    #[serde(default)]
    pub verification_uri: Option<String>,
    #[serde(default)]
    pub interval: Option<serde_json::Value>,
    #[serde(default)]
    pub expires_in: Option<u64>,
}

impl DeviceCodeResponse {
    /// The URL the user should open in their browser.
    pub fn verification_url(&self) -> &str {
        self.verification_uri.as_deref().unwrap_or(VERIFICATION_URL)
    }

    /// Seconds to wait between poll requests.
    pub fn poll_interval(&self) -> Duration {
        let secs = self
            .interval
            .as_ref()
            .and_then(|v| match v {
                serde_json::Value::Number(n) => n.as_u64(),
                serde_json::Value::String(s) => s.parse::<u64>().ok(),
                _ => None,
            })
            .unwrap_or(DEFAULT_POLL_INTERVAL_SECS);
        Duration::from_secs(secs)
    }
}

/// Response from the device-code poll endpoint (authorization pending → success).
#[derive(Debug, Clone, Deserialize)]
pub struct CodeSuccessResponse {
    pub authorization_code: String,
    pub code_verifier: String,
    #[serde(default)]
    pub code_challenge: Option<String>,
}

/// Token response from the token exchange endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<i64>,
}

/// Error variants returned during the device-auth poll.
#[derive(Debug)]
pub enum PollOutcome {
    /// User has not yet authorized — keep polling.
    Pending,
    /// Authorization complete; authorization_code + verifier are ready to exchange.
    Authorized(CodeSuccessResponse),
    /// Terminal error (expired, denied, etc.).
    Error(anyhow::Error),
}

/// Step 1: Request a device code from OpenAI.
pub async fn request_device_code() -> anyhow::Result<DeviceCodeResponse> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .context("failed to build HTTP client")?;

    let response = client
        .post(DEVICE_CODE_URL)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "client_id": CLIENT_ID }))
        .send()
        .await
        .context("failed to send device-code request to OpenAI")?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        if status.as_u16() == 404 {
            bail!(
                "ChatGPT device code login is not enabled. \
                 Go to https://chatgpt.com/settings/security and enable \
                 'Device code authorization' first."
            );
        }
        bail!("ChatGPT device-code request failed ({status}): {body}");
    }

    let data: DeviceCodeResponse = response
        .json()
        .await
        .context("failed to parse ChatGPT device-code response")?;

    if data.device_auth_id.is_empty() || data.user_code.is_empty() {
        bail!(
            "ChatGPT device-code response was missing required fields (device_auth_id or user_code)"
        );
    }

    Ok(data)
}

/// Step 2: Poll until the user authorizes or the code expires.
/// Returns `PollOutcome::Pending` when the user hasn't acted yet,
/// or `PollOutcome::Authorized(code_response)` when approved.
pub async fn poll_device_auth(device_auth_id: &str, user_code: &str) -> PollOutcome {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(e) => return PollOutcome::Error(anyhow::Error::new(e)),
    };

    let result = client
        .post(DEVICE_POLL_URL)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "device_auth_id": device_auth_id,
            "user_code": user_code,
        }))
        .send()
        .await;

    let response = match result {
        Ok(r) => r,
        Err(e) => {
            return PollOutcome::Error(
                anyhow::Error::new(e).context("ChatGPT device-auth poll request failed"),
            );
        }
    };

    let status = response.status();

    // 403/404 means "authorization_pending" — not yet approved.
    if status.as_u16() == 403 || status.as_u16() == 404 {
        return PollOutcome::Pending;
    }

    let body = match response.text().await {
        Ok(b) => b,
        Err(e) => {
            return PollOutcome::Error(
                anyhow::Error::new(e).context("failed to read ChatGPT poll response body"),
            );
        }
    };

    if !status.is_success() {
        if body.contains("authorization_pending") {
            return PollOutcome::Pending;
        }
        return PollOutcome::Error(anyhow::anyhow!(
            "ChatGPT device-auth poll failed ({status}): {body}"
        ));
    }

    // Success — response contains authorization_code + code_verifier.
    match serde_json::from_str::<CodeSuccessResponse>(&body) {
        Ok(code_resp) if !code_resp.authorization_code.is_empty() => {
            PollOutcome::Authorized(code_resp)
        }
        Ok(_) => PollOutcome::Pending,
        Err(e) => PollOutcome::Error(
            anyhow::Error::new(e).context(format!("failed to parse ChatGPT poll response: {body}")),
        ),
    }
}

/// Step 3: Exchange the authorization code for access + refresh tokens.
pub async fn exchange_code_for_tokens(
    authorization_code: &str,
    code_verifier: &str,
) -> anyhow::Result<TokenResponse> {
    let redirect_uri = "https://auth.openai.com/deviceauth/callback";

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .context("failed to build HTTP client")?;

    let form = [
        ("grant_type", "authorization_code"),
        ("client_id", CLIENT_ID),
        ("code", authorization_code),
        ("code_verifier", code_verifier),
        ("redirect_uri", redirect_uri),
    ];

    let response = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&form)
        .send()
        .await
        .context("ChatGPT token exchange request failed")?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        bail!("ChatGPT token exchange failed ({status}): {body}");
    }

    let tokens: TokenResponse = response
        .json()
        .await
        .context("failed to parse ChatGPT token exchange response")?;

    if tokens.access_token.is_empty() {
        bail!("ChatGPT token exchange did not return an access token");
    }

    Ok(tokens)
}

/// Refresh an existing access token using the stored refresh token.
pub async fn refresh_access_token(refresh_token: &str) -> anyhow::Result<TokenResponse> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .context("failed to build HTTP client")?;

    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", CLIENT_ID),
    ];

    let response = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&form)
        .send()
        .await
        .context("ChatGPT token refresh request failed")?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        bail!("ChatGPT token refresh failed ({status}): {body}");
    }

    let tokens: TokenResponse = response
        .json()
        .await
        .context("failed to parse ChatGPT refresh token response")?;

    if tokens.access_token.is_empty() {
        bail!("ChatGPT token refresh did not return an access token");
    }

    Ok(tokens)
}

/// Fetch available ChatGPT Codex models using the access token.
/// Returns an empty vec on any error (caller should surface to user).
pub async fn fetch_available_models(access_token: &str) -> Vec<String> {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let result = client
        .get("https://chatgpt.com/backend-api/codex/models?client_version=1.0.0")
        .header("Authorization", format!("Bearer {access_token}"))
        .header("Origin", "https://chatgpt.com")
        .header("Referer", "https://chatgpt.com/codex")
        .send()
        .await;

    let response = match result {
        Ok(r) if r.status().is_success() => r,
        _ => return Vec::new(),
    };

    let data: serde_json::Value = match response.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let entries = match data.get("models").and_then(|m| m.as_array()) {
        Some(e) => e.clone(),
        None => return Vec::new(),
    };

    let mut models: Vec<(i64, String)> = entries
        .iter()
        .filter_map(|item| {
            let slug = item.get("slug")?.as_str()?.trim().to_string();
            if slug.is_empty() {
                return None;
            }
            // Skip hidden models.
            let visibility = item
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if visibility.eq_ignore_ascii_case("hide") || visibility.eq_ignore_ascii_case("hidden")
            {
                return None;
            }
            let priority = item
                .get("priority")
                .and_then(|p| p.as_i64())
                .unwrap_or(10_000);
            Some((priority, slug))
        })
        .collect();

    models.sort_by_key(|(priority, slug)| (*priority, slug.clone()));

    let mut seen = std::collections::HashSet::new();
    models
        .into_iter()
        .filter_map(|(_, slug)| seen.insert(slug.clone()).then_some(slug))
        .collect()
}

/// Returns true if the given URL is the ChatGPT Codex backend.
pub fn is_chatgpt_subscription_base(url: &str) -> bool {
    let url = url.trim();
    let stripped = url
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let host_and_path = stripped.split_once('?').map(|(h, _)| h).unwrap_or(stripped);
    let (host, path) = host_and_path
        .split_once('/')
        .map(|(h, p)| (h, format!("/{p}")))
        .unwrap_or((host_and_path, String::new()));
    host.to_lowercase() == "chatgpt.com"
        && (path == "/backend-api/codex" || path.starts_with("/backend-api/codex/"))
}
