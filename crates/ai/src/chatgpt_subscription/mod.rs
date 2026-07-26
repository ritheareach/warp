//! Refresh orchestration for a connected ChatGPT Plus/Pro/Team subscription.
//!
//! Token storage and request injection live in [`ApiKeyManager`] (via
//! [`ChatGptTokens`] in `api_keys.rs`). This module owns proactive refresh
//! scheduling — converting a [`TokenResponse`] into stored [`ChatGptTokens`]
//! and rescheduling refreshes before each token expires.

pub mod oauth;

use std::time::{Duration, SystemTime};

use warp_errors::report_error;
use warpui_core::ModelContext;
use warpui_core::r#async::Timer;

use self::oauth::TokenResponse;
use crate::api_keys::{ApiKeyManager, ChatGptTokens};

/// Refresh this far before the hard expiry to avoid racing expiration at
/// request time.
const REFRESH_LEAD_TIME: Duration = Duration::from_secs(5 * 60);

/// Builds [`ChatGptTokens`] from a token-endpoint [`TokenResponse`].
pub fn chatgpt_tokens_from_response(
    response: TokenResponse,
    previous: Option<&ChatGptTokens>,
) -> ChatGptTokens {
    let expires_at = response
        .expires_in
        .and_then(|secs| u64::try_from(secs).ok())
        .and_then(|secs| SystemTime::now().checked_add(Duration::from_secs(secs)));
    ChatGptTokens {
        access_token: response.access_token,
        refresh_token: response
            .refresh_token
            .or_else(|| previous.and_then(|t| t.refresh_token.clone())),
        expires_at,
        connected_at: previous
            .and_then(|t| t.connected_at)
            .or_else(|| Some(SystemTime::now())),
    }
}

impl ApiKeyManager {
    /// Persists freshly obtained tokens and schedules the next proactive
    /// refresh.
    pub fn store_chatgpt_tokens(&mut self, response: TokenResponse, ctx: &mut ModelContext<Self>) {
        apply_chatgpt_tokens(self, response, ctx);
    }

    /// Updates whether background refresh of the stored ChatGPT tokens is
    /// allowed. Mirrors the BYO API key policy gate.
    pub fn set_chatgpt_refresh_allowed(&mut self, allowed: bool, ctx: &mut ModelContext<Self>) {
        if self.chatgpt_refresh_allowed == allowed {
            return;
        }
        self.chatgpt_refresh_allowed = allowed;
        if allowed {
            schedule_chatgpt_token_refresh(self, ctx);
        }
    }
}

fn apply_chatgpt_tokens(
    manager: &mut ApiKeyManager,
    response: TokenResponse,
    ctx: &mut ModelContext<ApiKeyManager>,
) {
    let tokens = chatgpt_tokens_from_response(response, manager.chatgpt_tokens());
    manager.set_chatgpt_tokens(Some(tokens), ctx);
    schedule_chatgpt_token_refresh(manager, ctx);
}

fn schedule_chatgpt_token_refresh(
    manager: &mut ApiKeyManager,
    ctx: &mut ModelContext<ApiKeyManager>,
) {
    if !manager.chatgpt_refresh_allowed {
        return;
    }
    let Some(tokens) = manager.chatgpt_tokens() else {
        return;
    };
    let Some(refresh_token) = tokens.refresh_token.clone() else {
        return;
    };
    let Some(expires_at) = tokens.expires_at else {
        return;
    };

    let now = SystemTime::now();
    let fire_at = expires_at.checked_sub(REFRESH_LEAD_TIME).unwrap_or(now);
    let delay = fire_at.duration_since(now).unwrap_or(Duration::ZERO);

    ctx.spawn(
        async move {
            Timer::after(delay).await;
        },
        move |manager, _output, ctx| {
            if !manager.chatgpt_refresh_allowed {
                return;
            }
            let still_current = manager
                .chatgpt_tokens()
                .and_then(|t| t.refresh_token.as_deref())
                == Some(refresh_token.as_str());
            if still_current {
                spawn_chatgpt_refresh(manager, refresh_token, ctx);
            }
        },
    );
}

fn spawn_chatgpt_refresh(
    manager: &mut ApiKeyManager,
    refresh_token: String,
    ctx: &mut ModelContext<ApiKeyManager>,
) {
    if manager.chatgpt_refresh_in_flight {
        return;
    }
    manager.chatgpt_refresh_in_flight = true;
    ctx.spawn(
        async move { oauth::refresh_access_token(&refresh_token).await },
        move |manager, result, ctx| {
            manager.chatgpt_refresh_in_flight = false;
            match result {
                Ok(response) => {
                    log::info!(
                        "Refreshed ChatGPT OAuth token (expires_in={:?}, has_refresh={})",
                        response.expires_in,
                        response.refresh_token.is_some(),
                    );
                    apply_chatgpt_tokens(manager, response, ctx);
                }
                Err(err) => {
                    report_error!(err.context("Failed to refresh ChatGPT OAuth token"));
                }
            }
        },
    );
}
