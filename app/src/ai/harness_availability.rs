use std::collections::HashMap;
use std::time::Duration;

use instant::Instant;
use serde::{Deserialize, Serialize};
use warp_cli::agent::Harness;
use warp_core::features::FeatureFlag;
use warp_core::user_preferences::GetUserPreferences;
use warp_errors::report_error;
use warp_managed_secrets::client::SecretOwner;
use warp_managed_secrets::{ManagedSecretManager, ManagedSecretValue};
use warpui::{Entity, ModelContext, RequestState, SingletonEntity};

use crate::ai::harness_display;
use crate::auth::AuthStateProvider;
use crate::auth::auth_manager::{AuthManager, AuthManagerEvent};
use crate::network::{NetworkStatus, NetworkStatusEvent, NetworkStatusKind};
use crate::server::retry_strategies::{
    OUT_OF_BAND_REQUEST_RETRY_STRATEGY, is_transient_graphql_or_http_error,
};
use crate::server::server_api::ServerApiProvider;
use crate::workspaces::user_workspaces::{UserWorkspaces, UserWorkspacesEvent};

const CACHE_KEY: &str = "AvailableHarnesses";
const AUTH_SECRET_FETCH_FAILURE_COOLDOWN: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HarnessModelInfo {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_level: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HarnessAvailability {
    pub harness: Harness,
    pub display_name: String,
    pub enabled: bool,
    #[serde(default)]
    pub available_models: Vec<HarnessModelInfo>,
}

/// Default fallback used before the server responds.
/// Oz is enabled by default so the UI is usable pre-fetch; the server
/// list (which respects admin overrides) replaces this once available.
fn default_harnesses() -> Vec<HarnessAvailability> {
    vec![HarnessAvailability {
        harness: Harness::Oz,
        display_name: "Warp".to_string(),
        enabled: true,
        available_models: vec![],
    }]
}

#[derive(Debug, Clone)]
pub enum AuthSecretFetchState {
    NotFetched,
    Loading,
    Loaded(Vec<AuthSecretEntry>),
    Failed(#[allow(dead_code)] String),
}

#[derive(Debug, Clone)]
pub struct AuthSecretEntry {
    pub name: String,
    pub owner: SecretOwner,
}

pub enum HarnessAvailabilityEvent {
    Changed,
    AuthSecretsLoaded,
    /// Emitted when a lazy auth-secrets fetch fails. Subscribers should
    /// re-render so any "Loading…" placeholders can transition to an
    /// error state — without this signal the picker would otherwise be
    /// stuck on the loading placeholder until the next refetch.
    AuthSecretsFetchFailed,
    AuthSecretCreated {
        harness: Harness,
        name: String,
    },
    AuthSecretCreationFailed {
        error: String,
    },
    AuthSecretDeleted {
        harness: Harness,
        name: String,
        owner: SecretOwner,
    },
    AuthSecretDeletionFailed {
        harness: Harness,
        name: String,
        owner: SecretOwner,
        error: String,
    },
}

pub struct HarnessAvailabilityModel {
    harnesses: Vec<HarnessAvailability>,
    auth_secrets: HashMap<Harness, AuthSecretFetchState>,
    auth_secret_retry_after: HashMap<Harness, Instant>,
}

impl HarnessAvailabilityModel {
    pub fn new(ctx: &mut ModelContext<Self>) -> Self {
        let harnesses = get_cached(ctx).unwrap_or_else(default_harnesses);

        ctx.subscribe_to_model(&NetworkStatus::handle(ctx), |me, _, event, ctx| {
            if let NetworkStatusEvent::NetworkStatusChanged {
                new_status: NetworkStatusKind::Online,
            } = event
            {
                me.refresh(ctx);
            }
        });

        ctx.subscribe_to_model(&AuthManager::handle(ctx), |me, _, event, ctx| {
            if let AuthManagerEvent::AuthComplete = event {
                let cached_harnesses: Vec<Harness> = me.auth_secrets.keys().copied().collect();
                for harness in cached_harnesses {
                    me.invalidate_auth_secrets(harness);
                }
                me.refresh(ctx);
            }
        });

        ctx.subscribe_to_model(&UserWorkspaces::handle(ctx), |me, _, event, ctx| {
            if let UserWorkspacesEvent::TeamsChanged = event {
                me.refresh(ctx);
            }
        });

        let me = Self {
            harnesses,
            auth_secrets: HashMap::new(),
            auth_secret_retry_after: HashMap::new(),
        };
        me.refresh(ctx);
        me
    }

    pub fn available_harnesses(&self) -> &[HarnessAvailability] {
        &self.harnesses
    }

    pub fn display_name_for(&self, harness: Harness) -> &str {
        self.harnesses
            .iter()
            .find(|h| h.harness == harness)
            .map(|h| h.display_name.as_str())
            .unwrap_or_else(|| harness_display::display_name(harness))
    }

    /// Whether the harness selector should be shown (>1 known harness, including disabled).
    pub fn should_show_harness_selector(&self) -> bool {
        FeatureFlag::AgentHarness.is_enabled() && self.harnesses.len() > 1
    }

    /// Whether any harness is available at all (at least one enabled).
    pub fn has_any_enabled_harness(&self) -> bool {
        self.harnesses.iter().any(|h| h.enabled)
    }

    /// Whether a harness is both known and enabled.
    pub fn is_harness_enabled(&self, harness: Harness) -> bool {
        self.harnesses
            .iter()
            .any(|h| h.harness == harness && h.enabled)
    }

    pub fn models_for(&self, harness: Harness) -> Option<&[HarnessModelInfo]> {
        self.harnesses
            .iter()
            .find(|h| h.harness == harness)
            .map(|h| h.available_models.as_slice())
            .filter(|m| !m.is_empty())
    }

    pub fn auth_secrets_for(&self, harness: Harness) -> &AuthSecretFetchState {
        self.auth_secrets
            .get(&harness)
            .unwrap_or(&AuthSecretFetchState::NotFetched)
    }

    pub fn ensure_auth_secrets_fetched(&mut self, harness: Harness, ctx: &mut ModelContext<Self>) {
        match self.auth_secrets_for(harness) {
            AuthSecretFetchState::NotFetched => self.fetch_auth_secrets(harness, ctx),
            AuthSecretFetchState::Failed(_) if self.can_retry_auth_secret_fetch(harness) => {
                self.fetch_auth_secrets(harness, ctx);
            }
            AuthSecretFetchState::Failed(_)
            | AuthSecretFetchState::Loading
            | AuthSecretFetchState::Loaded(_) => {}
        }
    }

    fn fetch_auth_secrets(&mut self, harness: Harness, ctx: &mut ModelContext<Self>) {
        let Some(agent_harness) = harness_to_graphql_harness(harness) else {
            return;
        };

        if !AuthStateProvider::as_ref(ctx).get().is_logged_in() {
            return;
        }

        self.auth_secrets
            .insert(harness, AuthSecretFetchState::Loading);
        self.auth_secret_retry_after.remove(&harness);

        let api = ServerApiProvider::as_ref(ctx).get_managed_secrets_client();
        ctx.spawn_with_retry_on_error_when(
            move || {
                let api = api.clone();
                let agent_harness = agent_harness.clone();
                async move { api.list_harness_auth_secrets(agent_harness).await }
            },
            OUT_OF_BAND_REQUEST_RETRY_STRATEGY,
            is_transient_graphql_or_http_error,
            move |me,
                  result: RequestState<Vec<warp_graphql::managed_secrets::ManagedSecret>>,
                  ctx| match result {
                RequestState::RequestSucceeded(secrets) => {
                    let entries = secrets
                        .into_iter()
                        .map(|s| AuthSecretEntry {
                            owner: secret_owner_from_space(&s.owner),
                            name: s.name,
                        })
                        .collect();
                    me.auth_secrets
                        .insert(harness, AuthSecretFetchState::Loaded(entries));
                    me.auth_secret_retry_after.remove(&harness);
                    ctx.emit(HarnessAvailabilityEvent::AuthSecretsLoaded);
                }
                RequestState::RequestFailedRetryPending(e) => {
                    log::warn!("Failed to fetch harness auth secrets; retrying: {e:#}");
                }
                RequestState::RequestFailed(e) => {
                    let msg = e.to_string();
                    report_error!(e.context("Failed to fetch harness auth secrets"));
                    me.auth_secrets
                        .insert(harness, AuthSecretFetchState::Failed(msg));
                    me.auth_secret_retry_after
                        .insert(harness, Instant::now() + AUTH_SECRET_FETCH_FAILURE_COOLDOWN);
                    // Notify subscribers so they can drop any
                    // "Loading…" placeholder rendered during the
                    // in-flight fetch and surface the error state.
                    ctx.emit(HarnessAvailabilityEvent::AuthSecretsFetchFailed);
                }
            },
        );
    }

    fn can_retry_auth_secret_fetch(&self, harness: Harness) -> bool {
        self.auth_secret_retry_after
            .get(&harness)
            .map(|retry_after| Instant::now() >= *retry_after)
            .unwrap_or(true)
    }

    pub fn invalidate_auth_secrets(&mut self, harness: Harness) {
        self.auth_secrets.remove(&harness);
        self.auth_secret_retry_after.remove(&harness);
    }

    pub fn create_auth_secret(
        &mut self,
        harness: Harness,
        name: String,
        value: ManagedSecretValue,
        owner: SecretOwner,
        ctx: &mut ModelContext<Self>,
    ) {
        let manager = ManagedSecretManager::handle(ctx);
        let create_future = manager.as_ref(ctx).create_secret(owner, name, value, None);
        ctx.spawn(create_future, move |me, result, ctx| match result {
            Ok(secret) => {
                let entry = AuthSecretEntry {
                    name: secret.name.clone(),
                    owner: secret_owner_from_space(&secret.owner),
                };
                match me.auth_secrets.get_mut(&harness) {
                    Some(AuthSecretFetchState::Loaded(entries)) => {
                        entries.push(entry);
                    }
                    _ => {
                        me.auth_secrets
                            .insert(harness, AuthSecretFetchState::Loaded(vec![entry]));
                    }
                }
                ctx.emit(HarnessAvailabilityEvent::AuthSecretCreated {
                    harness,
                    name: secret.name,
                });
            }
            Err(e) => {
                let msg = e.to_string();
                report_error!(e.context("Failed to create harness auth secret"));
                ctx.emit(HarnessAvailabilityEvent::AuthSecretCreationFailed { error: msg });
            }
        });
    }

    pub fn delete_auth_secret(
        &mut self,
        harness: Harness,
        name: String,
        owner: SecretOwner,
        ctx: &mut ModelContext<Self>,
    ) {
        let manager = ManagedSecretManager::handle(ctx);
        let delete_future = manager
            .as_ref(ctx)
            .delete_secret(owner.clone(), name.clone());
        ctx.spawn(delete_future, move |me, result, ctx| match result {
            Ok(()) => {
                if let Some(AuthSecretFetchState::Loaded(entries)) =
                    me.auth_secrets.get_mut(&harness)
                {
                    remove_deleted_auth_secret_entry(entries, &name, &owner);
                }
                ctx.emit(HarnessAvailabilityEvent::AuthSecretDeleted {
                    harness,
                    name,
                    owner,
                });
            }
            Err(e) => {
                let msg = e.to_string();
                report_error!(e.context("Failed to delete harness auth secret"));
                ctx.emit(HarnessAvailabilityEvent::AuthSecretDeletionFailed {
                    harness,
                    name,
                    owner,
                    error: msg,
                });
            }
        });
    }

    pub fn refresh(&self, ctx: &mut ModelContext<Self>) {
        // The endpoint queries `user`, which requires auth.
        if !AuthStateProvider::as_ref(ctx).get().is_logged_in() {
            return;
        }

        let ai_client = ServerApiProvider::as_ref(ctx).get_ai_client();
        ctx.spawn(
            async move {
                let server_harnesses = ai_client.get_available_harnesses().await?;
                // Fetch local CLI models in parallel with the server response.
                let agy_models = fetch_agy_models().await;
                anyhow::Ok((server_harnesses, agy_models))
            },
            |me, result, ctx| match result {
                Ok((mut new_harnesses, agy_models)) => {
                    // Inject locally-available harnesses with their model lists.
                    inject_local_harnesses(&mut new_harnesses, agy_models);
                    if new_harnesses != me.harnesses {
                        me.harnesses = new_harnesses;
                        me.cache(ctx);
                        let stale: Vec<Harness> = me.auth_secrets.keys().copied().collect();
                        for harness in stale {
                            me.invalidate_auth_secrets(harness);
                        }
                        ctx.emit(HarnessAvailabilityEvent::Changed);
                    }
                }
                Err(e) => {
                    report_error!(e.context("Failed to fetch available harnesses"));
                }
            },
        );
    }

    fn cache(&self, ctx: &ModelContext<Self>) {
        if let Ok(serialized) = serde_json::to_string(&self.harnesses)
            && let Err(e) = ctx
                .private_user_preferences()
                .write_value(CACHE_KEY, serialized)
        {
            report_error!(anyhow::anyhow!(e).context("Failed to cache available harnesses"));
        }
    }
}

fn get_cached(ctx: &ModelContext<HarnessAvailabilityModel>) -> Option<Vec<HarnessAvailability>> {
    let raw = ctx
        .private_user_preferences()
        .read_value(CACHE_KEY)
        .ok()??;
    serde_json::from_str::<Vec<HarnessAvailability>>(&raw).ok()
}

fn secret_owner_from_space(space: &warp_graphql::object::Space) -> SecretOwner {
    match space.type_ {
        warp_graphql::object::SpaceType::Team => SecretOwner::Team {
            team_uid: space.uid.clone().into_inner(),
        },
        warp_graphql::object::SpaceType::User => SecretOwner::CurrentUser,
    }
}

fn remove_deleted_auth_secret_entry(
    entries: &mut Vec<AuthSecretEntry>,
    name: &str,
    owner: &SecretOwner,
) {
    entries.retain(|entry| entry.name.as_str() != name || &entry.owner != owner);
}
fn harness_to_graphql_harness(harness: Harness) -> Option<warp_graphql::ai::AgentHarness> {
    match harness {
        Harness::Oz => Some(warp_graphql::ai::AgentHarness::Oz),
        Harness::Claude => Some(warp_graphql::ai::AgentHarness::ClaudeCode),
        Harness::Gemini => Some(warp_graphql::ai::AgentHarness::Gemini),
        Harness::Codex => Some(warp_graphql::ai::AgentHarness::Codex),
        Harness::OpenCode | Harness::Agy | Harness::Unknown => None,
    }
}

/// Fetch the list of available models from the `agy models` CLI command.
/// Returns an empty vec if `agy` is not installed or the command fails.
async fn fetch_agy_models() -> Vec<HarnessModelInfo> {
    #[cfg(not(target_family = "wasm"))]
    {
        use tokio::process::Command;
        use tokio::time::timeout;

        let result = timeout(
            std::time::Duration::from_secs(10),
            Command::new("agy").arg("models").output(),
        )
        .await;

        match result {
            Ok(Ok(output)) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|line| {
                        let id = line.trim().to_string();
                        // Make a human-readable display name from the model id.
                        // e.g. "claude-sonnet-4-6" -> "Claude Sonnet 4.6"
                        let display_name = make_display_name(&id);
                        HarnessModelInfo {
                            id,
                            display_name,
                            reasoning_level: None,
                        }
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }
    #[cfg(target_family = "wasm")]
    {
        Vec::new()
    }
}

/// Convert a model id like "claude-sonnet-4-6" into "Claude Sonnet 4.6".
fn make_display_name(id: &str) -> String {
    id.split('-')
        .map(|part| {
            // Capitalize first letter if alphabetic, leave numbers as-is.
            let mut chars = part.chars();
            match chars.next() {
                Some(c) if c.is_alphabetic() => {
                    c.to_uppercase().collect::<String>() + chars.as_str()
                }
                _ => part.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Known model lists for CLI harnesses that don't support model discovery.
fn codex_known_models() -> Vec<HarnessModelInfo> {
    [
        ("o3", "o3"),
        ("o4-mini", "o4-mini"),
        ("codex-mini-latest", "Codex Mini"),
        ("gpt-4o", "GPT-4o"),
    ]
    .iter()
    .map(|(id, name)| HarnessModelInfo {
        id: id.to_string(),
        display_name: name.to_string(),
        reasoning_level: None,
    })
    .collect()
}

fn claude_known_models() -> Vec<HarnessModelInfo> {
    [
        ("claude-opus-4-5", "Claude Opus 4.5"),
        ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
        ("claude-haiku-4-5", "Claude Haiku 4.5"),
        ("claude-opus-4", "Claude Opus 4"),
        ("claude-sonnet-4", "Claude Sonnet 4"),
        ("claude-haiku-4", "Claude Haiku 4"),
        ("fable", "Claude Fable (latest)"),
        ("opus", "Claude Opus (latest)"),
        ("sonnet", "Claude Sonnet (latest)"),
    ]
    .iter()
    .map(|(id, name)| HarnessModelInfo {
        id: id.to_string(),
        display_name: name.to_string(),
        reasoning_level: None,
    })
    .collect()
}

/// Inject locally-available harnesses (Agy, Codex, Claude) into the server-provided list.
/// Harnesses already in the list get their models updated; new ones are appended.
fn inject_local_harnesses(
    harnesses: &mut Vec<HarnessAvailability>,
    agy_models: Vec<HarnessModelInfo>,
) {
    let local_entries: [(Harness, &str, Vec<HarnessModelInfo>); 3] = [
        (Harness::Agy, "Agy", agy_models),
        (Harness::Codex, "Codex (ChatGPT)", codex_known_models()),
        (Harness::Claude, "Claude Code", claude_known_models()),
    ];

    for (harness, display_name, models) in local_entries {
        if let Some(existing) = harnesses.iter_mut().find(|h| h.harness == harness) {
            // Already in the list from the server — update models if we have better info.
            if existing.available_models.is_empty() && !models.is_empty() {
                existing.available_models = models;
            }
        } else {
            // Not in the server list — add it as a locally-available harness.
            harnesses.push(HarnessAvailability {
                harness,
                display_name: display_name.to_string(),
                enabled: true,
                available_models: models,
            });
        }
    }
}

impl Entity for HarnessAvailabilityModel {
    type Event = HarnessAvailabilityEvent;
}

impl SingletonEntity for HarnessAvailabilityModel {}
