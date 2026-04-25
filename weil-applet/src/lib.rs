//! # openclaw_weil
//!
//! OpenClaw reimplemented as a Weilliptic smart-contract applet.
//!
//! ## Query / Mutate discipline
//!
//! `#[mutate]` runs on **all** pod members and must be deterministic.
//! LLM / cerebrum calls are non-deterministic, so they live exclusively in
//! `#[query]` methods (single-pod execution, no state commit).
//!
//! Typical call sequence for a conversation turn:
//! ```text
//!  1. [query]  send_message(…)             → AgentResponse { reply, task_id }
//!  2. [mutate] append_message(session_key, User,      user_text)
//!  3. [mutate] append_message(session_key, Assistant, reply)
//!  4. [mutate] store_task_record(session_key, task_id, user_text, reply)
//! ```
//!
//! ## Architecture
//!
//! ```text
//!  Caller (channel / UI)
//!        │
//!        │ ① query
//!        ▼
//!  OpenClawState  (this contract)
//!        │
//!        │ Driver / ChainWithAgents / Pipeline   [query only]
//!        ▼
//!  cerebrum::Driver ──── LLM ────────────────────────────────────────────►
//!        │
//!        │ tool calls via Runtime::call_contract
//!        ▼
//!  MCP Contract  (external, or this contract for built-in read-only tools)
//!
//!        │ ② mutate  (after query returns)
//!        ▼
//!  OpenClawState  (append_message / store_task_record / touch_cron / …)
//! ```
//!
//! ## WeilId allocation
//!
//! IDs 1-9 are reserved for cerebrum internals (Memory 1+2, Actions 1+2+3,
//! AgentRegistry 1).  This applet starts at 10.
//!
//! | WeilId | Collection |
//! |--------|-----------|
//! | 10     | `sessions`      – session_key → Vec<ConversationMessage> |
//! | 11     | `cron_jobs`     – cron_id     → CronJob |
//! | 12     | `user_memory`   – "{caller}:{key}" → String |
//! | 13     | `task_history`  – session_key → Vec<TaskRecord> |
//! | 14     | `agent_registry`– name → contract_address |

#![allow(dead_code)]

mod types;
mod tools;

use types::*;
use tools::tool_schema_json;

use serde::{Deserialize, Serialize};
use weil_macros::{WeilType, constructor, mutate, query, smart_contract};
use weil_rs::{
    collections::{map::WeilMap, WeilId},
    http::{HttpClient, HttpMethod},
    runtime::Runtime,
};
use w_cerebrum::{
    core::{chain::ChainWithAgents, task::TaskWithAgent},
    llmutils::driver::Driver,
    pipeline::{ConditionType, Pipeline},
};

// ---------------------------------------------------------------------------
// WeilId constants
// ---------------------------------------------------------------------------

const ID_SESSIONS: WeilId = WeilId(10);
const ID_CRON_JOBS: WeilId = WeilId(11);
const ID_USER_MEMORY: WeilId = WeilId(12);
const ID_TASK_HISTORY: WeilId = WeilId(13);
const ID_AGENT_REGISTRY: WeilId = WeilId(14);

// ---------------------------------------------------------------------------
// Contract state
// ---------------------------------------------------------------------------

/// Persistent state of the OpenClaw Weilliptic applet.
#[derive(Serialize, Deserialize, WeilType)]
pub struct OpenClawState {
    /// Application configuration.
    pub config: AppletConfig,

    /// Conversation transcripts keyed by session_key.
    pub sessions: WeilMap<String, Vec<ConversationMessage>>,
    /// Insertion-ordered session keys for list_sessions.
    pub session_keys: Vec<String>,

    /// Cron job definitions.
    pub cron_jobs: WeilMap<String, CronJob>,
    /// Insertion-ordered cron ids for list_crons.
    pub cron_ids: Vec<String>,

    /// Per-caller flat key-value memory.  Keys are "{caller_addr}:{user_key}".
    pub user_memory: WeilMap<String, String>,

    /// Per-session task execution history.
    pub task_history: WeilMap<String, Vec<TaskRecord>>,

    /// Named external MCP agent registry.  Key = friendly name, value = contract address.
    pub agent_registry: WeilMap<String, String>,
    /// Insertion-ordered agent names for list_agents.
    pub agent_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// Contract trait
// ---------------------------------------------------------------------------

trait OpenClaw {
    // --- lifecycle ----------------------------------------------------------
    fn new(system_prompt: Option<String>) -> Result<Self, String>
    where
        Self: Sized;

    fn configure(&mut self, config: AppletConfig) -> Result<(), String>;

    // --- agent registry (mutate — deterministic) ----------------------------
    fn register_agent(&mut self, name: String, contract_address: String) -> Result<(), String>;
    fn unregister_agent(&mut self, name: String) -> Result<(), String>;
    fn list_agents(&self) -> Result<String, String>;

    // --- core AI interaction (query — non-deterministic LLM call) -----------
    /// Call the LLM and return the reply.  Does NOT write to state.
    /// After the caller receives the response it should call:
    ///   `append_message(session_key, User, message)`
    ///   `append_message(session_key, Assistant, reply)`
    ///   `store_task_record(session_key, task_id, message, reply)`
    async fn send_message(
        &self,
        session_key: String,
        message: String,
        mcp_contract_address: Option<String>,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<AgentResponse, String>;

    // --- transcript persistence (mutate — deterministic) --------------------
    /// Append one message to a session transcript.
    fn append_message(
        &mut self,
        session_key: String,
        role: ConversationRole,
        content: String,
    ) -> Result<(), String>;

    /// Persist the outcome of a `send_message` call in the task log.
    fn store_task_record(
        &mut self,
        session_key: String,
        task_id: String,
        description: String,
        response: String,
    ) -> Result<(), String>;

    // --- multi-agent chain (query — non-deterministic) ----------------------
    /// Run a sequential chain of tasks.  Does NOT write to state.
    async fn run_workflow(
        &self,
        tasks: Vec<WorkflowTask>,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<WorkflowResult, String>;

    // --- DAG pipeline (query — non-deterministic) ---------------------------
    /// Execute a directed acyclic graph of tasks.  Does NOT write to state.
    async fn run_pipeline(
        &self,
        spec: PipelineSpec,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<String, String>;

    // --- session management (mixed) -----------------------------------------
    fn get_transcript(&self, session_key: String) -> Result<Vec<ConversationMessage>, String>;
    fn clear_session(&mut self, session_key: String) -> Result<(), String>;
    fn list_sessions(&self) -> Result<Vec<String>, String>;

    // --- cron / scheduled tasks ---------------------------------------------
    /// Register a new scheduled task (mutate — deterministic).
    fn create_cron(
        &mut self,
        cron_id: String,
        description: String,
        mcp_contract_address: String,
        every_ms: u64,
    ) -> Result<(), String>;

    /// Execute a cron job on demand (query — non-deterministic LLM call).
    /// Does NOT update `last_run_timestamp`; call `touch_cron` afterwards.
    async fn run_cron(
        &self,
        cron_id: String,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<String, String>;

    /// Update a cron job's last-run timestamp (mutate — deterministic).
    fn touch_cron(&mut self, cron_id: String) -> Result<(), String>;

    fn delete_cron(&mut self, cron_id: String) -> Result<(), String>;
    fn list_crons(&self) -> Result<Vec<CronJob>, String>;

    // --- per-caller memory (mutate / query) ---------------------------------
    fn remember(&mut self, key: String, value: String) -> Result<(), String>;
    fn recall(&self, key: String) -> Result<Option<String>, String>;
    fn forget(&mut self, key: String) -> Result<bool, String>;

    // --- task history (query) -----------------------------------------------
    fn get_task_history(&self, session_key: String) -> Result<Vec<TaskRecord>, String>;

    // --- MCP built-in tools (query — safe for agentic loops) ----------------
    /// HTTP fetch.  Name matches entry in `tools()` schema: "web_fetch".
    async fn web_fetch(
        &self,
        url: String,
        method: Option<String>,
        body: Option<String>,
    ) -> Result<String, String>;

    /// Caller-scoped memory recall.  Name: "recall_memory".
    fn recall_memory(&self, key: String) -> Result<Option<String>, String>;

    /// JSON tool schema array for this contract's MCP server role.
    fn tools(&self) -> String;

    /// System-prompt string for this contract's MCP server role.
    fn prompts(&self) -> String;

    // --- status (query) -----------------------------------------------------
    fn status(&self) -> Result<AppletStatus, String>;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

#[smart_contract]
impl OpenClaw for OpenClawState {
    // ---- constructor -------------------------------------------------------

    #[constructor]
    fn new(system_prompt: Option<String>) -> Result<Self, String>
    where
        Self: Sized,
    {
        let mut config = AppletConfig::default();
        if let Some(prompt) = system_prompt {
            config.system_prompt = prompt;
        }
        Ok(OpenClawState {
            config,
            sessions: WeilMap::new(ID_SESSIONS),
            session_keys: Vec::new(),
            cron_jobs: WeilMap::new(ID_CRON_JOBS),
            cron_ids: Vec::new(),
            user_memory: WeilMap::new(ID_USER_MEMORY),
            task_history: WeilMap::new(ID_TASK_HISTORY),
            agent_registry: WeilMap::new(ID_AGENT_REGISTRY),
            agent_names: Vec::new(),
        })
    }

    // ---- configuration (mutate) --------------------------------------------

    #[mutate]
    fn configure(&mut self, config: AppletConfig) -> Result<(), String> {
        self.config = config;
        Ok(())
    }

    // ---- agent registry (mutate) -------------------------------------------

    #[mutate]
    fn register_agent(&mut self, name: String, contract_address: String) -> Result<(), String> {
        if self.agent_registry.get(&name).is_some() {
            return Err(format!("Agent '{}' is already registered", name));
        }
        self.agent_registry.insert(name.clone(), contract_address);
        self.agent_names.push(name);
        Ok(())
    }

    #[mutate]
    fn unregister_agent(&mut self, name: String) -> Result<(), String> {
        if self.agent_registry.get(&name).is_none() {
            return Err(format!("Agent '{}' is not registered", name));
        }
        self.agent_registry.remove(&name);
        self.agent_names.retain(|n| n != &name);
        Ok(())
    }

    #[query]
    fn list_agents(&self) -> Result<String, String> {
        let agents: Vec<AgentInfo> = self
            .agent_names
            .iter()
            .filter_map(|name| {
                self.agent_registry
                    .get(name)
                    .map(|addr| AgentInfo { name: name.clone(), contract_address: addr })
            })
            .collect();
        serde_json::to_string(&agents).map_err(|e| e.to_string())
    }

    // ---- core AI interaction (query) ---------------------------------------

    #[query]
    async fn send_message(
        &self,
        session_key: String,
        message: String,
        mcp_contract_address: Option<String>,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<AgentResponse, String> {
        let caller = Runtime::sender();
        let task_id = Runtime::uuid();
        let model = model.unwrap_or_else(|| self.config.default_model.clone());
        let mcp_addr = mcp_contract_address.unwrap_or_else(|| Runtime::contract_id());

        // Build task description incorporating recent conversation history.
        let history = self.sessions.get(&session_key).unwrap_or_default();
        let task_description = build_task_description(&message, &history, self.config.max_history_turns);

        let task = TaskWithAgent::new(task_id.clone(), task_description, mcp_addr);
        let reply = Driver::do_task_with_agent(caller, task, model.clone(), model_key)
            .await
            .map_err(|e| format!("Agent execution failed: {}", e))?;

        Ok(AgentResponse { reply, task_id, session_key, model_used: model })
    }

    // ---- transcript persistence (mutate) -----------------------------------

    #[mutate]
    fn append_message(
        &mut self,
        session_key: String,
        role: ConversationRole,
        content: String,
    ) -> Result<(), String> {
        let msg = ConversationMessage {
            role,
            content,
            timestamp: Runtime::block_timestamp(),
        };
        let mut transcript = self.sessions.get(&session_key).unwrap_or_default();
        transcript.push(msg);
        self.sessions.insert(session_key.clone(), transcript);
        if !self.session_keys.contains(&session_key) {
            self.session_keys.push(session_key);
        }
        Ok(())
    }

    #[mutate]
    fn store_task_record(
        &mut self,
        session_key: String,
        task_id: String,
        description: String,
        response: String,
    ) -> Result<(), String> {
        let record = TaskRecord {
            task_id,
            session_key: session_key.clone(),
            description,
            response,
            timestamp: Runtime::block_timestamp(),
        };
        let mut history = self.task_history.get(&session_key).unwrap_or_default();
        history.push(record);
        self.task_history.insert(session_key, history);
        Ok(())
    }

    // ---- multi-agent chain (query) -----------------------------------------

    #[query]
    async fn run_workflow(
        &self,
        tasks: Vec<WorkflowTask>,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<WorkflowResult, String> {
        let caller = Runtime::sender();
        let model = model.unwrap_or_else(|| self.config.default_model.clone());

        if tasks.is_empty() {
            return Err("At least one task is required".to_string());
        }

        let mut chain = ChainWithAgents::new();
        for (i, t) in tasks.iter().enumerate() {
            let task_id = t.task_id.clone().unwrap_or_else(|| format!("task_{}", i));
            chain.add_task(TaskWithAgent::new(
                task_id,
                t.description.clone(),
                t.mcp_contract_address.clone(),
            ));
        }

        match chain.run(caller, model, model_key).await {
            Ok(results) => Ok(WorkflowResult::Ok(results.join("\n\n---\n\n"))),
            Err(err) => Ok(WorkflowResult::Err {
                error: err.err_msg,
                resume_index: err.index,
                previous_result: err.previous_result,
            }),
        }
    }

    // ---- DAG pipeline (query) ----------------------------------------------

    #[query]
    async fn run_pipeline(
        &self,
        spec: PipelineSpec,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<String, String> {
        let caller = Runtime::sender();
        let model = model.unwrap_or_else(|| self.config.default_model.clone());

        let mut pipeline = Pipeline::new(spec.name.clone(), spec.description.clone(), spec.is_repeating);

        for (i, t) in spec.tasks.iter().enumerate() {
            let task_id = t.task_id.clone().unwrap_or_else(|| format!("task_{}", i));
            pipeline.add_task(TaskWithAgent::new(
                task_id,
                t.description.clone(),
                t.mcp_contract_address.clone(),
            ));
        }

        pipeline
            .set_root(spec.root_task_id.clone())
            .map_err(|e| format!("Invalid root task: {}", e))?;

        for edge in &spec.edges {
            let condition = match &edge.condition {
                PipelineCondition::AlwaysTrue => ConditionType::AlwaysTrue,
                PipelineCondition::AlwaysFalse => ConditionType::AlwaysFalse,
                PipelineCondition::Equals(v) => ConditionType::Equals(v.clone()),
                PipelineCondition::NotEquals(v) => ConditionType::NotEquals(v.clone()),
            };
            pipeline
                .add_edge(edge.from_task_id.clone(), edge.to_task_id.clone(), condition)
                .map_err(|e| format!("Invalid pipeline edge: {}", e))?;
        }

        pipeline.start();
        pipeline
            .run(caller, model, model_key)
            .await
            .map_err(|e| format!("Pipeline execution failed: {}", e))
    }

    // ---- session management ------------------------------------------------

    #[query]
    fn get_transcript(&self, session_key: String) -> Result<Vec<ConversationMessage>, String> {
        Ok(self.sessions.get(&session_key).unwrap_or_default())
    }

    #[mutate]
    fn clear_session(&mut self, session_key: String) -> Result<(), String> {
        self.sessions.remove(&session_key);
        self.session_keys.retain(|k| k != &session_key);
        self.task_history.remove(&session_key);
        Ok(())
    }

    #[query]
    fn list_sessions(&self) -> Result<Vec<String>, String> {
        Ok(self.session_keys.clone())
    }

    // ---- cron / scheduled tasks --------------------------------------------

    #[mutate]
    fn create_cron(
        &mut self,
        cron_id: String,
        description: String,
        mcp_contract_address: String,
        every_ms: u64,
    ) -> Result<(), String> {
        if self.cron_jobs.get(&cron_id).is_some() {
            return Err(format!("Cron job '{}' already exists", cron_id));
        }
        self.cron_jobs.insert(cron_id.clone(), CronJob {
            id: cron_id.clone(),
            description,
            mcp_contract_address,
            every_ms,
            last_run_timestamp: None,
            created_at: Runtime::block_timestamp(),
        });
        self.cron_ids.push(cron_id);
        Ok(())
    }

    /// Execute a cron job — query only (LLM call is non-deterministic).
    /// Call `touch_cron(cron_id)` afterwards to persist the last-run timestamp.
    #[query]
    async fn run_cron(
        &self,
        cron_id: String,
        model: Option<String>,
        model_key: Option<String>,
    ) -> Result<String, String> {
        let job = self
            .cron_jobs
            .get(&cron_id)
            .ok_or_else(|| format!("Cron job '{}' not found", cron_id))?;

        let caller = Runtime::sender();
        let model = model.unwrap_or_else(|| self.config.default_model.clone());

        let task = TaskWithAgent::new(Runtime::uuid(), job.description.clone(), job.mcp_contract_address.clone());
        Driver::do_task_with_agent(caller, task, model, model_key)
            .await
            .map_err(|e| format!("Cron execution failed: {}", e))
    }

    /// Record that a cron job ran (mutate — deterministic timestamp update).
    #[mutate]
    fn touch_cron(&mut self, cron_id: String) -> Result<(), String> {
        let mut job = self
            .cron_jobs
            .get(&cron_id)
            .ok_or_else(|| format!("Cron job '{}' not found", cron_id))?;
        job.last_run_timestamp = Some(Runtime::block_timestamp());
        self.cron_jobs.insert(cron_id, job);
        Ok(())
    }

    #[mutate]
    fn delete_cron(&mut self, cron_id: String) -> Result<(), String> {
        if self.cron_jobs.get(&cron_id).is_none() {
            return Err(format!("Cron job '{}' not found", cron_id));
        }
        self.cron_jobs.remove(&cron_id);
        self.cron_ids.retain(|id| id != &cron_id);
        Ok(())
    }

    #[query]
    fn list_crons(&self) -> Result<Vec<CronJob>, String> {
        Ok(self.cron_ids.iter().filter_map(|id| self.cron_jobs.get(id)).collect())
    }

    // ---- per-caller memory -------------------------------------------------

    #[mutate]
    fn remember(&mut self, key: String, value: String) -> Result<(), String> {
        let compound = format!("{}:{}", Runtime::sender(), key);
        self.user_memory.insert(compound, value);
        Ok(())
    }

    #[query]
    fn recall(&self, key: String) -> Result<Option<String>, String> {
        let compound = format!("{}:{}", Runtime::sender(), key);
        Ok(self.user_memory.get(&compound))
    }

    #[mutate]
    fn forget(&mut self, key: String) -> Result<bool, String> {
        let compound = format!("{}:{}", Runtime::sender(), key);
        let existed = self.user_memory.get(&compound).is_some();
        if existed {
            self.user_memory.remove(&compound);
        }
        Ok(existed)
    }

    // ---- task history ------------------------------------------------------

    #[query]
    fn get_task_history(&self, session_key: String) -> Result<Vec<TaskRecord>, String> {
        Ok(self.task_history.get(&session_key).unwrap_or_default())
    }

    // ---- MCP built-in tools (query) ----------------------------------------

    /// HTTP fetch exposed as MCP tool "web_fetch".
    #[query]
    async fn web_fetch(
        &self,
        url: String,
        method: Option<String>,
        body: Option<String>,
    ) -> Result<String, String> {
        let http_method = match method.as_deref().unwrap_or("GET") {
            "POST"   => HttpMethod::Post,
            "PUT"    => HttpMethod::Put,
            "DELETE" => HttpMethod::Delete,
            "PATCH"  => HttpMethod::Patch,
            _        => HttpMethod::Get,
        };
        let mut builder = HttpClient::request(&url, http_method);
        if let Some(b) = body {
            builder = builder.body(b);
        }
        let resp = builder.send().map_err(|e| format!("HTTP request failed: {}", e))?;
        if resp.status() >= 400 {
            return Err(format!("HTTP error {}: {}", resp.status(), resp.text()));
        }
        Ok(resp.text())
    }

    /// Caller-scoped memory recall exposed as MCP tool "recall_memory".
    #[query]
    fn recall_memory(&self, key: String) -> Result<Option<String>, String> {
        let compound = format!("{}:{}", Runtime::sender(), key);
        Ok(self.user_memory.get(&compound))
    }

    /// JSON tool schema array — called by cerebrum Driver to discover tools.
    #[query]
    fn tools(&self) -> String {
        tool_schema_json()
    }

    /// System-prompt string — called by cerebrum Driver for MCP context.
    #[query]
    fn prompts(&self) -> String {
        self.config.system_prompt.clone()
    }

    // ---- status ------------------------------------------------------------

    #[query]
    fn status(&self) -> Result<AppletStatus, String> {
        Ok(AppletStatus {
            contract_id: Runtime::contract_id(),
            active_sessions: self.session_keys.len() as u64,
            cron_jobs_count: self.cron_ids.len() as u64,
            registered_agents_count: self.agent_names.len() as u64,
            default_model: self.config.default_model.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers (not WASM exports)
// ---------------------------------------------------------------------------

/// Prepend recent conversation history to the current message so the LLM
/// maintains context across turns.
fn build_task_description(
    current_message: &str,
    history: &[ConversationMessage],
    max_turns: u32,
) -> String {
    if history.is_empty() {
        return current_message.to_string();
    }
    let start = history.len().saturating_sub(max_turns as usize * 2);
    let recent = &history[start..];

    let mut ctx = String::from("Previous conversation:\n");
    for msg in recent {
        let role = match msg.role {
            ConversationRole::User => "User",
            ConversationRole::Assistant => "Assistant",
            ConversationRole::System => "System",
        };
        ctx.push_str(&format!("{}: {}\n", role, msg.content));
    }
    ctx.push('\n');
    ctx.push_str(&format!("Current message: {}", current_message));
    ctx
}
