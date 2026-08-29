"""
Hexis Tools System

A modular, user-configurable tools system that provides external capabilities
to both the heartbeat (autonomous) and chat (interactive) contexts.

Key components:
- ToolHandler: Abstract base class for tool implementations
- ToolSpec: Tool definition exposed to LLMs
- ToolResult: Structured result from tool execution
- ToolRegistry: Central registry with policy enforcement
- ToolsConfig: Configuration (stored in database)

Example usage:

    from core.tools import ToolRegistry, ToolContext, ToolExecutionContext, create_default_registry

    # Create registry with default tools
    registry = create_default_registry(pool)

    # Get tool specs for LLM
    specs = await registry.get_specs(ToolContext.CHAT)

    # Execute a tool
    result = await registry.execute(
        "recall",
        {"query": "What do I know about Python?"},
        ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="123",
        ),
    )

    if result.success:
        print(result.output)
    else:
        print(f"Error: {result.error}")
"""

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolInvocation,
    OutboundSpec,
    ToolResult,
    ToolSpec,
    SyncToolHandler,
)

from .config import (
    ContextOverrides,
    MCPServerConfig,
    ToolsConfig,
    load_tools_config,
    save_tools_config,
)

from .hooks import (
    FunctionHookHandler,
    HookContext,
    HookEvent,
    HookHandler,
    HookOutcome,
    HookRegistry,
)

from .policy import (
    PolicyCheckResult,
    ToolPolicy,
    create_tool_boundary,
    grant_tool_approval,
    list_approved_tools,
    revoke_tool_approval,
)

from .registry import (
    ExecutionStats,
    ToolRegistry,
    ToolRegistryBuilder,
    create_default_registry,
    create_full_registry,
)

from .memory import create_memory_tools
from .memory_exchange import create_memory_exchange_tools
from .protected_replacement import create_protected_replacement_tools
from .web import (
    create_web_tools,
    WebSearchHandler,
    WebFetchHandler,
    WebSummarizeHandler,
)
from .filesystem import (
    create_filesystem_tools,
    ReadFileHandler,
    WriteFileHandler,
    EditFileHandler,
    GlobHandler,
    GrepHandler,
    ListDirectoryHandler,
)
from .shell import (
    create_shell_tools,
    ShellHandler,
    SafeShellHandler,
    ScriptRunnerHandler,
)
from .mcp import (
    MCPClient,
    MCPError,
    MCPManager,
    MCPToolHandler,
    create_mcp_manager,
)
from .sync_adapter import (
    SyncToolAdapter,
    CombinedToolHandler,
    create_sync_tool_handler,
)
from .calendar import (
    create_calendar_tools,
    GoogleCalendarHandler,
    CreateCalendarEventHandler,
    UpdateCalendarEventHandler,
    DeleteCalendarEventHandler,
)
from .email import (
    create_email_tools,
    EmailSendHandler,
    SendGridEmailHandler,
    EmailListHandler,
    EmailReadHandler,
    EmailSearchHandler,
)
from .messaging import (
    create_messaging_tools,
    DiscordSendHandler,
    SlackSendHandler,
    TelegramSendHandler,
    SignalSendHandler,
)
from .code_execution import (
    create_code_execution_tools,
    CodeExecutionHandler,
    cleanup_session_repl,
)
from .browser import (
    create_browser_tools,
    BrowserHandler,
    cleanup_browser_session,
)
from .ingest import (
    create_ingest_tools,
    FastIngestHandler,
    SlowIngestHandler,
    HybridIngestHandler,
    URLIngestHandler,
)
from .workflow import (
    create_workflow_tools,
    WorkflowHandler,
    WorkflowPlan,
    WorkflowStep,
    WorkflowStepResult,
)
from .dynamic import (
    create_dynamic_tools,
    CreateToolHandler,
    load_dynamic_tools,
)
from .goals import (
    create_goal_tools,
    ManageGoalsHandler,
)
from .backlog import (
    create_backlog_tools,
    ManageBacklogHandler,
)
from .cron import (
    create_cron_tools,
    ManageScheduleHandler,
)
from .responsibilities import (
    create_responsibility_tools,
    ManageResponsibilityHandler,
)
from .sessions import (
    create_session_tools,
    ManageSessionsHandler,
)
from .integrations import (
    create_integration_tools,
    IntegrationSetupStatusHandler,
    StartIntegrationSetupHandler,
    ConfigureChannelIntegrationHandler,
    VerifyChannelIntegrationHandler,
    GmailSetupStatusHandler,
    ConnectGmailHandler,
    CompleteGmailConnectionHandler,
    RevokeGmailConnectionHandler,
    ConnectTwitterXHandler,
    CompleteTwitterXConnectionHandler,
    RevokeTwitterXConnectionHandler,
    StartGmailBackfillHandler,
    GmailBackfillStatusHandler,
    ControlGmailBackfillHandler,
    StartConnectorBackfillHandler,
    ConnectorBackfillStatusHandler,
    ControlConnectorBackfillHandler,
    ConnectorActionPolicyStatusHandler,
    GrantConnectorActionPolicyHandler,
    RevokeConnectorActionPolicyHandler,
)
from .life_integrations import (
    create_life_integration_tools,
    ConnectLifeIntegrationHandler,
    ConnectSpotifyHandler,
    CompleteSpotifyConnectionHandler,
    RevokeLifeIntegrationHandler,
    NotionSearchHandler,
    NotionGetPageHandler,
    NotionQueryDataSourceHandler,
    NotionCreatePageHandler,
    SpotifySearchHandler,
    SpotifyPlaybackStateHandler,
    SpotifyControlPlaybackHandler,
    HomeAssistantStatesHandler,
    HomeAssistantCallServiceHandler,
    WeatherForecastHandler,
    TrelloListBoardsHandler,
    TrelloListCardsHandler,
    TrelloCreateCardHandler,
    TrelloUpdateCardHandler,
)
from .gmail_actions import (
    create_gmail_action_tools,
    GmailSendHandler,
    GmailReplyHandler,
    GmailLabelHandler,
    GmailSpamTriageHandler,
    GmailDeleteHandler,
)
from .twitter_x_actions import (
    create_twitter_x_action_tools,
    TwitterXPostHandler,
    TwitterXReplyHandler,
    TwitterXDMSendHandler,
)
from .self_repair import (
    create_self_repair_tools,
    SelfRepairHandler,
)
from .questions import AskUserHandler, create_question_tools
from .local_audio_analysis import (
    AnalyzeLocalAudioHandler,
    TranscribeAudioHandler,
    create_local_audio_analysis_tools,
)
from .operator_policies import (
    ManageOperatorPoliciesHandler,
    create_operator_policy_tools,
)
from .nodes import NodeInvokeHandler, StructuredNodeHandler, create_node_tools
from .speech import SpeakHandler, create_speech_tools

__all__ = [
    # Base classes
    "ToolCategory",
    "ToolContext",
    "ToolErrorType",
    "ToolExecutionContext",
    "ToolHandler",
    "ToolInvocation",
    "OutboundSpec",
    "ToolResult",
    "ToolSpec",
    "SyncToolHandler",
    "AskUserHandler",
    "create_question_tools",
    "AnalyzeLocalAudioHandler",
    "TranscribeAudioHandler",
    "create_local_audio_analysis_tools",
    "ManageOperatorPoliciesHandler",
    "create_operator_policy_tools",
    "NodeInvokeHandler",
    "StructuredNodeHandler",
    "create_node_tools",
    "SpeakHandler",
    "create_speech_tools",
    # Config
    "ContextOverrides",
    "MCPServerConfig",
    "ToolsConfig",
    "load_tools_config",
    "save_tools_config",
    # Hooks
    "FunctionHookHandler",
    "HookContext",
    "HookEvent",
    "HookHandler",
    "HookOutcome",
    "HookRegistry",
    # Policy
    "PolicyCheckResult",
    "ToolPolicy",
    "create_tool_boundary",
    "grant_tool_approval",
    "list_approved_tools",
    "revoke_tool_approval",
    # Registry
    "ExecutionStats",
    "ToolRegistry",
    "ToolRegistryBuilder",
    "create_default_registry",
    "create_full_registry",
    # Tool factories
    "create_memory_tools",
    "create_memory_exchange_tools",
    "create_protected_replacement_tools",
    "create_web_tools",
    # Web tools
    "WebSearchHandler",
    "WebFetchHandler",
    "WebSummarizeHandler",
    # Filesystem tools
    "create_filesystem_tools",
    "ReadFileHandler",
    "WriteFileHandler",
    "EditFileHandler",
    "GlobHandler",
    "GrepHandler",
    "ListDirectoryHandler",
    # Shell tools
    "create_shell_tools",
    "ShellHandler",
    "SafeShellHandler",
    "ScriptRunnerHandler",
    # MCP tools
    "MCPClient",
    "MCPError",
    "MCPManager",
    "MCPToolHandler",
    "create_mcp_manager",
    # Sync adapter
    "SyncToolAdapter",
    "CombinedToolHandler",
    "create_sync_tool_handler",
    # Calendar tools
    "create_calendar_tools",
    "GoogleCalendarHandler",
    "CreateCalendarEventHandler",
    "UpdateCalendarEventHandler",
    "DeleteCalendarEventHandler",
    # Email tools
    "create_email_tools",
    "EmailSendHandler",
    "SendGridEmailHandler",
    "EmailListHandler",
    "EmailReadHandler",
    "EmailSearchHandler",
    # Messaging tools
    "create_messaging_tools",
    "DiscordSendHandler",
    "SlackSendHandler",
    "TelegramSendHandler",
    "SignalSendHandler",
    # Code execution tools
    "create_code_execution_tools",
    "CodeExecutionHandler",
    "cleanup_session_repl",
    # Browser tools
    "create_browser_tools",
    "BrowserHandler",
    "cleanup_browser_session",
    # Ingest tools
    "create_ingest_tools",
    "FastIngestHandler",
    "SlowIngestHandler",
    "HybridIngestHandler",
    "URLIngestHandler",
    # Workflow tools
    "create_workflow_tools",
    "WorkflowHandler",
    "WorkflowPlan",
    "WorkflowStep",
    "WorkflowStepResult",
    # Dynamic tools
    "create_dynamic_tools",
    "CreateToolHandler",
    "load_dynamic_tools",
    # Goal tools
    "create_goal_tools",
    "ManageGoalsHandler",
    # Backlog tools
    "create_backlog_tools",
    "ManageBacklogHandler",
    # Cron/scheduling tools
    "create_cron_tools",
    "ManageScheduleHandler",
    # Ambient responsibility tools
    "create_responsibility_tools",
    "ManageResponsibilityHandler",
    # Self-repair tools
    "create_self_repair_tools",
    "SelfRepairHandler",
    # Sub-agent session tools
    "create_session_tools",
    "ManageSessionsHandler",
    "create_integration_tools",
    "IntegrationSetupStatusHandler",
    "StartIntegrationSetupHandler",
    "ConfigureChannelIntegrationHandler",
    "VerifyChannelIntegrationHandler",
    "GmailSetupStatusHandler",
    "ConnectGmailHandler",
    "CompleteGmailConnectionHandler",
    "RevokeGmailConnectionHandler",
    "ConnectTwitterXHandler",
    "CompleteTwitterXConnectionHandler",
    "RevokeTwitterXConnectionHandler",
    "StartGmailBackfillHandler",
    "GmailBackfillStatusHandler",
    "ControlGmailBackfillHandler",
    "StartConnectorBackfillHandler",
    "ConnectorBackfillStatusHandler",
    "ControlConnectorBackfillHandler",
    "ConnectorActionPolicyStatusHandler",
    "GrantConnectorActionPolicyHandler",
    "RevokeConnectorActionPolicyHandler",
    "create_life_integration_tools",
    "ConnectLifeIntegrationHandler",
    "ConnectSpotifyHandler",
    "CompleteSpotifyConnectionHandler",
    "RevokeLifeIntegrationHandler",
    "NotionSearchHandler",
    "NotionGetPageHandler",
    "NotionQueryDataSourceHandler",
    "NotionCreatePageHandler",
    "SpotifySearchHandler",
    "SpotifyPlaybackStateHandler",
    "SpotifyControlPlaybackHandler",
    "HomeAssistantStatesHandler",
    "HomeAssistantCallServiceHandler",
    "WeatherForecastHandler",
    "TrelloListBoardsHandler",
    "TrelloListCardsHandler",
    "TrelloCreateCardHandler",
    "TrelloUpdateCardHandler",
    "create_gmail_action_tools",
    "create_twitter_x_action_tools",
    "TwitterXPostHandler",
    "TwitterXReplyHandler",
    "TwitterXDMSendHandler",
    "GmailSendHandler",
    "GmailReplyHandler",
    "GmailLabelHandler",
    "GmailSpamTriageHandler",
    "GmailDeleteHandler",
]
