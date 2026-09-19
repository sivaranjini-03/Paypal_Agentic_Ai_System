# Architecture

## Request Flow

```text
User request
  -> AgentWorkflow.route
     -> HostAgent.route
        -> domain catalogue + LLM routing
  -> AgentWorkflow.select_domain
  -> AgentWorkflow.plan
     -> DomainAgent.prepare
        -> MCP tool discovery
        -> ToolRetriever candidate selection
        -> LLM plan
  -> AgentWorkflow.execute
     -> MCPClient.call_tool
        -> ToolExecutor or internal capability handler
  -> AgentWorkflow.evaluate / recover
  -> AgentWorkflow.respond
     -> HostAgent synthesis
```

## Components

- `app/tools`: Parses the Postman collection, stores normalized tools, retrieves
  candidates, and executes HTTP requests with credentials held outside prompts.
- `app/mcp`: Exposes registry tools to agents through a common discovery and
  invocation interface.
- `app/agents`: The host chooses domains; domain agents choose and execute
  tools. Plans carry explicit input/output context references.
- `app/capabilities`: Internal tools for knowledge retrieval and system
  introspection. They are registered like API tools and do not call PayPal.
- `app/workflows`: LangGraph orchestration, state transitions, and recovery.
- `app/observability.py`: Records request, workflow, and tool metadata without
  storing credentials or API payloads.

## Three Smoke Scenarios

1. An authorization-refund question should route to `Knowledge` and use RAG.
2. A tools-available question should route to `System` and query the registry.
3. An order refund should route `Orders` before `Payments`, passing the capture
   identifier from the order lookup into the refund call.