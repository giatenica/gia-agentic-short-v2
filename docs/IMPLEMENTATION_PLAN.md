# GIA Agentic v2 - Implementation Plan

**Version:** 1.0  
**Author:** Gia Tenica*  
**Date:** 2 January 2026

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher. For more information see: https://giatenica.com

---

## Executive Summary

This document outlines the implementation plan for migrating the GIA Agentic Research Pipeline from v1 (custom agent framework) to v2 (LangGraph-based architecture). The new system transforms complex research queries into auditable academic outputs through a multi-agent workflow with self-critique loops, evidence tracking, and human-in-the-loop capabilities.

---

## Architecture Comparison

### v1 to v2 Migration

| v1 Component | v2 LangGraph Equivalent |
|--------------|------------------------|
| 25 discrete agents (A01-A25) | 5 core nodes with sub-graphs |
| Custom workflow orchestrator | `StateGraph` with conditional edges |
| Manual state passing | `WorkflowState` TypedDict with reducers |
| No persistence | `MemorySaver` / `SqliteSaver` checkpointers |
| Custom tracing | LangSmith automatic tracing |
| Direct Anthropic client | `langchain-anthropic` with tool binding |
| Sequential phases | Parallel execution with conditional routing |

### v2 Node Architecture

The workflow follows the established academic research process, respecting the iterative and non-linear nature of scholarly inquiry.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ACADEMIC RESEARCH WORKFLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [HTML Form] ──► INTAKE ──► LITERATURE_REVIEWER ──► GAP_IDENTIFIER      │
│                    │               │                      │              │
│               [explore data]       │                      ▼              │
│               [validate RQ]        │              [refine question]      │
│                    │               │                      │              │
│                    ▼               ▼                      ▼              │
│               HITL (approve) ◄─── [iterative] ◄──── PLANNER             │
│                    │                                      │              │
│                    │                                      ▼              │
│                    │           ┌──────────────────────────┴───┐         │
│                    │           │     Research Type Router     │         │
│                    │           └──────┬───────────┬───────────┘         │
│                    │                  │           │                      │
│                    │           [empirical]   [theoretical]               │
│                    │                  ▼           ▼                      │
│                    │           DATA_ANALYST  CONCEPTUAL_SYNTHESIZER     │
│                    │                  │           │                      │
│                    │                  └─────┬─────┘                      │
│                    │                        ▼                            │
│                    │                    WRITER                           │
│                    │                        │                            │
│                    │                        ▼                            │
│                    │                    REVIEWER                         │
│                    │                        │                            │
│                    │                   ┌────┴────┐                       │
│                    │                   │         │                       │
│                    │               APPROVE    REVISE                     │
│                    │                   │         │                       │
│                    │                   ▼         ▼                       │
│                    └────────────► OUTPUT ◄── WRITER                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Academic Research Process

The workflow mirrors how experienced researchers actually work:

| Stage | Academic Activity | Node(s) | Key Principle |
|-------|-------------------|---------|---------------|
| 1 | **Problem Identification** | INTAKE | Start with a question, not a hypothesis |
| 2 | **Preliminary Literature Survey** | LITERATURE_REVIEWER | Understand the field before planning |
| 3 | **Gap Identification** | GAP_IDENTIFIER | Find what's missing; refine question |
| 4 | **Research Design** | PLANNER | Methodology follows from gap + data |
| 5 | **Data Collection/Analysis** | DATA_ANALYST | For empirical work |
| 5a | **Conceptual Development** | CONCEPTUAL_SYNTHESIZER | For theoretical work |
| 6 | **Writing** | WRITER | Argument construction, not just reporting |
| 7 | **Critical Review** | REVIEWER | Self-critique before submission |

### Why Literature Review Must Come First

In the current plan, PLANNER comes before SEARCHER. This is backwards. The canonical academic process requires:

1. **You cannot plan research without knowing what exists** - A hypothesis formed in ignorance may duplicate prior work or miss established consensus
2. **Research questions evolve through reading** - Initial questions are refined, narrowed, or redirected based on literature
3. **Methodology is constrained by precedent** - What methods have others used? What worked? What didn't?
4. **Gap identification IS the contribution** - The gap in literature becomes the paper's contribution statement

### Entry Point: Research Intake Form

The workflow always begins with the HTML intake form (`public/research_intake_form.html`). This ensures:

1. **User intent is captured first** - Initial research question, area of interest
2. **Existing data is explored** - Any uploaded datasets are analyzed
3. **Context is established** - But NOT a fixed hypothesis (which comes later)
4. **The question may change** - Literature review may redirect the inquiry

### Node Responsibilities (Revised for Academic Process)

| Node | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| **INTAKE** | Parse form, validate, explore data, establish initial scope | Form data, files | Validated project context |
| **LITERATURE_REVIEWER** | Systematic literature search, extract key findings, identify seminal works | Initial question, seed literature | Literature synthesis, key themes, methodology precedents |
| **GAP_IDENTIFIER** | Identify what's missing, contradictions, opportunities | Literature synthesis | Refined research question, contribution statement, gap analysis |
| **PLANNER** | Design methodology, create research plan, define success criteria | Refined question, gaps, available data | Research plan, methodology, analysis approach |
| **DATA_ANALYST** | Statistical analysis, pattern detection (empirical work) | Data, methodology | Findings, statistical results |
| **CONCEPTUAL_SYNTHESIZER** | Theory building, framework construction (theoretical work) | Literature, gaps | Conceptual framework, propositions |
| **WRITER** | Construct argument, integrate evidence, format output | All prior outputs | Draft manuscript |
| **REVIEWER** | Critical evaluation, consistency check, style enforcement | Draft | Critique, revision requests |

---

## Writing Style Guide Reference

All written output must conform to the **Academic Finance Paper Style Guide** located at:

📄 **[docs/writing_style_guide.md](writing_style_guide.md)**

This guide defines:

| Section | Contents |
|---------|----------|
| **Journal Overview** | Target journals (RFS, JFE, JF, JFQA), acceptance rates, submission fees |
| **Document Formatting** | Page layout, typography, section numbering |
| **Paper Structure** | Standard sections, short paper format (5-10 pages) |
| **Writing Style** | Voice, tone, precision, hedging language |
| **Banned Words** | 100+ words to avoid (marketing language, buzzwords) |
| **Citations** | Chicago Author-Date format, in-text and reference list |
| **Tables and Figures** | Formatting standards, placement rules |
| **LaTeX Templates** | Ready-to-use templates for target journals |

### Style Guide Usage by Node

| Node | Style Guide Sections Used |
|------|---------------------------|
| WRITER | Paper Structure, Writing Style, Banned Words, Citations |
| REVIEWER | All sections (validation) |
| OUTPUT | Document Formatting, LaTeX Templates |

### Extending the Style Guide

The style guide can be extended for:
- Additional target journals (add to Journal Overview)
- Domain-specific terminology (add to Writing Style)
- Custom banned words (add to Banned Words section)
- Alternative citation styles (add to Citations)

---

## Intake Form Processing

### Form Fields to State Mapping

| Form Field | State Key | Processing |
|------------|-----------|------------|
| `title` | `project_title` | Direct mapping |
| `research_question` | `original_query` | Primary input for PLANNER |
| `target_journal` | `target_journal` | Influences style/length requirements |
| `paper_type` | `paper_type` | Determines output format |
| `research_type` | `research_type` | Guides methodology selection |
| `hypothesis` | `user_hypothesis` | Optional; if provided, shapes research direction |
| `data_files` | `uploaded_data` | Triggers data exploration before planning |
| `data_description` | `data_context` | User's description of their data |
| `data_sources` | `planned_data_sources` | Where to look for additional data |
| `key_variables` | `key_variables` | Variables to focus on |
| `methodology` | `proposed_methodology` | User's preferred approach |
| `related_literature` | `seed_literature` | Starting point for literature search |
| `expected_contribution` | `expected_contribution` | What gap the paper fills |
| `deadline` | `deadline` | Time constraints |
| `constraints` | `constraints` | Page limits, requirements |

### Data Exploration (Before Planning)

When users upload data files or describe existing data:

1. **File Analysis** - Parse uploaded ZIP archives, identify file types
2. **Schema Detection** - Detect columns, data types, date ranges
3. **Summary Statistics** - Generate descriptive stats for numeric columns
4. **Variable Mapping** - Match detected columns to user's `key_variables`
5. **Quality Assessment** - Check for missing values, outliers, data issues
6. **Feasibility Check** - Can the research question be answered with this data?

```python
class IntakeState(TypedDict):
    # Form inputs
    project_title: str
    original_query: str  # research_question
    target_journal: str
    paper_type: str
    research_type: str
    user_hypothesis: str | None
    
    # Data context
    uploaded_data: list[DataFile] | None
    data_context: str | None
    data_exploration_results: DataExplorationResult | None
    
    # Research context
    planned_data_sources: list[str]
    key_variables: list[str]
    proposed_methodology: str | None
    seed_literature: list[str]
    expected_contribution: str | None
    
    # Constraints
    deadline: date | None
    constraints: str | None
```

---

## Tool Strategy: Best Tool for Each Task

The system dynamically selects the optimal tool for each sub-task based on query type, recency requirements, and source needs.

### Search Tool Selection Matrix

| Query Type | Primary Tool | Fallback | Use Case |
|------------|-------------|----------|----------|
| Current events (< 7 days) | **Tavily** (real-time) | Claude web search | Breaking news, recent developments |
| General web research | **Tavily** | Webpage parsing | Broad topic exploration |
| Academic/scientific | **Semantic Scholar API** | arXiv API | Peer-reviewed research, citations |
| Deep page analysis | **Claude web search** | Tavily + parse | Complex page understanding |
| Specific URL content | **Webpage parser** | Claude web fetch | Extract from known sources |
| Statistical data | **Tavily** + validation | Claude analysis | Numbers requiring verification |

### Claude's Extended Capabilities

1. **Native Tool Use (Function Calling)**
   - Claude binds to all tools via `model.bind_tools()`
   - ReAct pattern: Claude reasons about which tool to call
   - Parallel tool calls when queries are independent
   - Tool result synthesis across multiple sources

2. **Claude Web Search Integration**
   - Use Claude's built-in web capabilities for complex queries
   - Fallback when Tavily results are insufficient
   - Deep content understanding vs. snippet extraction

3. **Dynamic Tool Routing**
   ```python
   def select_search_tool(query: str, query_type: QueryType) -> Tool:
       if query_type == QueryType.BREAKING_NEWS:
           return tavily_search  # Real-time priority
       elif query_type == QueryType.ACADEMIC:
           return semantic_scholar_search  # Peer-reviewed
       elif query_type == QueryType.DEEP_ANALYSIS:
           return claude_web_search  # Complex understanding
       else:
           return tavily_search  # Default
   ```

4. **Multi-Tool Orchestration**
   - SEARCHER node can invoke multiple tools per sub-question
   - Results are merged and deduplicated
   - Confidence scores weight tool reliability

### Tool Reliability Hierarchy

| Priority | Tool | Strengths | Limitations |
|----------|------|-----------|-------------|
| 1 | Tavily | Fast, real-time, broad coverage | Snippet-level depth |
| 2 | Semantic Scholar | Academic authority, citations | Limited to research papers |
| 3 | arXiv API | Preprints, cutting-edge | Not peer-reviewed |
| 4 | Claude web search | Deep understanding | Rate limited |
| 5 | Direct webpage parse | Full content | Requires known URLs |

---

## LangGraph Capabilities to Leverage

This section documents all LangGraph capabilities that yield benefits for our workflow. Each capability is mapped to where it should be used.

### Core Graph Features

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **StateGraph** | Graph with shared state via reducers | Main workflow | Central state management |
| **add_messages** | Built-in message list reducer | All nodes | Automatic message merging |
| **Conditional Edges** | Route based on state/function output | After each node | Dynamic workflow paths |
| **add_sequence** | Chain nodes in order | Section writers | Simplified sequential flow |

### Parallel Execution and Map-Reduce

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **Send API** | Spawn parallel node executions | SEARCHER (multi-query), WRITER (sections) | 3-5x faster execution |
| **Map-Reduce Pattern** | Fan-out to parallel tasks, aggregate results | Search sub-questions, section writing | Process multiple items simultaneously |
| **Parallel Tool Calls** | Multiple tool calls in single step | SEARCHER node | Concurrent API requests |

```python
# Example: Map-Reduce for parallel search
from langgraph.types import Send

def route_to_searches(state: WorkflowState):
    """Fan out to parallel search nodes for each sub-question."""
    return [
        Send("search_single", {"query": q, "question_id": i})
        for i, q in enumerate(state["research_plan"].sub_questions)
    ]
```

### Human-in-the-Loop (HITL)

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **interrupt()** | Pause for human input | Plan approval, final review | Quality gates |
| **interrupt_before** | Pause before node execution | Pre-WRITER, pre-OUTPUT | Review before action |
| **interrupt_after** | Pause after node execution | Post-REVIEWER | Inspect results |
| **Command(resume=)** | Resume with human-provided value | All HITL points | Incorporate feedback |
| **HumanInterrupt** | Structured interrupt with config | Tool approval | Define allowed actions |

```python
# Example: HITL at plan approval
from langgraph.types import interrupt, Command

def planner_node(state: WorkflowState):
    plan = generate_research_plan(state["original_query"])
    
    # Pause for human review
    approved_plan = interrupt({
        "action": "review_plan",
        "plan": plan.model_dump(),
        "message": "Please review and approve the research plan"
    })
    
    return {"research_plan": approved_plan}
```

### Streaming Modes

| Mode | Description | Where to Use | Benefit |
|------|-------------|--------------|---------|
| **values** | Full state after each step | Debugging | See complete state |
| **updates** | Only changed values per step | Production UI | Efficient updates |
| **messages** | Token-by-token LLM output | WRITER node | Real-time text streaming |
| **custom** | User-defined events via StreamWriter | Progress tracking | Custom progress events |
| **debug** | Checkpoints + tasks | Development | Full visibility |

```python
# Example: Multi-mode streaming for UI
async for event in graph.astream(
    input, 
    config,
    stream_mode=["updates", "messages", "custom"]
):
    if event[0] == "messages":
        # Stream tokens to UI
        yield {"type": "token", "content": event[1]}
    elif event[0] == "custom":
        # Progress updates
        yield {"type": "progress", "data": event[1]}
```

### Persistence and Memory

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **Checkpointer** | Save/restore graph state | All workflows | Resume after failure |
| **SqliteSaver** | Persistent checkpoints | Production | Durable execution |
| **MemorySaver** | In-memory checkpoints | Development/testing | Fast iteration |
| **BaseStore** | Cross-session key-value storage | User preferences, facts | Long-term memory |
| **get_state()** | Retrieve current state | HITL, debugging | Inspect workflow |
| **get_state_history()** | Full state timeline | Time travel, auditing | Replay any point |
| **update_state()** | Modify state externally | HITL corrections | Human edits |

```python
# Example: Time travel debugging
history = list(graph.get_state_history(config))
# Replay from specific checkpoint
old_config = history[5].config
result = graph.invoke(None, old_config)
```

### Subgraphs

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **Nested Graphs** | Graph as node in parent graph | Section writers, tool groups | Modular design |
| **get_subgraphs()** | List nested graphs | Debugging | Inspect structure |
| **subgraphs=True** | Stream events from nested graphs | UI visualization | Full visibility |

```python
# Example: Section writer as subgraph
section_writer_graph = create_section_writer_graph()

main_graph = StateGraph(WorkflowState)
main_graph.add_node("writer", section_writer_graph)  # Subgraph as node
```

### Error Handling and Retry

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **RetryPolicy** | Automatic retry with backoff | API calls, search nodes | Resilience |
| **handle_tool_errors** | Tool error handling | ToolNode | Graceful degradation |
| **Fallback nodes** | Alternative paths on error | All nodes | Partial results |

```python
# Example: Retry policy for search node
from langgraph.types import RetryPolicy

graph.add_node(
    "searcher",
    searcher_node,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        retry_on=(RateLimitError, TimeoutError)
    )
)
```

### Caching

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **CachePolicy** | Cache node results | Expensive LLM calls | Cost reduction |
| **cache_policy.ttl** | Time-to-live for cache | Search results | Fresh data balance |

```python
# Example: Cache expensive analysis
from langgraph.types import CachePolicy

graph.add_node(
    "analyst",
    analyst_node,
    cache_policy=CachePolicy(ttl=3600)  # Cache for 1 hour
)
```

### Multi-Agent Patterns

| Pattern | Description | Where to Use | Benefit |
|---------|-------------|--------------|---------|
| **Supervisor** | Central coordinator delegates to workers | Complex research queries | Orchestrated collaboration |
| **Swarm** | Peer agents with handoff tools | Dynamic task routing | Flexible coordination |
| **Handoff Tools** | Transfer control between agents | Agent collaboration | Seamless transitions |

```python
# Example: Research supervisor pattern
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    agents=[planner_agent, searcher_agent, writer_agent],
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    prompt="You coordinate academic research. Delegate to specialists."
)
```

### Prebuilt Components

| Component | Description | Where to Use | Benefit |
|-----------|-------------|--------------|---------|
| **create_react_agent** | ReAct agent with tools | Individual nodes | Quick agent setup |
| **ToolNode** | Execute tools from messages | All tool-using nodes | Standard tool execution |
| **tools_condition** | Route based on tool calls | After agent nodes | Standard routing |
| **InjectedState** | Pass state to tools | Context-aware tools | State access in tools |
| **InjectedStore** | Pass store to tools | Memory-aware tools | Persistent data access |

### Functional API (Alternative Pattern)

| Capability | Description | Where to Use | Benefit |
|------------|-------------|--------------|---------|
| **@entrypoint** | Function-based workflow | Simple linear flows | Cleaner syntax |
| **@task** | Parallel task execution | Fan-out operations | Easy parallelism |
| **entrypoint.final** | Return different from saved state | Stateful workflows | Flexible state management |

```python
# Example: Functional API for simple research
from langgraph.func import entrypoint, task

@task
def search_source(query: str) -> SearchResult:
    return tavily_search(query)

@entrypoint(checkpointer=checkpointer)
def research_workflow(query: str) -> ResearchOutput:
    # Fan out to parallel searches
    futures = [search_source(q) for q in decompose_query(query)]
    results = [f.result() for f in futures]
    return synthesize(results)
```

### Capability Usage Matrix by Sprint

| Capability | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 | Sprint 5 | Sprint 6 | Sprint 7 | Sprint 8 |
|------------|----------|----------|----------|----------|----------|----------|----------|----------|
| StateGraph | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| add_messages | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Conditional Edges | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| interrupt() | ✓ | | ✓ | ✓ | | | ✓ | |
| Send API (parallel) | | ✓ | | | ✓ | | | |
| Streaming | | | | | | ✓ | | ✓ |
| RetryPolicy | | ✓ | | | ✓ | | | |
| CachePolicy | | ✓ | | | ✓ | | | |
| Subgraphs | | | | | | ✓ | | ✓ |
| ToolNode | | ✓ | ✓ | ✓ | ✓ | | | |
| InjectedState | | ✓ | ✓ | | ✓ | | | |

### Sprint Overview (Academic Research Sequence)

| Sprint | Node(s) | Academic Stage | Key Deliverable |
|--------|---------|----------------|-----------------|
| 0 | Foundation | Setup | Project infrastructure |
| 1 | INTAKE | Problem Identification | Validated research context |
| 2 | LITERATURE_REVIEWER | Literature Survey | Systematic review, key themes |
| 3 | GAP_IDENTIFIER | Gap Analysis | Refined question, contribution |
| 4 | PLANNER | Research Design | Methodology, analysis plan |
| 5 | DATA_ANALYST / CONCEPTUAL_SYNTHESIZER | Analysis | Findings or framework |
| 6 | WRITER | Writing | Draft manuscript |
| 7 | REVIEWER | Critical Review | Quality assessment, revisions |
| 8 | Assembly | Integration | Complete workflow |

### Academic Workflow Entry Point

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ACADEMIC RESEARCH ENTRY POINT                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   [User fills HTML Form with initial research question]             │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────┐                                               │
│   │   INTAKE Node   │◄── Validates input, explores any data         │
│   └────────┬────────┘                                               │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────────────┐                                       │
│   │  LITERATURE_REVIEWER    │◄── What does the field know?          │
│   └────────┬────────────────┘                                       │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────────────┐                                       │
│   │   GAP_IDENTIFIER        │◄── What's missing? Refine question    │
│   └────────┬────────────────┘                                       │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────────────┐                                       │
│   │      PLANNER            │◄── NOW design methodology             │
│   └─────────────────────────┘                                       │
│                                                                      │
│   KEY: Literature review BEFORE planning. Gaps inform methodology.  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Sprints

### Sprint 0: Foundation (Complete)
**Duration:** Done  
**Status:** Complete

#### Deliverables
- [x] Project structure with `uv`
- [x] Basic agent framework (`create_react_agent`, `create_research_agent`)
- [x] Memory system (checkpointer + store)
- [x] Tool definitions (Tavily search, utilities)
- [x] LangGraph Studio integration
- [x] Environment configuration
- [x] Copilot instructions

---

### Sprint 1: Intake Processing and State Schema (Complete)
**Duration:** Done  
**Status:** Complete

**Goal:** Process intake form, explore user data, and define workflow state

The intake form is the **mandatory entry point** for all research projects. This sprint ensures:
1. All user-provided information is captured and validated
2. Any uploaded data is explored before planning begins
3. The research question and context are fully understood

#### LangGraph Capabilities Used
- `StateGraph` with `TypedDict` state schema
- `add_messages` reducer for message list management
- `interrupt()` for data exploration approval (if issues found)
- `Conditional Edges` for routing based on data presence
- Pydantic models for intake form validation

#### Tasks

1. **Create INTAKE node** (`src/nodes/intake.py`)
   ```python
   from langgraph.types import interrupt
   
   def intake_node(state: WorkflowState) -> dict:
       """Process intake form submission and validate inputs."""
       # Parse form data into structured state
       intake_data = parse_intake_form(state["form_data"])
       
       # Validate required fields
       validation = validate_intake(intake_data)
       if validation.errors:
           # Pause for user to fix issues
           fixed = interrupt({
               "action": "fix_intake_errors",
               "errors": validation.errors,
               "original_data": intake_data
           })
           intake_data = fixed
       
       return {
           "original_query": intake_data.research_question,
           "project_title": intake_data.title,
           "target_journal": intake_data.target_journal,
           "paper_type": intake_data.paper_type,
           "user_hypothesis": intake_data.hypothesis,
           "proposed_methodology": intake_data.methodology,
           "seed_literature": intake_data.related_literature,
           "expected_contribution": intake_data.expected_contribution,
           "constraints": intake_data.constraints,
           "deadline": intake_data.deadline,
           "uploaded_data": intake_data.data_files,
           "data_context": intake_data.data_description,
           "key_variables": intake_data.key_variables,
           "status": ResearchStatus.INTAKE_COMPLETE
       }
   ```

2. **Create DATA_EXPLORER node** (`src/nodes/data_explorer.py`)
   ```python
   def data_explorer_node(state: WorkflowState) -> dict:
       """Analyze uploaded data before planning phase."""
       if not state.get("uploaded_data"):
           return {"data_exploration_results": None}
       
       results = DataExplorationResult(
           files_analyzed=[],
           schema_detected={},
           summary_statistics={},
           variable_mapping={},
           quality_issues=[],
           feasibility_assessment=""
       )
       
       for data_file in state["uploaded_data"]:
           # Analyze each file
           file_analysis = analyze_data_file(data_file)
           results.files_analyzed.append(file_analysis)
           
           # Map detected columns to user's key variables
           if state.get("key_variables"):
               mapping = map_variables(
                   detected_columns=file_analysis.columns,
                   user_variables=state["key_variables"]
               )
               results.variable_mapping.update(mapping)
       
       # Assess if research question can be answered with this data
       results.feasibility_assessment = assess_data_feasibility(
           research_question=state["original_query"],
           data_analysis=results
       )
       
       # Alert user if data quality issues found
       if results.quality_issues:
           interrupt({
               "action": "review_data_quality",
               "issues": results.quality_issues,
               "message": "Data quality issues detected. Review before proceeding."
           })
       
       return {"data_exploration_results": results}
   ```

3. **Add conditional routing for data exploration**
   ```python
   def route_after_intake(state: WorkflowState) -> str:
       """Route to data explorer if data uploaded, else to planner."""
       if state.get("uploaded_data"):
           return "data_explorer"
       return "planner"
   
   # In graph setup
   graph.add_conditional_edges(
       "intake",
       route_after_intake,
       {"data_explorer": "data_explorer", "planner": "planner"}
   )
   graph.add_edge("data_explorer", "planner")
   ```

4. **Create state schema** (`src/state/schema.py`)
   ```python
   class WorkflowState(TypedDict):
       # Intake form data
       form_data: dict | None  # Raw form submission
       project_title: str
       original_query: str  # research_question from form
       target_journal: str | None
       paper_type: str | None
       research_type: str | None
       user_hypothesis: str | None
       
       # Data context (from intake)
       uploaded_data: list[DataFile] | None
       data_context: str | None
       data_exploration_results: DataExplorationResult | None
       key_variables: list[str] | None
       
       # Research context (from intake)
       proposed_methodology: str | None
       seed_literature: list[str] | None
       expected_contribution: str | None
       deadline: date | None
       constraints: str | None
       
       # Workflow state
       messages: Annotated[list[AnyMessage], add_messages]
       research_plan: ResearchPlan | None
       search_results: list[SearchResult]
       analysis: AnalysisResult | None
       draft: ResearchDraft | None
       critique: Critique | None
       status: ResearchStatus
       iteration_count: int
       errors: list[WorkflowError]
   ```

5. **Define Pydantic models** (`src/state/models.py`)
   - `IntakeFormData` (all form fields with validation)
   - `DataFile` (filename, content_type, size, path)
   - `DataExplorationResult` (schema, stats, variable mapping, quality issues)
   - `ResearchPlan` (sub-questions, search queries, methodology)
   - `SearchResult` (title, url, snippet, relevance, metadata)
   - `AnalysisResult` (findings, themes, contradictions, gaps)
   - `ResearchDraft` (sections, citations, metadata)
   - `Critique` (items, scores, recommendation)
   - `EvidenceItem` (claim, source, locator, confidence)

6. **Create enums and constants** (`src/state/enums.py`)
   - `ResearchStatus` (intake_pending, intake_complete, data_exploring, planning, searching, analyzing, writing, reviewing, complete, failed)
   - `CritiqueSeverity` (critical, major, minor, suggestion)
   - `EvidenceStrength` (strong, moderate, weak, gap)
   - `PaperType` (research_article, review, perspective, commentary, case_study)
   - `ResearchType` (empirical, theoretical, meta_analysis, case_study, mixed_methods)

7. **Implement data analysis tools** (`src/tools/data_exploration.py`)
   - `parse_csv_file` - Load and analyze CSV data
   - `parse_excel_file` - Load and analyze Excel data
   - `detect_schema` - Infer column types and relationships
   - `generate_summary_stats` - Descriptive statistics
   - `detect_missing_values` - Identify data gaps
   - `detect_outliers` - Flag anomalous values
   - `assess_data_quality` - Overall quality score

#### Acceptance Criteria
- [x] All intake form fields map to workflow state
- [x] Uploaded data files are parsed and analyzed
- [x] Variable mapping matches user's key variables to data columns
- [x] Data quality issues trigger HITL interruption
- [x] Feasibility assessment considers research question vs available data
- [x] Workflow cannot proceed to LITERATURE_REVIEWER without valid intake data
- [x] All state models have JSON schema validation
- [x] State serialization works with checkpointer
- [x] Unit tests for intake parsing and data exploration

---

### Sprint 2: LITERATURE_REVIEWER Node (Complete)
**Duration:** Done  
**Status:** Complete

**Goal:** Systematic literature search BEFORE research planning

This is the critical academic step that must precede methodology design. The literature review:
- Maps the existing knowledge landscape
- Identifies seminal works and key authors
- Reveals methodological precedents
- Surfaces debates and contradictions
- Provides the foundation for gap identification

#### LangGraph Capabilities Used
- **Send API** for parallel search across multiple databases
- `RetryPolicy` for handling API rate limits
- `CachePolicy` for caching literature search results
- `ToolNode` with academic search tools
- `InjectedState` for tools accessing research context

#### Tasks

1. **Create LITERATURE_REVIEWER node** (`src/nodes/literature_reviewer.py`)
   ```python
   from langgraph.types import Send
   
   def literature_reviewer_node(state: WorkflowState) -> dict:
       """Conduct systematic literature review before planning."""
       
       # Generate search queries from initial research question
       search_queries = generate_literature_queries(
           research_question=state["original_query"],
           seed_literature=state.get("seed_literature", []),
           key_variables=state.get("key_variables", [])
       )
       
       # Search will be fanned out to parallel searches
       return {
           "literature_queries": search_queries,
           "status": ResearchStatus.LITERATURE_REVIEW
       }
   
   def route_to_parallel_lit_searches(state: WorkflowState):
       """Fan out to parallel database searches."""
       return [
           Send("search_database", {
               "query": q.query,
               "database": q.target_database,
               "query_id": i
           })
           for i, q in enumerate(state["literature_queries"])
       ]
   ```

2. **Implement academic search tools** (`src/tools/academic_search.py`)
   - `semantic_scholar_search` - Peer-reviewed papers with citation graphs
   - `arxiv_search` - Preprints and cutting-edge research
   - `google_scholar_search` - Broad academic coverage
   - `pubmed_search` - Biomedical literature (if applicable)
   - `ssrn_search` - Working papers in social sciences

3. **Create literature synthesis** (`src/nodes/literature_synthesizer.py`)
   ```python
   def synthesize_literature(state: WorkflowState) -> dict:
       """Synthesize findings from parallel literature searches."""
       
       synthesis = LiteratureSynthesis(
           key_themes=[],
           seminal_works=[],
           methodology_precedents=[],
           debates_and_contradictions=[],
           author_network=[],
           temporal_trends=[]
       )
       
       for result in state["literature_search_results"]:
           # Extract themes
           themes = extract_themes(result)
           synthesis.key_themes.extend(themes)
           
           # Identify highly-cited seminal works
           if result.citation_count > SEMINAL_THRESHOLD:
               synthesis.seminal_works.append(result)
           
           # Track methodologies used
           if result.methodology:
               synthesis.methodology_precedents.append({
                   "method": result.methodology,
                   "source": result.citation,
                   "context": result.methodology_context
               })
       
       return {"literature_synthesis": synthesis}
   ```

4. **Implement citation graph analysis** (`src/tools/citation_analysis.py`)
   - `get_citing_papers` - Who cites this work (forward citations)
   - `get_references` - What this work cites (backward citations)
   - `identify_citation_clusters` - Find related work groups
   - `find_review_papers` - Locate existing reviews on topic

5. **Add literature quality assessment**
   - Journal impact metrics
   - Citation velocity
   - Recency weighting
   - Methodological rigor indicators

#### Acceptance Criteria
- [x] Searches execute in parallel across multiple databases
- [x] Results include citation counts and links
- [x] Seminal works (high citation) are flagged
- [x] Methodology precedents are extracted
- [x] Key themes are identified and clustered
- [x] Contradictions between papers are detected
- [x] Literature synthesis provides foundation for gap analysis

---

### Sprint 3: GAP_IDENTIFIER Node (Complete)
**Duration:** Done  
**Status:** Complete

**Goal:** Identify research gaps and refine the research question

The gap identifier is where the real contribution emerges. It:
- Compares what the literature covers vs. what the user asked
- Identifies methodological gaps (new methods needed)
- Spots empirical gaps (untested contexts, populations)
- Finds theoretical gaps (unexplained phenomena)
- Refines the research question to target a specific gap

#### LangGraph Capabilities Used
- `interrupt()` for human approval of refined research question
- `Command(resume=)` for accepting/modifying the refined question
- Conditional edges for iterative refinement
- `ToolNode` for gap analysis tools

#### Tasks

1. **Create GAP_IDENTIFIER node** (`src/nodes/gap_identifier.py`)
   ```python
   from langgraph.types import interrupt
   
   def gap_identifier_node(state: WorkflowState) -> dict:
       """Identify gaps and refine research question."""
       
       # Analyze what literature covers vs. what user asked
       gap_analysis = identify_gaps(
           original_question=state["original_query"],
           literature_synthesis=state["literature_synthesis"],
           user_data=state.get("data_exploration_results")
       )
       
       # Generate refined research question targeting a gap
       refined_question = refine_research_question(
           original=state["original_query"],
           gaps=gap_analysis.gaps,
           literature_context=state["literature_synthesis"]
       )
       
       # Generate contribution statement
       contribution = generate_contribution_statement(
           refined_question=refined_question,
           gap=gap_analysis.primary_gap,
           literature_context=state["literature_synthesis"]
       )
       
       # Human approves refined question and contribution
       approved = interrupt({
           "action": "approve_refined_question",
           "original_question": state["original_query"],
           "refined_question": refined_question,
           "gap_analysis": gap_analysis.model_dump(),
           "contribution_statement": contribution,
           "message": "Review the refined research question and contribution statement"
       })
       
       return {
           "refined_query": approved["refined_question"],
           "gap_analysis": gap_analysis,
           "contribution_statement": approved["contribution"],
           "status": ResearchStatus.GAP_IDENTIFIED
       }
   ```

2. **Implement gap analysis tools** (`src/tools/gap_analysis.py`)
   - `compare_coverage` - What's covered vs. what's asked
   - `identify_methodological_gaps` - New methods needed
   - `identify_empirical_gaps` - Untested contexts
   - `identify_theoretical_gaps` - Unexplained phenomena
   - `assess_gap_significance` - Is this gap worth filling?

3. **Create contribution generator** (`src/tools/contribution.py`)
   - `generate_contribution_statement` - Clear articulation of what's new
   - `position_in_literature` - How this fits with existing work
   - `differentiate_from_prior` - What's different from similar papers

#### Acceptance Criteria
- [x] Gaps are categorized by type (methodological, empirical, theoretical)
- [x] Refined research question is more specific than original
- [x] Contribution statement clearly articulates what's new
- [x] Human can approve or modify the refined question
- [x] Gap analysis considers available user data

---

### Sprint 4: PLANNER Node (Complete)
**Duration:** 3-4 days  
**Status:** ✅ Complete

**Goal:** Design methodology AFTER literature review and gap identification

Now that we know:
1. What the literature says (LITERATURE_REVIEWER)
2. What gaps exist (GAP_IDENTIFIER)
3. What data the user has (INTAKE)

We can properly design a methodology that:
- Addresses the identified gap
- Uses appropriate methods from the literature
- Is feasible given the available data
- Follows best practices for the research type

#### LangGraph Capabilities Used
- `interrupt()` for human approval of research plan
- `Command(resume=)` for incorporating feedback
- `ToolNode` for methodology tools
- Conditional edges for research type routing

#### Implementation Details

##### Files Created
- `src/nodes/planner.py` - Main PLANNER node implementation
- `src/tools/methodology.py` - Methodology selection and validation tools
- `src/tools/analysis_design.py` - Analysis design and paper structure tools
- `tests/unit/test_planner.py` - Comprehensive test suite (34 tests)

##### State Schema Updates
- Added `MethodologyType` enum (25+ methodology types)
- Added `AnalysisApproach` enum (15+ analysis approaches)
- Added `PlanApprovalStatus` enum (PENDING, APPROVED, REVISION_REQUESTED, REJECTED)
- Enhanced `ResearchPlan` model with methodology, analysis, and approval fields
- Added `gap_analysis`, `refined_research_question`, `contribution` to WorkflowState

##### Methodology Selection Logic
```python
# Maps research type → appropriate methodologies
RESEARCH_TYPE_METHODOLOGIES = {
    "empirical": [PANEL_DATA, EVENT_STUDY, REGRESSION_ANALYSIS, ...],
    "theoretical": [ANALYTICAL_MODEL, SIMULATION, CONCEPTUAL_FRAMEWORK],
    "mixed": [SEQUENTIAL_MIXED, CONCURRENT_MIXED],
    ...
}

# Maps gap type → preferred methodologies
GAP_TYPE_METHODOLOGIES = {
    "methodological": [INSTRUMENTAL_VARIABLES, DIFFERENCE_IN_DIFFERENCES, ...],
    "empirical": [PANEL_DATA, REGRESSION_ANALYSIS, ...],
    "theoretical": [ANALYTICAL_MODEL, CONCEPTUAL_FRAMEWORK],
    ...
}
```

##### HITL Integration
The PLANNER node uses LangGraph's `interrupt()` for human approval:
```python
from langgraph.types import interrupt

# Present plan for approval
approval_response = interrupt({
    "action": "approve_research_plan",
    "plan_summary": plan_summary,
    "methodology": plan.methodology,
    "methodology_type": plan.methodology_type.value if plan.methodology_type else None,
    "gap_addressed": gap_info.get("gap_title", ""),
    ...
})
```

##### Routing After PLANNER
```python
def route_after_planner(state: WorkflowState) -> str:
    """Route to data_analyst or conceptual_synthesizer."""
    plan = state.get("research_plan")
    has_data = state.get("data_exploration_results") is not None
    
    # Theoretical research → conceptual_synthesizer
    if methodology_type in THEORETICAL_METHODOLOGIES:
        return "conceptual_synthesizer"
    
    # Has data → data_analyst
    if has_data:
        return "data_analyst"
    
    return "conceptual_synthesizer"
```

#### Tasks (Completed)

1. **Create PLANNER node** (`src/nodes/planner.py`) ✅
   - Extracts context from gap analysis, literature synthesis, and data exploration
   - Selects methodology based on research type and gap type
   - Validates methodology fit against research context
   - Assesses feasibility with available resources
   - Designs analysis approach (quantitative/qualitative)
   - Determines paper sections and expected outputs
   - Defines measurable success criteria
   - Implements HITL approval via interrupt()

2. **Implement methodology selection** (`src/tools/methodology.py`) ✅
   - `select_methodology` - Choose based on research type, gap type, and data availability
   - `validate_methodology_fit` - Validate methodology against research context
   - `assess_feasibility` - Assess feasibility with available data and time
   - `explain_methodology_choice` - Generate detailed justification with citations

3. **Create analysis design** (`src/tools/analysis_design.py`) ✅
   - `design_quantitative_analysis` - Generate quantitative analysis plan with statistical tests
   - `design_qualitative_analysis` - Design qualitative coding and analysis approach
   - `design_mixed_methods` - Integration strategy for mixed methods research
   - `map_variables_to_analysis` - Map available variables to analysis roles
   - `determine_paper_sections` - Determine appropriate paper sections by type
   - `define_success_criteria` - Generate measurable success criteria

4. **Update workflow graph** (`studio/graphs.py`) ✅
   - Added planner_node to graph
   - Updated routing from GAP_IDENTIFIER to PLANNER
   - Added conditional edge after PLANNER for routing

5. **Write comprehensive tests** (`tests/unit/test_planner.py`) ✅
   - 34 tests covering all components
   - Tests for methodology selection, validation, feasibility
   - Tests for analysis design, paper sections, success criteria
   - Tests for PLANNER node helpers and routing
   - Tests for plan approval processing
   - Edge case and error handling tests

#### Acceptance Criteria
- [x] Methodology is justified by literature precedents
- [x] Analysis approach matches available data
- [x] Plan addresses the identified gap
- [x] Success criteria are measurable
- [x] Human can approve or modify the plan

---

### Sprint 5: DATA_ANALYST and CONCEPTUAL_SYNTHESIZER Nodes
**Duration:** 5-6 days  
**Goal:** Execute analysis based on research type (empirical vs theoretical)

After planning, the workflow routes to one of two nodes based on research type:
- **DATA_ANALYST** for empirical research with data
- **CONCEPTUAL_SYNTHESIZER** for theoretical/conceptual research

#### LangGraph Capabilities Used
- **Send API** for parallel analysis tasks
- `InjectedState` for tools accessing data and literature
- `InjectedStore` for persisting findings
- `RetryPolicy` for robust computation
- `CachePolicy` for caching expensive analyses
- Conditional edges for research type routing

#### Tasks

1. **Create research type router** (`src/nodes/analysis_router.py`)
   ```python
   def route_by_research_type(state: WorkflowState) -> str:
       """Route to appropriate analysis node based on research type."""
       research_type = state.get("research_type", "theoretical")
       has_data = state.get("uploaded_data") is not None
       
       if research_type in ["empirical", "mixed_methods"] and has_data:
           return "data_analyst"
       else:
           return "conceptual_synthesizer"
   ```

2. **Create DATA_ANALYST node** (`src/nodes/data_analyst.py`)
   ```python
   def data_analyst_node(state: WorkflowState) -> dict:
       """Execute data analysis per research plan."""
       
       plan = state["research_plan"]
       data = state["data_exploration_results"]
       
       # Execute analysis per methodology
       analysis_results = execute_analysis(
           methodology=plan.methodology,
           analysis_approach=plan.analysis_approach,
           data=data,
           variables=state.get("key_variables")
       )
       
       # Generate findings with statistical backing
       findings = generate_findings(
           results=analysis_results,
           research_question=state["refined_query"]
       )
       
       # Assess whether findings address the gap
       gap_addressed = assess_gap_coverage(
           findings=findings,
           gap=state["gap_analysis"].primary_gap
       )
       
       return {
           "analysis_results": analysis_results,
           "findings": findings,
           "gap_addressed": gap_addressed,
           "status": ResearchStatus.ANALYSIS_COMPLETE
       }
   ```

3. **Create CONCEPTUAL_SYNTHESIZER node** (`src/nodes/conceptual_synthesizer.py`)
   ```python
   def conceptual_synthesizer_node(state: WorkflowState) -> dict:
       """Build theoretical framework from literature."""
       
       # Synthesize concepts from literature
       framework = build_conceptual_framework(
           literature=state["literature_synthesis"],
           gap=state["gap_analysis"],
           research_question=state["refined_query"]
       )
       
       # Generate propositions or theoretical contributions
       propositions = generate_propositions(
           framework=framework,
           gap=state["gap_analysis"].primary_gap
       )
       
       # Link framework to existing theory
       theoretical_grounding = ground_in_theory(
           framework=framework,
           seminal_works=state["literature_synthesis"].seminal_works
       )
       
       return {
           "conceptual_framework": framework,
           "propositions": propositions,
           "theoretical_grounding": theoretical_grounding,
           "status": ResearchStatus.SYNTHESIS_COMPLETE
       }
   ```

4. **Implement analysis tools** (`src/tools/analysis.py`)
   - `execute_descriptive_stats` - Summary statistics
   - `execute_inferential_stats` - Hypothesis testing
   - `execute_regression` - Regression models
   - `execute_correlation` - Correlation analysis
   - `generate_visualizations` - Charts and plots

5. **Implement synthesis tools** (`src/tools/synthesis.py`)
   - `build_conceptual_framework` - Theory construction
   - `generate_propositions` - Testable statements
   - `identify_theoretical_mechanisms` - Causal pathways
   - `map_concept_relationships` - Concept mapping

#### Acceptance Criteria
- [ ] Research type routing works correctly
- [ ] Data analysis follows methodology from plan
- [ ] Findings link to research question
- [ ] Conceptual framework is grounded in literature
- [ ] Propositions are logically derived
- [ ] Gap coverage is assessed

---

### Sprint 6: WRITER Node
**Duration:** 5-6 days  
**Goal:** Construct the academic argument through section-by-section writing

Writing in academic research is not mere reporting; it is argument construction. Each section serves a rhetorical purpose:
- **Introduction**: Establish the gap and promise the contribution
- **Literature Review**: Position within existing scholarship
- **Methods**: Justify analytical approach
- **Results/Findings**: Present evidence
- **Discussion**: Interpret and connect to literature
- **Conclusion**: Restate contribution and implications

#### LangGraph Capabilities Used
- **Send API** for parallel section writing (where independent)
- **Subgraphs** for section writer encapsulation
- `stream_mode="messages"` for real-time token streaming
- `StreamWriter` for progress events

#### Tasks

1. **Create WRITER node** (`src/nodes/writer.py`)
   ```python
   from langgraph.types import Send
   
   def writer_node(state: WorkflowState) -> dict:
       """Orchestrate section writing based on paper type."""
       
       sections = determine_sections(
           paper_type=state.get("paper_type"),
           research_type=state.get("research_type"),
           methodology=state["research_plan"].methodology
       )
       
       # Some sections must be sequential (intro before discussion)
       # Others can be parallel (methods and results)
       return {
           "sections_to_write": sections,
           "status": ResearchStatus.WRITING
       }
   
   def route_to_section_writers(state: WorkflowState):
       """Fan out to parallel section writers where possible."""
       parallel_sections = ["methods", "results", "related_work"]
       return [
           Send(f"write_{s}", state) 
           for s in state["sections_to_write"] 
           if s in parallel_sections
       ]
   ```

2. **Implement section writers** (`src/writers/`)
   - `introduction_writer.py` - Hook, context, gap, contribution, roadmap
   - `literature_review_writer.py` - Thematic synthesis, not annotated bibliography
   - `methods_writer.py` - Procedure, justification, limitations
   - `results_writer.py` - Findings constrained by data (no interpretation)
   - `discussion_writer.py` - Interpret findings, connect to literature, limitations
   - `conclusion_writer.py` - Contribution summary, implications, future work
   - `abstract_writer.py` - Written LAST; summarizes the paper

3. **Create argument structure manager** (`src/writers/argument.py`)
   - `build_argument_thread` - Ensure logical flow across sections
   - `verify_claim_support` - Every claim backed by evidence
   - `check_contribution_delivery` - Does paper deliver promised contribution?

4. **Implement citation integration** (`src/citations/`)
   - Inline citations from literature synthesis
   - Reference list generation (APA, Chicago, etc.)
   - Citation verification against sources

5. **Add style enforcement** (`src/style/`)
   - Load rules from `docs/writing_style_guide.md`
   - `BannedWordsFilter` - Filter 100+ banned marketing/buzzwords
   - `AcademicToneChecker` - Verify formal academic voice
   - `JournalStyleMatcher` - Match target journal conventions (RFS, JFE, JF, JFQA)
   - `HedgingLanguageChecker` - Ensure appropriate uncertainty language
   - `PrecisionChecker` - Flag vague terms ("many", "various", "some")

   ```python
   from src.style import StyleEnforcer
   
   enforcer = StyleEnforcer.from_guide("docs/writing_style_guide.md")
   
   # Check draft against style guide
   violations = enforcer.check(draft)
   # Returns: [{"type": "banned_word", "word": "utilize", "suggestion": "use", "location": ...}]
   ```

#### Acceptance Criteria
- [ ] Each section serves its rhetorical purpose
- [ ] Claims are backed by evidence or citations
- [ ] Argument flows logically across sections
- [ ] Contribution promised in intro is delivered in conclusion
- [ ] Banned words are filtered per `docs/writing_style_guide.md`
- [ ] Style matches target journal (RFS, JFE, JF, JFQA conventions)
- [ ] Citations follow Chicago Author-Date format
- [ ] Hedging language used appropriately for claims

---

### Sprint 7: REVIEWER Node and Revision Loop
**Duration:** 4-5 days  
**Goal:** Critical self-review before submission

The reviewer simulates a journal referee, applying critical evaluation:
- Does the paper make a clear contribution?
- Is the methodology appropriate?
- Are claims supported by evidence?
- Is the argument coherent?
- Does it meet journal standards?

#### LangGraph Capabilities Used
- `interrupt()` for final human approval
- `interrupt_after=["reviewer"]` for post-review inspection
- `Command(goto=)` for routing to revision or output
- `update_state()` for human edits
- Cycle limits for revision iterations

#### Tasks

1. **Create REVIEWER node** (`src/nodes/reviewer.py`)
   ```python
   from langgraph.types import interrupt
   
   def reviewer_node(state: WorkflowState) -> dict:
       """Critically evaluate draft against academic standards."""
       
       critique = evaluate_manuscript(
           draft=state["draft"],
           research_plan=state["research_plan"],
           gap_analysis=state["gap_analysis"],
           contribution=state["contribution_statement"],
           target_journal=state.get("target_journal")
       )
       
       # Score against thresholds
       score = calculate_quality_score(critique)
       
       if score >= APPROVAL_THRESHOLD:
           decision = "approve"
       elif score >= REVISION_THRESHOLD:
           decision = "revise"
       else:
           decision = "major_revision"
       
       # Human reviews critique and decision
       approved = interrupt({
           "action": "review_critique",
           "critique": critique.model_dump(),
           "score": score,
           "decision": decision,
           "message": "Review the critique and approve or request changes"
       })
       
       return {
           "critique": critique,
           "review_score": score,
           "review_decision": approved["decision"],
           "status": ResearchStatus.REVIEWED
       }
   ```

2. **Implement review criteria** (`src/review/criteria.py`)
   - `evaluate_contribution_clarity` - Is the contribution clear?
   - `evaluate_methodology_rigor` - Is the method appropriate?
   - `evaluate_evidence_quality` - Are claims supported?
   - `evaluate_argument_coherence` - Does the logic flow?
   - `evaluate_writing_quality` - Is it well-written?
   - `evaluate_journal_fit` - Does it meet target journal norms?
   - `evaluate_style_compliance` - Does it follow `docs/writing_style_guide.md`?

3. **Implement style validation** (`src/review/style_validator.py`)
   ```python
   def validate_against_style_guide(draft: ResearchDraft) -> StyleReport:
       """Validate draft against writing_style_guide.md."""
       guide = load_style_guide("docs/writing_style_guide.md")
       
       return StyleReport(
           banned_word_violations=check_banned_words(draft, guide.banned_words),
           citation_format_errors=check_citations(draft, guide.citation_style),
           tone_issues=check_academic_tone(draft, guide.tone_rules),
           formatting_issues=check_formatting(draft, guide.format_rules),
           journal_fit=assess_journal_fit(draft, guide.journal_requirements)
       )
   ```

3. **Create revision router** (`src/nodes/revision_router.py`)
   ```python
   def route_after_review(state: WorkflowState) -> str:
       """Route based on review decision."""
       decision = state["review_decision"]
       iterations = state.get("revision_count", 0)
       
       if decision == "approve":
           return "output"
       elif iterations >= MAX_REVISIONS:
           return "output"  # Output with caveats
       else:
           return "writer"  # Back to revision
   ```

4. **Implement revision tracking**
   - Track what changed between versions
   - Ensure critique issues are addressed
   - Limit revision cycles (max 3)

#### Acceptance Criteria
- [ ] Critique identifies specific, actionable issues
- [ ] Scoring follows defined thresholds
- [ ] Human can override decisions
- [ ] Revision loop addresses critique issues
- [ ] Max iterations prevent infinite loops

---

### Sprint 8: Graph Assembly and Full Workflow Integration
**Duration:** 4-5 days  
**Goal:** Wire all nodes into complete academic research workflow

This sprint assembles all nodes into the proper academic research sequence:
1. INTAKE → 2. LITERATURE_REVIEWER → 3. GAP_IDENTIFIER → 4. PLANNER → 5. DATA_ANALYST/CONCEPTUAL_SYNTHESIZER → 6. WRITER → 7. REVIEWER → OUTPUT

#### LangGraph Capabilities Used
- Full `StateGraph` assembly with all nodes
- `add_conditional_edges` for research type routing
- `compile()` with checkpointer, store, interrupt configuration
- `get_state()` / `get_state_history()` for debugging
- `stream()` with multiple modes for UI
- **Subgraphs** for modular composition
- `interrupt_before` / `interrupt_after` for HITL gates

#### Tasks

1. **Build main graph** (`src/graphs/research_workflow.py`)
   ```python
   from langgraph.graph import StateGraph, START, END
   from langgraph.checkpoint.sqlite import SqliteSaver
   from langgraph.store.memory import InMemoryStore
   from langgraph.types import RetryPolicy, CachePolicy
   
   graph = StateGraph(WorkflowState)
   
   # Add all nodes in academic research sequence
   graph.add_node("intake", intake_node)
   graph.add_node("literature_reviewer", literature_reviewer_node,
                  retry_policy=RetryPolicy(max_attempts=3))
   graph.add_node("lit_synthesizer", synthesize_literature)
   graph.add_node("gap_identifier", gap_identifier_node)
   graph.add_node("planner", planner_node)
   graph.add_node("data_analyst", data_analyst_node,
                  cache_policy=CachePolicy(ttl=3600))
   graph.add_node("conceptual_synthesizer", conceptual_synthesizer_node)
   graph.add_node("writer", writer_node)
   graph.add_node("reviewer", reviewer_node)
   graph.add_node("output", output_node)
   
   # Academic research sequence edges
   graph.add_edge(START, "intake")
   graph.add_edge("intake", "literature_reviewer")
   graph.add_edge("literature_reviewer", "lit_synthesizer")
   graph.add_edge("lit_synthesizer", "gap_identifier")
   graph.add_edge("gap_identifier", "planner")
   
   # Research type routing after planner
   graph.add_conditional_edges(
       "planner",
       route_by_research_type,
       {
           "data_analyst": "data_analyst",
           "conceptual_synthesizer": "conceptual_synthesizer"
       }
   )
   
   # Both analysis paths lead to writer
   graph.add_edge("data_analyst", "writer")
   graph.add_edge("conceptual_synthesizer", "writer")
   
   # Review loop
   graph.add_edge("writer", "reviewer")
   graph.add_conditional_edges(
       "reviewer",
       route_after_review,
       {
           "approve": "output",
           "revise": "writer"
       }
   )
   graph.add_edge("output", END)
   
   # Compile with HITL gates at critical points
   checkpointer = SqliteSaver.from_conn_string("sqlite:///research.db")
   store = InMemoryStore()
   
   app = graph.compile(
       checkpointer=checkpointer,
       store=store,
       interrupt_before=["gap_identifier", "planner"],  # Review refined question and plan
       interrupt_after=["reviewer"]  # Final approval gate
   )
   ```

2. **Implement streaming for UI** (`src/graphs/streaming.py`)
   ```python
   async def stream_research_workflow(
       form_data: dict,
       thread_id: str
   ) -> AsyncGenerator[dict, None]:
       """Stream workflow progress to UI."""
       
       config = {"configurable": {"thread_id": thread_id}}
       
       async for event in app.astream(
           {"form_data": form_data},
           config=config,
           stream_mode=["updates", "messages", "custom"]
       ):
           mode, data = event[0], event[1]
           
           if mode == "messages":
               yield {"type": "token", "content": data}
           elif mode == "custom":
               yield {"type": "progress", "stage": data}
           elif mode == "updates":
               yield {"type": "state", "node": data}
   ```

3. **Add time travel and debugging** (`src/graphs/debug.py`)
   ```python
   def inspect_workflow_state(thread_id: str):
       """Inspect current and historical state."""
       config = {"configurable": {"thread_id": thread_id}}
       
       # Current state
       current = app.get_state(config)
       print(f"Current node: {current.next}")
       print(f"Status: {current.values.get('status')}")
       
       # History
       for snapshot in app.get_state_history(config):
           print(f"{snapshot.created_at}: {snapshot.next}")
   
   def replay_from_checkpoint(thread_id: str, checkpoint_index: int):
       """Replay workflow from specific checkpoint."""
       config = {"configurable": {"thread_id": thread_id}}
       history = list(app.get_state_history(config))
       
       if checkpoint_index < len(history):
           old_config = history[checkpoint_index].config
           return app.invoke(None, old_config)
   ```

4. **Update Studio graphs** (`studio/graphs.py`)
   ```python
   # Export for LangGraph Studio
   from src.graphs.research_workflow import create_research_workflow
   
   # Studio manages its own persistence
   research_graph = create_research_workflow(
       checkpointer=None,
       store=None
   )
   ```

#### Acceptance Criteria
- [ ] Full workflow executes: INTAKE → LIT_REVIEW → GAP → PLAN → ANALYZE → WRITE → REVIEW → OUTPUT
- [ ] Research type routing works (empirical vs theoretical)
- [ ] HITL gates pause at gap identification, planning, and final review
- [ ] Workflow is resumable from any checkpoint
- [ ] Streaming provides real-time progress to UI
- [ ] Time travel debugging works

---

### Sprint 9: Error Handling and Fallbacks
**Duration:** 2-3 days  
**Goal:** Implement robust error handling and graceful degradation

#### LangGraph Capabilities Used
- `RetryPolicy` with exponential backoff and jitter
- `handle_tool_errors` in ToolNode for graceful tool failures
- `Command(goto="fallback")` for explicit error routing
- `StateSnapshot.interrupts` for inspecting failed states
- Conditional edges to fallback node on error states

#### Tasks

1. **Create error handling with RetryPolicy** (`src/errors/`)
   ```python
   from langgraph.types import RetryPolicy
   
   # Different policies for different error types
   api_retry = RetryPolicy(
       max_attempts=3,
       initial_interval=1.0,
       backoff_factor=2.0,
       jitter=True,
       retry_on=(RateLimitError, TimeoutError)
   )
   
   # Apply to nodes
   graph.add_node("literature_reviewer", lit_reviewer_node, retry_policy=api_retry)
   ```

2. **Implement ToolNode error handling**
   ```python
   from langgraph.prebuilt import ToolNode
   
   def handle_search_errors(error: Exception) -> str:
       if isinstance(error, RateLimitError):
           return "Search rate limited. Try again later."
       return f"Search failed: {str(error)}"
   
   tool_node = ToolNode(
       tools=[semantic_scholar_search, arxiv_search],
       handle_tool_errors=handle_search_errors
   )
   ```

3. **Implement fallback node** (`src/nodes/fallback.py`)
   - Generate partial output when workflow fails
   - Include available findings even if incomplete
   - Document what could not be completed
   - Provide recovery suggestions

4. **Add graceful degradation routing**
   ```python
   def route_on_error(state: WorkflowState):
       if state.get("errors") and len(state["errors"]) > 3:
           return "fallback"
       if state.get("status") == ResearchStatus.FAILED:
           return "fallback"
       return "continue"
   ```

#### Acceptance Criteria
- [ ] Rate limits trigger backoff via RetryPolicy
- [ ] Context overflow reduces content automatically
- [ ] Fallback produces usable partial output
- [ ] Errors are logged with full context
- [ ] Tool errors handled gracefully via ToolNode
- [ ] Max 3 retries before fallback activation

---

### Sprint 10: Testing and Evaluation
**Duration:** 3-4 days  
**Goal:** Comprehensive testing and evaluation suite

#### Tasks

1. **Unit tests** (`tests/`)
   - Test each node in isolation
   - Test routing logic
   - Test state serialization
   - Test tool functions

2. **Integration tests** (`tests/integration/`)
   - End-to-end workflow tests
   - HITL simulation tests
   - Persistence/resume tests
   - Error recovery tests

3. **Evaluation suite** (`evaluation/`)
   - Test queries with expected outputs
   - Quality metrics (accuracy, completeness, citation coverage)
   - Regression testing for prompt changes

4. **LangSmith evaluation** 
   - Configure evaluation datasets
   - Set up automatic scoring
   - Track quality over time

#### Acceptance Criteria
- [ ] Greater than 80% unit test coverage
- [ ] Integration tests pass end-to-end
- [ ] Evaluation suite runs automatically
- [ ] LangSmith tracks quality metrics

---

### Sprint 11: Documentation and Polish
**Duration:** 2-3 days  
**Goal:** Complete documentation and production readiness

#### Tasks

1. **Update README** with v2 architecture
2. **API documentation** for all public interfaces
3. **Usage examples** for common workflows
4. **Deployment guide** for production
5. **Performance optimization**
   - Identify bottlenecks
   - Optimize prompt lengths
   - Add caching where appropriate

#### Acceptance Criteria
- [ ] README reflects v2 architecture
- [ ] All public functions have docstrings
- [ ] Examples work out of the box
- [ ] Performance meets targets (less than 5 minutes for simple queries)

---

## Timeline Summary

| Sprint | Node(s) | Duration | Cumulative |
|--------|---------|----------|------------|
| 0: Foundation | Setup | Done | Done |
| 1: Intake | INTAKE | 3-4 days | Week 1 |
| 2: Literature Review | LITERATURE_REVIEWER | 4-5 days | Week 1-2 |
| 3: Gap Analysis | GAP_IDENTIFIER | 3-4 days | Week 2-3 |
| 4: Planning | PLANNER | 3-4 days | Week 3 |
| 5: Analysis | DATA_ANALYST / CONCEPTUAL_SYNTHESIZER | 5-6 days | Week 4-5 |
| 6: Writing | WRITER | 5-6 days | Week 5-6 |
| 7: Review | REVIEWER | 4-5 days | Week 6-7 |
| 8: Assembly | Full Graph | 4-5 days | Week 7-8 |
| 9: Error Handling | Fallbacks | 2-3 days | Week 8 |
| 10: Testing | Evaluation | 3-4 days | Week 9 |
| 11: Documentation | Polish | 2-3 days | Week 9-10 |

**Total Estimated Duration:** 9-10 weeks

---

## Target File Structure

```
src/
├── state/                 # Workflow state and models
│   ├── __init__.py
│   ├── schema.py          # WorkflowState definition
│   ├── models.py          # Pydantic models
│   └── enums.py           # Status enums and constants
├── nodes/                 # LangGraph node implementations
│   ├── __init__.py
│   ├── intake.py          # Form processing, data exploration
│   ├── literature_reviewer.py  # Systematic literature search
│   ├── gap_identifier.py  # Gap analysis, question refinement
│   ├── planner.py         # Methodology design
│   ├── data_analyst.py    # Statistical analysis (empirical)
│   ├── conceptual_synthesizer.py  # Theory building (theoretical)
│   ├── writer.py          # Section writing orchestration
│   ├── reviewer.py        # Critical evaluation
│   ├── output.py          # Final output formatting
│   └── fallback.py        # Graceful degradation
├── tools/                 # LangChain tools
│   ├── __init__.py
│   ├── search.py          # Web search (exists)
│   ├── academic_search.py # Semantic Scholar, arXiv, etc.
│   ├── data_exploration.py # CSV/Excel analysis
│   ├── gap_analysis.py    # Gap identification
│   ├── methodology.py     # Method selection
│   ├── analysis.py        # Statistical tools
│   ├── synthesis.py       # Conceptual framework building
│   └── basic.py           # Utility tools (exists)
├── writers/               # Section-specific writers
│   ├── __init__.py
│   ├── argument.py        # Argument structure manager
│   ├── abstract.py
│   ├── introduction.py
│   ├── literature_review.py
│   ├── methods.py
│   ├── results.py
│   ├── discussion.py
│   └── conclusion.py
├── evidence/              # Evidence tracking
│   ├── __init__.py
│   ├── registry.py        # Evidence storage
│   ├── extraction.py      # Evidence extraction
│   └── validation.py      # Evidence verification
├── citations/             # Citation management
│   ├── __init__.py
│   ├── manager.py         # Citation registry
│   ├── formatter.py       # Citation formatting
│   └── verification.py    # Citation checking
├── review/                # Review criteria
│   ├── __init__.py
│   ├── criteria.py        # Scoring definitions
│   ├── style_validator.py # Style guide validation
│   └── gates.py           # Quality gates
├── graphs/                # LangGraph definitions
│   ├── __init__.py
│   ├── research_workflow.py  # Main workflow
│   ├── routing.py         # Conditional routing
│   ├── streaming.py       # UI streaming
│   ├── debug.py           # Time travel, inspection
│   └── persistence.py     # Checkpoint management
├── errors/                # Error handling
│   ├── __init__.py
│   ├── handlers.py        # Error handlers
│   └── recovery.py        # Recovery strategies
├── style/                 # Style enforcement (loads from docs/writing_style_guide.md)
│   ├── __init__.py
│   ├── enforcer.py        # Main StyleEnforcer class
│   ├── banned_words.py    # Word filter (100+ banned words)
│   ├── tone_checker.py    # Academic tone validation
│   ├── journal_matcher.py # Target journal style matching
│   └── precision_checker.py # Vague term detection
├── memory/                # Persistence (exists)
│   ├── __init__.py
│   ├── checkpointer.py
│   └── store.py
├── config/                # Configuration (exists)
│   ├── __init__.py
│   └── settings.py
├── agents/                # Legacy compatibility (exists)
│   ├── __init__.py
│   ├── base.py
│   └── research.py
└── main.py                # CLI entrypoint (exists)

public/
├── research_intake_form.html  # Entry point HTML form

studio/
├── langgraph.json
└── graphs.py

tests/
├── __init__.py
├── unit/
│   ├── test_state.py
│   ├── test_nodes.py
│   ├── test_tools.py
│   └── test_routing.py
├── integration/
│   ├── test_workflow.py
│   ├── test_persistence.py
│   └── test_hitl.py
└── conftest.py

evaluation/
├── test_queries.json
├── expected_outputs/
└── run_evaluation.py

docs/
├── IMPLEMENTATION_PLAN.md     # This file
├── writing_style_guide.md     # Academic writing standards (REQUIRED)
├── copilot-instructions.md
└── architecture/
    └── langgraph_architecture_spec.md
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Context overflow on large searches | Implement progressive summarization, chunk results |
| Search API rate limits | Exponential backoff, parallel request throttling |
| Low-quality search results | Multi-pass search with query refinement |
| Critique loop never approves | Max iteration limit with fallback output |
| HITL delays workflow | Timeout with default action, async notification |
| Model hallucination | Evidence gate blocks unsupported claims |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| End-to-end completion rate | Greater than 90% |
| Average workflow time (simple query) | Less than 5 minutes |
| Average workflow time (complex query) | Less than 15 minutes |
| Citation accuracy (claims with valid sources) | Greater than 95% |
| Critique pass rate (first attempt) | Greater than 60% |
| Critique pass rate (after revision) | Greater than 85% |
| Test coverage | Greater than 80% |

---

## Dependencies

### Required Packages (in pyproject.toml)
```toml
dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langsmith>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-tavily>=0.1.0",
    "langgraph-checkpoint-sqlite>=3.0.0",
    "tavily-python>=0.5.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
]
```

### Environment Variables
```bash
ANTHROPIC_API_KEY=...      # Required
LANGSMITH_API_KEY=...      # Required
TAVILY_API_KEY=...         # Required
LANGSMITH_TRACING=true     # Recommended
LANGSMITH_PROJECT=gia-agentic-v2
```

---

## Next Steps

1. **Immediate:** Begin Sprint 1 - State Schema and Core Types
2. **This week:** Complete state models and start PLANNER node
3. **Review point:** End of Sprint 3 - Demo search capability
4. **Milestone:** End of Sprint 7 - Full workflow operational

---

*Document maintained by Gia Tenica. Last updated: 2 January 2026.*
