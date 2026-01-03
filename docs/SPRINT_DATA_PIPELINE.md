# Autonomous Data Acquisition & Analysis Pipeline

**Version**: 1.0  
**Date**: 2026-01-03  
**Author**: Gia Tenica  
**Status**: Planning

---

## Executive Summary

This document outlines a 5-sprint implementation plan to transform GIA into a fully autonomous research system capable of:

1. **Intelligent Data Exploration** - Automatically profile and describe any uploaded dataset
2. **External Data Acquisition** - Fetch missing data from 15+ external sources
3. **Dynamic Tool Creation** - Generate and execute custom data acquisition code
4. **Publication-Ready Artifacts** - Generate tables, figures, and LaTeX output
5. **Graceful Degradation** - Human checkpoints when data cannot be acquired

---

## External Data Sources Inventory

### Finance & Economics (API Key Required - Free Tier)

| Source | API | Auth | Free Tier | Data Types |
|--------|-----|------|-----------|------------|
| **FRED** | `api.stlouisfed.org` | API Key | Unlimited | Economic indicators, interest rates, employment |
| **Alpha Vantage** | `alphavantage.co` | API Key | 25 req/day | Stock prices, forex, crypto, fundamentals |
| **Finnhub** | `finnhub.io` | API Key | 60 req/min | Real-time quotes, news, fundamentals |
| **Polygon.io** | `api.polygon.io` | API Key | 5 req/min | US stocks, options, forex |
| **Nasdaq Data Link** | `data.nasdaq.com` | API Key | 300 req/10s | Quandl datasets, commodities |
| **Yahoo Finance** | `yfinance` (Python) | None | Unlimited | Prices, options, financials |
| **CoinGecko** | `api.coingecko.com` | None | 30 req/min | Crypto prices, market data |
| **Open Exchange Rates** | `openexchangerates.org` | API Key | 1000 req/mo | Currency exchange rates |
| **Frankfurter** | `frankfurter.app` | None | Unlimited | ECB exchange rates |

### Science & Research (Free / Open)

| Source | API | Auth | Data Types |
|--------|-----|------|------------|
| **World Bank** | `api.worldbank.org` | None | Development indicators, demographics |
| **UN Data** | `data.un.org` | None | Population, trade, environment |
| **Eurostat** | `ec.europa.eu/eurostat` | None | European statistics |
| **OECD** | `stats.oecd.org` | None | Economic, social, environmental data |
| **NASA** | `api.nasa.gov` | API Key (free) | Earth science, astronomy |
| **PubMed/NCBI** | `eutils.ncbi.nlm.nih.gov` | API Key (free) | Biomedical literature |
| **ClinicalTrials.gov** | `clinicaltrials.gov/api` | None | Clinical trial data |
| **OpenFDA** | `api.fda.gov` | None | Drug, device adverse events |

### General Data (Free)

| Source | API | Auth | Data Types |
|--------|-----|------|------------|
| **REST Countries** | `restcountries.com` | None | Country info, demographics |
| **Open-Meteo** | `open-meteo.com` | None | Weather, climate data |
| **SEC EDGAR** | `sec.gov/cgi-bin/` | None | Company filings, financials |
| **USPTO** | `patentsview.org/api` | None | Patent data |
| **arXiv** | `export.arxiv.org/api` | None | Academic preprints |

---

## Architecture Overview

### Current Flow
```
INTAKE → DATA_EXPLORER → LITERATURE → GAP → PLANNER → DATA_ANALYST → WRITER
                ↓                                          ↓
         (basic profiling)                          (runs analysis)
```

### Target Flow
```
INTAKE → DATA_EXPLORER → LITERATURE → GAP → PLANNER → DATA_ACQUISITION → DATA_ANALYST → WRITER
              ↓                                 ↓              ↓                ↓
     (LLM summarization)              (data requirements)  (fetch APIs)   (tables/figures)
              ↓                                              ↓
     (quality warnings)                              (human interrupt
                                                      if unavailable)
```

### New Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE REGISTRY                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Finance   │  │  Economics  │  │   Health    │  │   Science   │        │
│  │  - yfinance │  │  - FRED     │  │  - PubMed   │  │  - arXiv    │        │
│  │  - Polygon  │  │  - WorldBank│  │  - OpenFDA  │  │  - NASA     │        │
│  │  - Finnhub  │  │  - OECD     │  │  - ClinTrial│  │  - USPTO    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CODE EXECUTION ENGINE                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  exec() with restricted globals (Phase 1)                            │   │
│  │  → Docker sandbox (Phase 2 - before production)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VISUALIZATION ENGINE                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Tables    │  │   Figures   │  │   LaTeX     │  │   Export    │        │
│  │  - Summary  │  │  - Line     │  │  - stargazer│  │  - PNG      │        │
│  │  - Regression│ │  - Scatter  │  │  - threeptab│  │  - PDF      │        │
│  │  - Crosstab │  │  - Heatmap  │  │  - longtable│  │  - HTML     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sprint Breakdown

### Sprint 12: Enhanced Data Explorer (1 week)

**Goal**: Transform data_explorer into an intelligent, LLM-powered data profiler that generates human-readable summaries.

#### Tasks

1. **Extend State Schema** (`src/state/schema.py`, `src/state/models.py`)
   - Add `DataExplorationSummary` model:
     ```python
     class DataExplorationSummary(BaseModel):
         prose_description: str  # 100-200 word LLM-generated description
         dataset_inventory: list[DatasetInfo]  # name, rows, cols, date_range
         quality_flags: list[QualityFlag]  # warnings about data issues
         recommended_variables: list[str]  # variables relevant to research
         data_gaps: list[str]  # what's missing for research question
     ```
   - Add `QualityFlag` enum: `MISSING_VALUES`, `UNREADABLE_FILE`, `ENCODING_ERROR`, `SCHEMA_MISMATCH`, `DATE_PARSING_FAILED`
   - Add state fields: `data_exploration_summary`, `file_read_errors`

2. **Upgrade Data Profiling Tools** (`src/tools/data_profiling.py`)
   - Add `deep_profile_dataset()` - comprehensive statistical profiling
   - Add `detect_data_types()` - intelligent type inference (dates, categories, numerics)
   - Add `assess_data_quality()` - missing values, outliers, duplicates
   - Add `identify_time_series()` - detect if data has temporal structure
   - Add `detect_panel_structure()` - identify entity-time panel data

3. **Add LLM Summarization** (`src/tools/data_profiling.py`)
   - Add `generate_data_prose_summary()` tool:
     - Input: DataExplorationResult, research_question
     - Output: 100-200 word description suitable for Methods section
     - Include: data source, time period, key variables, sample size, limitations

4. **Update Data Explorer Node** (`src/nodes/data_explorer.py`)
   - After loading all datasets, call LLM to generate prose summary
   - Detect and flag quality issues with specific error messages
   - Map uploaded data to research question variables
   - Output `DataExplorationSummary` to state

5. **Add Format Handling** (`src/tools/data_loading.py`)
   - Add encoding detection (chardet/charset-normalizer)
   - Add delimiter sniffing for CSV
   - Add date parsing with multiple format attempts
   - Add compressed file support (gzip, bz2)

#### Deliverables
- [x] `DataExplorationSummary` model in state ✅
- [x] `deep_profile_dataset()` tool ✅
- [x] `generate_data_prose_summary()` tool ✅
- [x] Updated data_explorer node with LLM summarization ✅
- [x] Quality flag system for data issues ✅
- [x] Unit tests for new profiling tools ✅

#### Success Criteria
- Data explorer outputs prose like: "The uploaded datasets contain 46.2 million options contract records spanning 2004-2024 for GOOG, GOOGL, SPY, and VIX. Each dataset includes strike price, expiration date, bid/ask quotes, volume, and implied volatility. GOOG/GOOGL data begins April 2014 (post-stock split). Data quality is high with <0.1% missing values. Key limitation: Pre-2014 GOOG data unavailable for long-horizon analysis."

**Status**: ✅ **COMPLETE** (PR #27 merged 2026-01-03)

---

### Sprint 13: Data Source Registry & External APIs (1.5 weeks)

**Goal**: Build extensible registry of external data sources and implement core acquisition tools.

#### Tasks

1. **Create Data Source Registry** (`src/data_sources/__init__.py`)
   ```python
   class DataSourceRegistry:
       """Extensible registry of external data sources by domain."""
       
       _sources: dict[str, list[DataSource]] = {}
       
       @classmethod
       def register(cls, domain: str, source: DataSource) -> None: ...
       
       @classmethod
       def get_sources(cls, domain: str) -> list[DataSource]: ...
       
       @classmethod
       def find_source(cls, data_type: str) -> DataSource | None: ...
   ```

2. **Define DataSource Protocol** (`src/data_sources/base.py`)
   ```python
   class DataSource(Protocol):
       name: str
       domain: str  # finance, economics, health, science
       requires_api_key: bool
       rate_limit: RateLimit
       data_types: list[str]  # ["stock_prices", "options", "fundamentals"]
       
       async def fetch(self, params: dict) -> pd.DataFrame: ...
       async def check_availability(self, params: dict) -> bool: ...
       def get_required_params(self, data_type: str) -> list[str]: ...
   ```

3. **Implement Finance Sources** (`src/data_sources/finance.py`)
   - `YFinanceSource` - Stock prices, options, fundamentals (no key)
   - `FREDSource` - Economic indicators (free API key)
   - `AlphaVantageSource` - Stock data, forex (free API key)
   - `PolygonSource` - US equities, options (free API key)
   - `CoinGeckoSource` - Crypto prices (no key)

4. **Implement Economics Sources** (`src/data_sources/economics.py`)
   - `WorldBankSource` - Development indicators (no key)
   - `OECDSource` - Economic statistics (no key)
   - `EurostatSource` - European data (no key)
   - `FrankfurterSource` - Exchange rates (no key)

5. **Implement Science Sources** (`src/data_sources/science.py`)
   - `PubMedSource` - Biomedical literature (free key)
   - `ClinicalTrialsSource` - Trial data (no key)
   - `OpenFDASource` - Drug/device data (no key)
   - `SECEdgarSource` - Company filings (no key)

6. **Create External Data Tools** (`src/tools/external_data.py`)
   ```python
   @tool
   def acquire_stock_data(
       ticker: str,
       start_date: str,
       end_date: str,
       frequency: Literal["1d", "1wk", "1mo"] = "1d"
   ) -> str:
       """Fetch historical stock price data via yfinance."""
   
   @tool
   def acquire_economic_indicator(
       series_id: str,
       start_date: str | None = None,
       end_date: str | None = None
   ) -> str:
       """Fetch economic data from FRED."""
   
   @tool
   def acquire_world_bank_data(
       indicator: str,
       countries: list[str],
       start_year: int,
       end_year: int
   ) -> str:
       """Fetch development indicators from World Bank."""
   
   @tool
   def fetch_api_json(
       url: str,
       method: Literal["GET", "POST"] = "GET",
       headers: dict | None = None,
       params: dict | None = None
   ) -> str:
       """Generic REST API fetch for JSON endpoints."""
   
   @tool
   def scrape_html_table(
       url: str,
       table_index: int = 0
   ) -> str:
       """Extract HTML table from webpage into DataFrame."""
   ```

7. **Add Config Settings** (`src/config/settings.py`)
   - `FRED_API_KEY`
   - `ALPHA_VANTAGE_API_KEY`
   - `POLYGON_API_KEY`
   - `FINNHUB_API_KEY`
   - `PUBMED_API_KEY`

#### Deliverables
- [x] `DataSourceRegistry` class with domain registration ✅
- [x] `DataSource` protocol with sync fetch ✅
- [x] 3 finance data sources (YFinance, FRED, CoinGecko) ✅
- [x] `acquire_stock_data()` tool ✅
- [x] `acquire_economic_indicator()` tool ✅
- [x] `acquire_crypto_data()` tool ✅
- [x] `fetch_api_json()` generic tool ✅
- [x] `list_available_data_sources()` discovery tool ✅
- [x] Rate limiting infrastructure ✅
- [x] Unit tests with mocked API responses ✅

#### Success Criteria
- `DataSourceRegistry.find_source("stock_prices")` returns YFinanceSource ✅
- `acquire_stock_data("AAPL", "2020-01-01", "2024-12-31")` returns valid DataFrame ✅
- Tools gracefully handle rate limits and missing API keys ✅

**Status**: ✅ **COMPLETE** (PR #29 merged 2026-01-03)

---

### Sprint 14: Data Acquisition Node & Code Execution (1.5 weeks)

**Goal**: Create intelligent data acquisition agent that can fetch external data and generate custom acquisition code.

#### Tasks

1. **Extend State Schema** (`src/state/schema.py`)
   ```python
   # New state fields
   data_acquisition_plan: DataAcquisitionPlan | None
   acquired_datasets: list[AcquiredDataset]
   acquisition_failures: list[AcquisitionFailure]
   generated_code_snippets: list[CodeSnippet]
   ```

2. **Create Acquisition Models** (`src/state/models.py`)
   ```python
   class DataRequirement(BaseModel):
       variable_name: str  # e.g., "daily_returns"
       data_type: str  # e.g., "stock_prices"
       description: str
       required_fields: list[str]  # e.g., ["close", "volume"]
       time_range: TimeRange | None
       entities: list[str] | None  # e.g., ["AAPL", "GOOG"]
       priority: Literal["required", "preferred", "optional"]
   
   class DataAcquisitionPlan(BaseModel):
       requirements: list[DataRequirement]
       available_in_upload: list[str]  # which requirements met by uploads
       to_acquire: list[DataAcquisitionTask]
       estimated_time: str
   
   class DataAcquisitionTask(BaseModel):
       requirement: DataRequirement
       source: str  # e.g., "yfinance", "FRED"
       tool_to_use: str
       params: dict
   
   class AcquisitionFailure(BaseModel):
       requirement: DataRequirement
       attempted_sources: list[str]
       error_messages: list[str]
       user_action_needed: str
   ```

3. **Create Code Execution Engine** (`src/tools/code_execution.py`)
   ```python
   SAFE_GLOBALS = {
       "pd": pd,
       "np": np,
       "datetime": datetime,
       "requests": requests,
       "json": json,
       "re": re,
       "math": math,
       "statistics": statistics,
       # Explicitly exclude: os, sys, subprocess, eval, exec, open, __import__
   }
   
   @tool
   def execute_python_code(
       code: str,
       description: str,
       timeout_seconds: int = 30
   ) -> str:
       """
       Execute Python code in restricted environment.
       
       Allowed: pandas, numpy, requests, json, datetime, math, statistics
       Forbidden: file I/O, subprocess, imports beyond allowed
       """
       # Validate code doesn't contain forbidden patterns
       # Execute with restricted globals
       # Return result as string
   ```

4. **Create Data Acquisition Node** (`src/nodes/data_acquisition.py`)
   ```python
   async def data_acquisition_node(state: WorkflowState) -> dict:
       """
       Intelligent data acquisition agent.
       
       1. Parse research_plan.data_requirements
       2. Check what's available in loaded_datasets
       3. For gaps: query DataSourceRegistry for appropriate source
       4. Execute acquisition tools
       5. If still missing: generate custom code via LLM
       6. If still failing: raise human interrupt
       """
   ```

5. **Update Planner Node** (`src/nodes/planner.py`)
   - Add `DataRequirement` generation to research plan
   - LLM should specify exactly what data is needed
   - Include time ranges, entities, required fields

6. **Add Human Interrupt Handler** (`src/nodes/data_acquisition.py`)
   ```python
   class DataAcquisitionInterrupt(Exception):
       """Raised when data cannot be acquired autonomously."""
       
       def __init__(
           self,
           missing_data: list[DataRequirement],
           attempted_actions: list[str],
           suggested_user_actions: list[str]
       ): ...
   ```

7. **Update Workflow Graph** (`studio/graphs.py`)
   - Add `data_acquisition` node between `planner` and `data_analyst`
   - Add conditional edge: if acquisition fails → human interrupt
   - Add resume capability after user provides data

#### Deliverables
- [x] `DataAcquisitionPlan` and related models ✅
- [x] `execute_python_code()` tool with safety checks ✅
- [x] `data_acquisition_node()` with routing logic ✅
- [x] Human interrupt routing for unresolvable data gaps ✅
- [x] Updated workflow graph with acquisition node ✅
- [x] Unit tests for code execution (28 tests) ✅
- [x] Unit tests for data acquisition (22 tests) ✅

#### Success Criteria
- ✅ Data requirements parsed from research plan
- ✅ Code execution sandbox blocks dangerous operations
- ✅ Safe modules (pandas, numpy, etc.) work in sandbox
- ✅ Workflow routes through data_acquisition node
- ✅ Routing handles theoretical vs empirical research

**Status**: ✅ **COMPLETE** (PR #31 merged 2026-01-04)

---

### Sprint 15: Visualization & Table Generation (1 week) ✅ COMPLETE

**Goal**: Generate publication-ready tables and figures for academic papers.

**Implementation**: See [SPRINT_15.md](../sprints/SPRINT_15.md) for full details.

#### Tasks

1. **Extend State Schema** (`src/state/models.py`)
   ```python
   class TableArtifact(BaseModel):
       table_id: str
       title: str
       caption: str
       format: Literal["latex", "markdown", "html"]
       content: str
       source_data: str  # dataset name
       notes: str | None
   
   class FigureArtifact(BaseModel):
       figure_id: str
       title: str
       caption: str
       format: Literal["png", "pdf", "svg"]
       content_base64: str
       source_data: str
       notes: str | None
   ```

2. **Create Table Generation Tools** (`src/tools/visualization.py`)
   ```python
   @tool
   def create_summary_statistics_table(
       dataset_name: str,
       variables: list[str],
       statistics: list[str] = ["mean", "std", "min", "max", "n"],
       format: Literal["latex", "markdown"] = "latex",
       title: str = "Summary Statistics"
   ) -> str:
       """Generate publication-style summary statistics table."""
   
   @tool
   def create_regression_table(
       regression_results: list[dict],
       model_names: list[str],
       format: Literal["latex", "markdown"] = "latex",
       include_diagnostics: bool = True
   ) -> str:
       """Generate Stargazer-style regression table."""
   
   @tool
   def create_correlation_matrix_table(
       dataset_name: str,
       variables: list[str],
       format: Literal["latex", "markdown"] = "latex",
       include_significance: bool = True
   ) -> str:
       """Generate correlation matrix with significance stars."""
   
   @tool
   def create_crosstab_table(
       dataset_name: str,
       row_var: str,
       col_var: str,
       values_var: str | None = None,
       aggfunc: str = "count"
   ) -> str:
       """Generate cross-tabulation / pivot table."""
   ```

3. **Create Figure Generation Tools** (`src/tools/visualization.py`)
   ```python
   @tool
   def create_time_series_plot(
       dataset_name: str,
       date_column: str,
       value_columns: list[str],
       title: str,
       ylabel: str,
       figsize: tuple[int, int] = (10, 6)
   ) -> str:
       """Generate time series line plot, return base64 PNG."""
   
   @tool
   def create_scatter_plot(
       dataset_name: str,
       x_column: str,
       y_column: str,
       color_column: str | None = None,
       title: str = "",
       add_regression_line: bool = False
   ) -> str:
       """Generate scatter plot with optional regression line."""
   
   @tool
   def create_distribution_plot(
       dataset_name: str,
       column: str,
       plot_type: Literal["histogram", "density", "box"] = "histogram",
       title: str = ""
   ) -> str:
       """Generate distribution visualization."""
   
   @tool
   def create_heatmap(
       dataset_name: str,
       columns: list[str],
       title: str = "Correlation Heatmap"
   ) -> str:
       """Generate correlation heatmap."""
   ```

4. **Update Data Analyst Node** (`src/nodes/data_analyst.py`)
   - Convert to ReAct agent with visualization tool access
   - After analysis, generate appropriate tables/figures
   - Store artifacts in state: `tables`, `figures`

5. **Update Writer Integration** (`src/writers/results.py`, `src/writers/methods.py`)
   - Results writer embeds table/figure references
   - Methods writer includes data description from explorer
   - Generate LaTeX `\begin{table}` and `\begin{figure}` blocks

6. **Add Export Functionality** (`src/tools/visualization.py`)
   ```python
   @tool
   def export_artifacts_to_files(
       output_dir: str,
       format: Literal["latex", "pdf", "docx"] = "latex"
   ) -> str:
       """Export all tables and figures to files."""
   ```

#### Deliverables
- [x] `TableArtifact` and `FigureArtifact` models ✅
- [x] `create_summary_statistics_table()` tool ✅
- [x] `create_regression_table()` with Stargazer-style output ✅
- [x] `create_time_series_plot()` tool ✅
- [x] `create_scatter_plot()` tool ✅
- [x] `create_distribution_plot()` tool ✅
- [x] Data analyst generates viz artifacts automatically ✅
- [ ] Writer integration with table/figure embedding (Sprint 16)
- [x] Unit tests for all visualization tools (37 tests) ✅

#### Success Criteria
- ✅ Data analyst produces: Table 1 (Summary Stats), Table 2 (Regressions), Table 3 (Correlation)
- ✅ Tables render correctly in LaTeX with significance stars
- ✅ Figures are publication quality (300 DPI, serif fonts)
- ⏳ Writer outputs: "Table 1 presents summary statistics..." (Sprint 16)

---

### Sprint 16: Integration, Testing & Polish (1 week)

**Goal**: End-to-end testing, error handling, and production readiness.

#### Tasks

1. **Integration Tests**
   - Test full workflow: upload → explore → literature → plan → acquire → analyze → write
   - Test with finance data (options pricing)
   - Test with economics data (macro indicators)
   - Test with health data (clinical outcomes)
   - Test acquisition failure → human interrupt → resume

2. **Error Handling**
   - Graceful API failures with retry logic
   - Rate limit handling with exponential backoff
   - Timeout handling for long-running acquisitions
   - Clear error messages for user

3. **Documentation**
   - Update API.md with new tools
   - Update README with data acquisition capabilities
   - Add examples for external data fetching
   - Document supported data sources

4. **Performance Optimization**
   - Async data acquisition where possible
   - Caching for repeated API calls
   - Parallel figure generation
   - Memory optimization for large datasets

5. **Security Review** (pre-production)
   - Audit code execution restrictions
   - Review API key handling
   - Plan Docker sandbox migration path

6. **Demo & Showcase**
   - Create demo video of full workflow
   - Prepare sample research projects
   - Document case studies

#### Deliverables
- [ ] Full integration test suite
- [ ] Error handling for all external APIs
- [ ] Updated documentation
- [ ] Performance benchmarks
- [ ] Security audit checklist
- [ ] Demo materials

---

## Dependency Installation

Add to `pyproject.toml`:

```toml
[project.dependencies]
# Existing...

# Data acquisition
yfinance = ">=0.2.0"
fredapi = ">=0.5.0"
wbgapi = ">=1.0.0"  # World Bank
pandas-datareader = ">=0.10.0"
alpha-vantage = ">=2.3.0"
beautifulsoup4 = ">=4.12.0"
lxml = ">=5.0.0"
aiohttp = ">=3.9.0"
charset-normalizer = ">=3.0.0"

# Visualization
matplotlib = ">=3.8.0"
seaborn = ">=0.13.0"
plotly = ">=5.18.0"  # Interactive figures
kaleido = ">=0.2.0"  # Plotly export

# Tables
tabulate = ">=0.9.0"
stargazer = ">=0.0.7"  # Regression tables
```

---

## Environment Variables

Add to `.env`:

```bash
# Existing...

# External Data APIs
FRED_API_KEY=your_fred_key_here
ALPHA_VANTAGE_API_KEY=your_av_key_here
POLYGON_API_KEY=your_polygon_key_here
FINNHUB_API_KEY=your_finnhub_key_here
PUBMED_API_KEY=your_pubmed_key_here

# Optional
NASDAQ_DATA_LINK_API_KEY=
WORLD_BANK_API_KEY=  # Usually not needed
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API rate limits hit during analysis | High | Medium | Implement caching, backoff, parallel calls |
| Code execution security issues | Medium | High | Restricted globals, Docker sandbox (Phase 2) |
| Data source API changes | Medium | Medium | Abstract behind DataSource protocol |
| Large dataset memory issues | Medium | High | Stream processing, DuckDB optimizations |
| LLM hallucinating data sources | Low | Medium | Validate source existence before suggesting |

---

## Timeline Summary

| Sprint | Duration | Focus |
|--------|----------|-------|
| Sprint 12 | 1 week | Enhanced Data Explorer with LLM summarization |
| Sprint 13 | 1.5 weeks | Data Source Registry & External APIs |
| Sprint 14 | 1.5 weeks | Data Acquisition Node & Code Execution |
| Sprint 15 | 1 week | Visualization & Table Generation |
| Sprint 16 | 1 week | Integration, Testing & Polish |
| **Total** | **6 weeks** | |

---

## Appendix: Code Execution Safety

### Phase 1: Restricted exec() (Current Implementation)

```python
FORBIDDEN_PATTERNS = [
    r'\bimport\s+os\b',
    r'\bimport\s+sys\b', 
    r'\bimport\s+subprocess\b',
    r'\b__import__\b',
    r'\beval\b',
    r'\bexec\b',
    r'\bopen\s*\(',
    r'\bfile\s*\(',
    r'\bcompile\b',
    r'\bglobals\s*\(',
    r'\blocals\s*\(',
    r'\bgetattr\b',
    r'\bsetattr\b',
    r'\bdelattr\b',
]

SAFE_BUILTINS = {
    'True': True,
    'False': False,
    'None': None,
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'print': print,
    'range': range,
    'round': round,
    'set': set,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
}
```

### Phase 2: Docker Sandbox (Pre-Production)

```yaml
# docker-compose.sandbox.yml
services:
  code-executor:
    image: python:3.12-slim
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:size=100M
    mem_limit: 512m
    cpus: 1
    network_mode: none  # No network access
    volumes:
      - ./sandbox_code:/code:ro
```

---

## Appendix: Data Source Examples

### yfinance (Stock Data)
```python
import yfinance as yf

ticker = yf.Ticker("AAPL")
hist = ticker.history(period="max")
options = ticker.options  # Available expiration dates
opt_chain = ticker.option_chain("2024-01-19")
```

### FRED (Economic Data)
```python
from fredapi import Fred

fred = Fred(api_key=FRED_API_KEY)
gdp = fred.get_series("GDP")
unemployment = fred.get_series("UNRATE")
```

### World Bank
```python
import wbgapi as wb

# GDP per capita for all countries, 2010-2020
data = wb.data.DataFrame("NY.GDP.PCAP.CD", time=range(2010, 2021))
```

### PubMed
```python
from Bio import Entrez

Entrez.email = "your@email.com"
handle = Entrez.esearch(db="pubmed", term="machine learning cancer", retmax=100)
record = Entrez.read(handle)
```

---

## Sign-off

- [ ] Architecture reviewed by: _______________
- [ ] Security reviewed by: _______________
- [ ] Sprint planning approved: _______________

**Next Action**: Begin Sprint 12 implementation
