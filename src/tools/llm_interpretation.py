"""LLM-powered interpretation tools for statistical results.

These tools generate academic prose interpretations of statistical analyses,
suitable for inclusion in research papers' methods and results sections.
"""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import settings
from src.state.models import StatisticalResult, RegressionResult


# =============================================================================
# Helper Functions
# =============================================================================


def _get_llm() -> ChatAnthropic:
    """Get a Claude model for interpretation generation."""
    return ChatAnthropic(
        model=settings.default_model,
        temperature=0.3,  # Slight creativity for prose but mostly deterministic
        max_tokens=2000,
        api_key=settings.anthropic_api_key,
    )


def _build_style_instructions() -> str:
    """Build style instructions for academic prose generation."""
    return """
CRITICAL STYLE REQUIREMENTS:
1. NEVER use em dashes (—). Use semicolons, colons, or periods instead.
2. NEVER use emojis.
3. Use precise, formal academic language.
4. Report statistics in APA format: t(df) = X.XX, p = .XXX
5. Include effect sizes with interpretations (small, medium, large).
6. Avoid banned words: delve, groundbreaking, novel, unique, harness, leverage, paradigm, synergy, utilize, impactful.
7. Write in third person passive voice where appropriate.
8. Be concise but complete.

FORMATTING:
- Statistical values: round to 2-3 decimal places
- p-values: report as p < .001 for very small values, otherwise p = .XXX
- Confidence intervals: report as 95% CI [lower, upper]
- Effect sizes: report with interpretation label
"""


# =============================================================================
# Pydantic Input Schemas
# =============================================================================


class InterpretRegressionInput(BaseModel):
    """Input schema for interpret_regression tool."""
    
    model_type: str = Field(description="Type of regression model (OLS, logistic, etc.)")
    dependent_variable: str = Field(description="Name of the dependent variable")
    r_squared: float = Field(description="R-squared value")
    adjusted_r_squared: float = Field(description="Adjusted R-squared value")
    f_statistic: float | None = Field(default=None, description="F-statistic")
    f_p_value: float | None = Field(default=None, description="P-value for F-test")
    n_observations: int = Field(description="Number of observations")
    coefficients: list[dict[str, Any]] = Field(description="List of coefficient dictionaries")
    research_context: str = Field(default="", description="Brief context about the research question")


class InterpretHypothesisTestInput(BaseModel):
    """Input schema for interpret_hypothesis_test tool."""
    
    test_name: str = Field(description="Name of the statistical test")
    hypothesis: str = Field(description="The hypothesis being tested")
    statistic: float = Field(description="Test statistic value")
    p_value: float = Field(description="P-value")
    degrees_of_freedom: int | None = Field(default=None, description="Degrees of freedom")
    effect_size: float | None = Field(default=None, description="Effect size value")
    effect_size_type: str | None = Field(default=None, description="Type of effect size")
    group_statistics: dict[str, Any] | None = Field(default=None, description="Group-level statistics")
    research_context: str = Field(default="", description="Brief context about the research question")


class SummarizeFindingsInput(BaseModel):
    """Input schema for summarize_findings tool."""
    
    findings: list[dict[str, Any]] = Field(description="List of findings with statistical support")
    research_question: str = Field(description="The research question being addressed")
    primary_gap: str = Field(default="", description="The gap this research addresses")
    include_limitations: bool = Field(default=True, description="Whether to include limitations")


class GenerateMethodsSectionInput(BaseModel):
    """Input schema for generate_methods_section tool."""
    
    analysis_approach: str = Field(description="Overall analytical approach")
    statistical_tests: list[str] = Field(description="List of statistical tests used")
    variables: dict[str, Any] = Field(description="Variables and their roles")
    sample_description: str = Field(description="Description of the sample")
    data_quality_notes: str = Field(default="", description="Notes on data quality")


# =============================================================================
# Interpretation Tools
# =============================================================================


@tool(args_schema=InterpretRegressionInput)
def interpret_regression(
    model_type: str,
    dependent_variable: str,
    r_squared: float,
    adjusted_r_squared: float,
    f_statistic: float | None,
    f_p_value: float | None,
    n_observations: int,
    coefficients: list[dict[str, Any]],
    research_context: str = "",
) -> dict[str, Any]:
    """
    Generate academic prose interpretation of regression results.
    
    Creates publication-ready text describing regression results in standard
    academic format, including model fit, significant predictors, and
    implications for the research question.
    
    Args:
        model_type: Type of regression model (OLS, logistic, etc.)
        dependent_variable: Name of the dependent variable
        r_squared: R-squared value
        adjusted_r_squared: Adjusted R-squared value
        f_statistic: F-statistic for model significance
        f_p_value: P-value for F-test
        n_observations: Number of observations
        coefficients: List of coefficient dictionaries with variable, coefficient,
                     std_error, t_statistic, p_value, is_significant
        research_context: Brief context about the research question
    
    Returns:
        Dictionary with formatted academic prose sections.
    """
    llm = _get_llm()
    
    # Build coefficient summary
    sig_coeffs = [c for c in coefficients if c.get("is_significant", False) and c.get("variable") != "(Intercept)"]
    insig_coeffs = [c for c in coefficients if not c.get("is_significant", True) and c.get("variable") != "(Intercept)"]
    
    coef_details = "\n".join([
        f"- {c['variable']}: b = {c['coefficient']:.3f}, SE = {c['std_error']:.3f}, "
        f"t = {c['t_statistic']:.2f}, p = {c['p_value']:.4f}, significant: {c['is_significant']}"
        for c in coefficients
    ])
    
    prompt = f"""Generate academic prose interpreting these regression results.

MODEL: {model_type.upper()} regression
DEPENDENT VARIABLE: {dependent_variable}
SAMPLE SIZE: n = {n_observations}

MODEL FIT:
- R² = {r_squared:.3f}
- Adjusted R² = {adjusted_r_squared:.3f}
- F-statistic = {f_statistic:.2f if f_statistic else 'N/A'}
- F p-value = {f_p_value:.4f if f_p_value else 'N/A'}

COEFFICIENTS:
{coef_details}

SIGNIFICANT PREDICTORS: {len(sig_coeffs)}
NON-SIGNIFICANT PREDICTORS: {len(insig_coeffs)}

RESEARCH CONTEXT: {research_context or 'Not provided'}

{_build_style_instructions()}

Generate TWO sections:
1. MODEL FIT PARAGRAPH: Describe overall model fit and significance
2. PREDICTOR EFFECTS PARAGRAPH: Describe each significant predictor's effect

Output format:
MODEL_FIT:
[paragraph about model fit]

PREDICTOR_EFFECTS:
[paragraph about predictor effects]
"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Parse response
        model_fit = ""
        predictor_effects = ""
        
        if "MODEL_FIT:" in content:
            parts = content.split("PREDICTOR_EFFECTS:")
            model_fit = parts[0].replace("MODEL_FIT:", "").strip()
            if len(parts) > 1:
                predictor_effects = parts[1].strip()
        else:
            # Fallback: use entire response
            model_fit = content.strip()
        
        return {
            "status": "success",
            "model_fit_paragraph": model_fit,
            "predictor_effects_paragraph": predictor_effects,
            "significant_predictors": [c["variable"] for c in sig_coeffs],
            "model_explains_variance": r_squared * 100,
        }
        
    except Exception as e:
        # Fallback to template-based interpretation
        return {
            "status": "fallback",
            "model_fit_paragraph": (
                f"The {model_type.upper()} regression model with {dependent_variable} as the "
                f"dependent variable was statistically significant, F = {f_statistic:.2f}, "
                f"p {'< .001' if f_p_value and f_p_value < 0.001 else f'= {f_p_value:.3f}' if f_p_value else 'N/A'}, "
                f"explaining {r_squared * 100:.1f}% of the variance in {dependent_variable} "
                f"(R² = {r_squared:.3f}, adjusted R² = {adjusted_r_squared:.3f})."
            ),
            "predictor_effects_paragraph": (
                f"Among the predictors, {len(sig_coeffs)} showed statistically significant "
                f"effects on {dependent_variable}."
            ),
            "significant_predictors": [c["variable"] for c in sig_coeffs],
            "model_explains_variance": r_squared * 100,
            "error": str(e),
        }


@tool(args_schema=InterpretHypothesisTestInput)
def interpret_hypothesis_test(
    test_name: str,
    hypothesis: str,
    statistic: float,
    p_value: float,
    degrees_of_freedom: int | None = None,
    effect_size: float | None = None,
    effect_size_type: str | None = None,
    group_statistics: dict[str, Any] | None = None,
    research_context: str = "",
) -> dict[str, Any]:
    """
    Generate academic prose interpretation of hypothesis test results.
    
    Creates publication-ready text describing hypothesis test results in APA
    format with effect size interpretation.
    
    Args:
        test_name: Name of the statistical test
        hypothesis: The hypothesis being tested
        statistic: Test statistic value
        p_value: P-value
        degrees_of_freedom: Degrees of freedom if applicable
        effect_size: Effect size value (Cohen's d, eta-squared, etc.)
        effect_size_type: Type of effect size measure
        group_statistics: Group-level statistics (means, SDs, ns)
        research_context: Brief context about the research question
    
    Returns:
        Dictionary with formatted academic prose.
    """
    llm = _get_llm()
    
    # Determine significance
    is_significant = p_value < 0.05
    
    # Format effect size interpretation
    effect_interpretation = ""
    if effect_size is not None and effect_size_type:
        if effect_size_type.lower() in ["cohen's d", "cohens_d", "d"]:
            if abs(effect_size) >= 0.8:
                effect_interpretation = "large"
            elif abs(effect_size) >= 0.5:
                effect_interpretation = "medium"
            elif abs(effect_size) >= 0.2:
                effect_interpretation = "small"
            else:
                effect_interpretation = "negligible"
        elif effect_size_type.lower() in ["eta-squared", "eta_squared", "η²"]:
            if effect_size >= 0.14:
                effect_interpretation = "large"
            elif effect_size >= 0.06:
                effect_interpretation = "medium"
            elif effect_size >= 0.01:
                effect_interpretation = "small"
            else:
                effect_interpretation = "negligible"
    
    # Format group statistics if available
    group_info = ""
    if group_statistics:
        group_info = "\n".join([
            f"- {g}: M = {s.get('mean', 'N/A'):.2f}, SD = {s.get('std', 'N/A'):.2f}, n = {s.get('n', 'N/A')}"
            for g, s in group_statistics.items()
        ])
    
    prompt = f"""Generate academic prose interpreting these hypothesis test results.

TEST: {test_name}
HYPOTHESIS: {hypothesis}

RESULTS:
- Test statistic = {statistic:.3f}
- Degrees of freedom = {degrees_of_freedom if degrees_of_freedom else 'N/A'}
- p-value = {p_value:.4f}
- Effect size ({effect_size_type or 'not specified'}) = {effect_size:.3f if effect_size else 'N/A'}
- Effect interpretation: {effect_interpretation or 'N/A'}
- Significant at α = 0.05: {'Yes' if is_significant else 'No'}

GROUP STATISTICS:
{group_info or 'Not provided'}

RESEARCH CONTEXT: {research_context or 'Not provided'}

{_build_style_instructions()}

Generate a single paragraph interpreting these results in APA format. Include:
1. The test conducted and hypothesis
2. Statistical values in proper APA format
3. Effect size with interpretation
4. Conclusion about the hypothesis

INTERPRETATION:
"""
    
    try:
        response = llm.invoke(prompt)
        interpretation = response.content.replace("INTERPRETATION:", "").strip()
        
        return {
            "status": "success",
            "interpretation": interpretation,
            "is_significant": is_significant,
            "effect_size_interpretation": effect_interpretation,
            "supports_hypothesis": is_significant,
        }
        
    except Exception as e:
        # Fallback template
        df_str = f"({degrees_of_freedom})" if degrees_of_freedom else ""
        p_str = "< .001" if p_value < 0.001 else f"= {p_value:.3f}"
        
        return {
            "status": "fallback",
            "interpretation": (
                f"A {test_name} was conducted to test the hypothesis that {hypothesis}. "
                f"Results {'were' if is_significant else 'were not'} statistically significant, "
                f"{test_name.split()[0].lower()}{df_str} = {statistic:.2f}, p {p_str}"
                f"{f', {effect_size_type} = {effect_size:.2f} ({effect_interpretation} effect)' if effect_size else ''}. "
                f"The null hypothesis {'is rejected' if is_significant else 'cannot be rejected'}."
            ),
            "is_significant": is_significant,
            "effect_size_interpretation": effect_interpretation,
            "supports_hypothesis": is_significant,
            "error": str(e),
        }


@tool(args_schema=SummarizeFindingsInput)
def summarize_findings(
    findings: list[dict[str, Any]],
    research_question: str,
    primary_gap: str = "",
    include_limitations: bool = True,
) -> dict[str, Any]:
    """
    Generate a comprehensive summary of research findings.
    
    Creates a cohesive narrative summarizing multiple findings, their
    relationship to the research question, and implications.
    
    Args:
        findings: List of findings with statements and statistical support
        research_question: The research question being addressed
        primary_gap: The gap this research addresses
        include_limitations: Whether to include limitations section
    
    Returns:
        Dictionary with summary narrative and key takeaways.
    """
    llm = _get_llm()
    
    # Format findings
    findings_text = "\n\n".join([
        f"Finding {i+1} ({f.get('finding_type', 'general')}):\n"
        f"Statement: {f.get('statement', 'No statement')}\n"
        f"Evidence strength: {f.get('evidence_strength', 'unknown')}\n"
        f"Statistical support: {f.get('detailed_description', 'None provided')}"
        for i, f in enumerate(findings)
    ])
    
    # Categorize findings
    main_findings = [f for f in findings if f.get("finding_type") == "main_result"]
    supporting = [f for f in findings if f.get("finding_type") == "supporting"]
    unexpected = [f for f in findings if f.get("finding_type") == "unexpected"]
    null_results = [f for f in findings if f.get("finding_type") == "null_result"]
    
    prompt = f"""Generate an academic summary of these research findings.

RESEARCH QUESTION: {research_question}
GAP ADDRESSED: {primary_gap or 'Not specified'}

FINDINGS:
{findings_text}

FINDING BREAKDOWN:
- Main results: {len(main_findings)}
- Supporting findings: {len(supporting)}
- Unexpected findings: {len(unexpected)}
- Null results: {len(null_results)}

{_build_style_instructions()}

Generate the following sections:
1. KEY_FINDINGS: 2-3 sentences summarizing the main results
2. SUPPORTING_EVIDENCE: How secondary findings support main conclusions
3. IMPLICATIONS: What these findings mean for the research question
{'4. LIMITATIONS: Brief note on limitations' if include_limitations else ''}

Output format:
KEY_FINDINGS:
[text]

SUPPORTING_EVIDENCE:
[text]

IMPLICATIONS:
[text]

{'LIMITATIONS:' if include_limitations else ''}
{'[text]' if include_limitations else ''}
"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Parse sections
        sections = {
            "key_findings": "",
            "supporting_evidence": "",
            "implications": "",
            "limitations": "",
        }
        
        current_section = None
        current_content = []
        
        for line in content.split("\n"):
            line_upper = line.upper().strip()
            if "KEY_FINDINGS:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "key_findings"
                current_content = []
            elif "SUPPORTING_EVIDENCE:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "supporting_evidence"
                current_content = []
            elif "IMPLICATIONS:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "implications"
                current_content = []
            elif "LIMITATIONS:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "limitations"
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()
        
        return {
            "status": "success",
            **sections,
            "total_findings": len(findings),
            "main_findings_count": len(main_findings),
            "addresses_gap": any(f.get("addresses_gap", False) for f in findings),
        }
        
    except Exception as e:
        # Fallback
        return {
            "status": "fallback",
            "key_findings": f"The analysis yielded {len(main_findings)} main finding(s) addressing the research question.",
            "supporting_evidence": f"{len(supporting)} supporting finding(s) provide additional context.",
            "implications": "These findings contribute to addressing the identified research gap.",
            "limitations": "Limitations of the analysis should be considered when interpreting results.",
            "total_findings": len(findings),
            "main_findings_count": len(main_findings),
            "addresses_gap": any(f.get("addresses_gap", False) for f in findings),
            "error": str(e),
        }


@tool(args_schema=GenerateMethodsSectionInput)
def generate_methods_section(
    analysis_approach: str,
    statistical_tests: list[str],
    variables: dict[str, Any],
    sample_description: str,
    data_quality_notes: str = "",
) -> dict[str, Any]:
    """
    Generate a methods section describing the analytical approach.
    
    Creates publication-ready text for the methods section describing
    the statistical methods used, variables, and analytical decisions.
    
    Args:
        analysis_approach: Overall analytical approach (e.g., "OLS regression")
        statistical_tests: List of statistical tests used
        variables: Dictionary with dependent, independent, control variables
        sample_description: Description of the sample
        data_quality_notes: Notes on data quality and preprocessing
    
    Returns:
        Dictionary with methods section prose.
    """
    llm = _get_llm()
    
    # Format variables
    var_text = ""
    if "dependent" in variables:
        var_text += f"Dependent variable: {variables['dependent']}\n"
    if "independent" in variables:
        ind = variables["independent"]
        if isinstance(ind, list):
            var_text += f"Independent variables: {', '.join(ind)}\n"
        else:
            var_text += f"Independent variable: {ind}\n"
    if "control" in variables:
        ctrl = variables["control"]
        if isinstance(ctrl, list) and ctrl:
            var_text += f"Control variables: {', '.join(ctrl)}\n"
    
    prompt = f"""Generate a methods section for an academic paper.

ANALYTICAL APPROACH: {analysis_approach}
STATISTICAL TESTS USED: {', '.join(statistical_tests)}

VARIABLES:
{var_text}

SAMPLE: {sample_description}
DATA QUALITY NOTES: {data_quality_notes or 'None'}

{_build_style_instructions()}

Generate a methods section with:
1. ANALYTICAL_STRATEGY: Overall approach and justification
2. VARIABLES: Description of variables and operationalization
3. STATISTICAL_METHODS: Description of statistical methods used
4. DATA_HANDLING: Any preprocessing or data handling decisions

Output format:
ANALYTICAL_STRATEGY:
[text]

VARIABLES:
[text]

STATISTICAL_METHODS:
[text]

DATA_HANDLING:
[text]
"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Parse sections
        sections = {
            "analytical_strategy": "",
            "variables": "",
            "statistical_methods": "",
            "data_handling": "",
        }
        
        current_section = None
        current_content = []
        
        for line in content.split("\n"):
            line_upper = line.upper().strip()
            if "ANALYTICAL_STRATEGY:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "analytical_strategy"
                current_content = []
            elif "VARIABLES:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "variables"
                current_content = []
            elif "STATISTICAL_METHODS:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "statistical_methods"
                current_content = []
            elif "DATA_HANDLING:" in line_upper:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "data_handling"
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()
        
        # Combine into full methods section
        full_section = "\n\n".join([
            sections["analytical_strategy"],
            sections["variables"],
            sections["statistical_methods"],
            sections["data_handling"],
        ])
        
        return {
            "status": "success",
            "full_methods_section": full_section,
            **sections,
        }
        
    except Exception as e:
        return {
            "status": "fallback",
            "analytical_strategy": f"This study employed {analysis_approach} to address the research question.",
            "variables": var_text.replace("\n", " "),
            "statistical_methods": f"Statistical analyses included {', '.join(statistical_tests)}.",
            "data_handling": data_quality_notes or "Standard data preprocessing procedures were applied.",
            "full_methods_section": "",
            "error": str(e),
        }


# =============================================================================
# Exports
# =============================================================================

LLM_INTERPRETATION_TOOLS = [
    interpret_regression,
    interpret_hypothesis_test,
    summarize_findings,
    generate_methods_section,
]


def get_interpretation_tools() -> list:
    """Get list of all LLM interpretation tools."""
    return LLM_INTERPRETATION_TOOLS


__all__ = [
    "interpret_regression",
    "interpret_hypothesis_test",
    "summarize_findings",
    "generate_methods_section",
    "get_interpretation_tools",
    "LLM_INTERPRETATION_TOOLS",
]
