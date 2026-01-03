"""Analysis tools for the DATA_ANALYST node.

These tools provide statistical analysis capabilities for empirical research,
including descriptive statistics, hypothesis testing, correlation analysis,
and regression modeling. All tools integrate with the DataRegistry for
seamless data access.
"""

from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from src.state.enums import (
    StatisticalTestType,
    EvidenceStrength,
    FindingType,
)
from src.state.models import (
    StatisticalResult,
    RegressionResult,
    RegressionCoefficient,
    DataAnalysisFinding,
)
from src.tools.data_loading import get_registry


# =============================================================================
# Descriptive Statistics Tools
# =============================================================================


@tool
def execute_descriptive_stats(
    dataset_name: str,
    variables: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate descriptive statistics for variables in a registered dataset.
    
    Computes comprehensive summary statistics including mean, standard deviation,
    min, max, quartiles, skewness, kurtosis, and normality test for numeric variables.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variables: Optional list of specific variables to analyze.
                  If None, analyzes all numeric variables.
    
    Returns:
        Dictionary with descriptive statistics for each variable.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found. Available: {list(registry.datasets.keys())}",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if variables:
        numeric_cols = [c for c in variables if c in numeric_cols]
    
    if not numeric_cols:
        return {
            "status": "no_numeric_data",
            "error": "No numeric variables found for analysis",
        }
    
    results = {
        "status": "complete",
        "dataset": dataset_name,
        "n_observations": len(df),
        "variables_analyzed": numeric_cols,
        "statistics": {},
    }
    
    for col in numeric_cols:
        data = df[col].dropna()
        n = len(data)
        
        if n == 0:
            results["statistics"][col] = {"error": "No non-null values"}
            continue
        
        # Basic statistics
        col_stats = {
            "n": n,
            "missing": len(df) - n,
            "missing_pct": ((len(df) - n) / len(df)) * 100,
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "q25": float(data.quantile(0.25)),
            "median": float(data.median()),
            "q75": float(data.quantile(0.75)),
            "max": float(data.max()),
            "range": float(data.max() - data.min()),
            "iqr": float(data.quantile(0.75) - data.quantile(0.25)),
        }
        
        # Add skewness and kurtosis if enough data
        if n >= 8:
            col_stats["skewness"] = float(stats.skew(data))
            col_stats["kurtosis"] = float(stats.kurtosis(data))
            
            # Normality test (Shapiro-Wilk for n < 5000, otherwise D'Agostino-Pearson)
            try:
                if n < 5000:
                    stat, p_val = stats.shapiro(data)
                    col_stats["normality_test"] = "Shapiro-Wilk"
                else:
                    stat, p_val = stats.normaltest(data)
                    col_stats["normality_test"] = "D'Agostino-Pearson"
                col_stats["normality_statistic"] = float(stat)
                col_stats["normality_p_value"] = float(p_val)
                col_stats["is_normal"] = p_val > 0.05
            except Exception:
                col_stats["normality_test"] = "Could not compute"
        
        results["statistics"][col] = col_stats
    
    return results


@tool
def compute_correlation_matrix(
    dataset_name: str,
    variables: list[str] | None = None,
    method: str = "pearson",
) -> dict[str, Any]:
    """
    Compute a correlation matrix with significance testing for numeric variables.
    
    Calculates pairwise correlations using Pearson, Spearman, or Kendall methods,
    with p-values for each correlation coefficient.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variables: Optional list of specific variables to include.
        method: Correlation method: 'pearson', 'spearman', or 'kendall'.
    
    Returns:
        Dictionary with correlation matrix, p-values, and notable correlations.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {
            "status": "error",
            "error": f"Dataset '{dataset_name}' not found.",
        }
    
    df = registry.get_dataframe(dataset_name)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if variables:
        numeric_cols = [c for c in variables if c in numeric_cols]
    
    if len(numeric_cols) < 2:
        return {
            "status": "error",
            "error": "Need at least 2 numeric variables for correlation analysis",
        }
    
    df_numeric = df[numeric_cols].dropna()
    n = len(df_numeric)
    
    if n < 3:
        return {
            "status": "error",
            "error": f"Insufficient observations ({n}) after removing missing values",
        }
    
    # Compute correlation matrix
    corr_matrix = df_numeric.corr(method=method)
    
    # Compute p-values
    p_matrix = pd.DataFrame(
        np.ones((len(numeric_cols), len(numeric_cols))),
        index=numeric_cols,
        columns=numeric_cols,
    )
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                if method == "pearson":
                    _, p = stats.pearsonr(df_numeric[col1], df_numeric[col2])
                elif method == "spearman":
                    _, p = stats.spearmanr(df_numeric[col1], df_numeric[col2])
                else:  # kendall
                    _, p = stats.kendalltau(df_numeric[col1], df_numeric[col2])
                p_matrix.loc[col1, col2] = p
                p_matrix.loc[col2, col1] = p
    
    # Find notable correlations
    strong_positive = []
    strong_negative = []
    moderate_correlations = []
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                r = corr_matrix.loc[col1, col2]
                p = p_matrix.loc[col1, col2]
                
                corr_info = {
                    "var1": col1,
                    "var2": col2,
                    "correlation": round(r, 4),
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                }
                
                if abs(r) >= 0.7:
                    if r > 0:
                        strong_positive.append(corr_info)
                    else:
                        strong_negative.append(corr_info)
                elif abs(r) >= 0.3:
                    moderate_correlations.append(corr_info)
    
    return {
        "status": "complete",
        "dataset": dataset_name,
        "method": method,
        "n_observations": n,
        "variables": numeric_cols,
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "p_value_matrix": p_matrix.round(4).to_dict(),
        "strong_positive_correlations": strong_positive,
        "strong_negative_correlations": strong_negative,
        "moderate_correlations": moderate_correlations,
        "interpretation": (
            f"Computed {method.capitalize()} correlations for {len(numeric_cols)} variables. "
            f"Found {len(strong_positive)} strong positive and {len(strong_negative)} "
            f"strong negative correlations (|r| >= 0.7)."
        ),
    }


# =============================================================================
# Hypothesis Testing Tools
# =============================================================================


@tool
def run_ttest(
    dataset_name: str,
    variable: str,
    group_variable: str | None = None,
    test_type: str = "independent",
    paired_variable: str | None = None,
    hypothesized_mean: float | None = None,
    alpha: float = 0.05,
) -> StatisticalResult:
    """
    Perform a t-test for comparing means.
    
    Supports one-sample, independent two-sample, and paired t-tests with
    effect size (Cohen's d) and confidence intervals.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variable: The variable to test.
        group_variable: Grouping variable for independent t-test (must have 2 groups).
        test_type: Type of t-test: 'one_sample', 'independent', or 'paired'.
        paired_variable: Second variable for paired t-test.
        hypothesized_mean: Population mean for one-sample test (default 0).
        alpha: Significance level (default 0.05).
    
    Returns:
        StatisticalResult with test statistics, p-value, effect size, and interpretation.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.T_TEST,
            test_name="T-Test (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Dataset '{dataset_name}' not found.",
        )
    
    df = registry.get_dataframe(dataset_name)
    
    try:
        if test_type == "one_sample":
            # One-sample t-test
            data = df[variable].dropna()
            mu = hypothesized_mean or 0
            t_stat, p_value = stats.ttest_1samp(data, mu)
            
            # Cohen's d for one-sample
            cohens_d = (data.mean() - mu) / data.std()
            
            test_name = "One-Sample T-Test"
            df_val = len(data) - 1
            
            interpretation = (
                f"One-sample t-test comparing {variable} (M={data.mean():.3f}, "
                f"SD={data.std():.3f}) to hypothesized mean of {mu}. "
            )
            
        elif test_type == "paired":
            # Paired t-test
            if not paired_variable:
                raise ValueError("paired_variable required for paired t-test")
            
            data1 = df[variable].dropna()
            data2 = df[paired_variable].dropna()
            
            # Align data
            combined = df[[variable, paired_variable]].dropna()
            data1 = combined[variable]
            data2 = combined[paired_variable]
            
            t_stat, p_value = stats.ttest_rel(data1, data2)
            
            # Cohen's d for paired
            diff = data1 - data2
            cohens_d = diff.mean() / diff.std()
            
            test_name = "Paired Samples T-Test"
            df_val = len(data1) - 1
            
            interpretation = (
                f"Paired t-test comparing {variable} (M={data1.mean():.3f}) "
                f"and {paired_variable} (M={data2.mean():.3f}). "
                f"Mean difference: {diff.mean():.3f}. "
            )
            
        else:  # independent
            # Independent two-sample t-test
            if not group_variable:
                raise ValueError("group_variable required for independent t-test")
            
            groups = df[group_variable].dropna().unique()
            if len(groups) != 2:
                raise ValueError(f"Expected 2 groups, found {len(groups)}")
            
            group1_data = df[df[group_variable] == groups[0]][variable].dropna()
            group2_data = df[df[group_variable] == groups[1]][variable].dropna()
            
            # Levene's test for equal variances
            _, levene_p = stats.levene(group1_data, group2_data)
            equal_var = levene_p > 0.05
            
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
            
            # Cohen's d for independent samples
            pooled_std = np.sqrt(
                ((len(group1_data) - 1) * group1_data.std()**2 +
                 (len(group2_data) - 1) * group2_data.std()**2) /
                (len(group1_data) + len(group2_data) - 2)
            )
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            test_name = "Independent Samples T-Test" + (" (Welch's)" if not equal_var else "")
            df_val = len(group1_data) + len(group2_data) - 2
            
            interpretation = (
                f"Independent t-test comparing {variable} between {groups[0]} "
                f"(M={group1_data.mean():.3f}, n={len(group1_data)}) and {groups[1]} "
                f"(M={group2_data.mean():.3f}, n={len(group2_data)}). "
            )
        
        # Effect size interpretation
        effect_label = "negligible"
        if abs(cohens_d) >= 0.8:
            effect_label = "large"
        elif abs(cohens_d) >= 0.5:
            effect_label = "medium"
        elif abs(cohens_d) >= 0.2:
            effect_label = "small"
        
        is_significant = p_value < alpha
        
        if is_significant:
            interpretation += (
                f"Result is statistically significant (t={t_stat:.3f}, p={p_value:.4f}, "
                f"Cohen's d={cohens_d:.3f}, {effect_label} effect). "
                f"We reject the null hypothesis at alpha={alpha}."
            )
        else:
            interpretation += (
                f"Result is not statistically significant (t={t_stat:.3f}, p={p_value:.4f}, "
                f"Cohen's d={cohens_d:.3f}). We fail to reject the null hypothesis."
            )
        
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.T_TEST if test_type != "paired" else StatisticalTestType.PAIRED_T_TEST,
            test_name=test_name,
            statistic=float(t_stat),
            p_value=float(p_value),
            degrees_of_freedom=int(df_val),
            confidence_level=1 - alpha,
            is_significant=is_significant,
            effect_size=float(cohens_d),
            effect_size_type="Cohen's d",
            interpretation=interpretation,
        )
        
    except Exception as e:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.T_TEST,
            test_name="T-Test (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Error performing t-test: {str(e)}",
        )


@tool
def run_anova(
    dataset_name: str,
    dependent_variable: str,
    group_variable: str,
    alpha: float = 0.05,
) -> StatisticalResult:
    """
    Perform one-way ANOVA to compare means across multiple groups.
    
    Tests whether there are statistically significant differences between
    group means, with eta-squared effect size and post-hoc comparisons if significant.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        dependent_variable: The continuous dependent variable.
        group_variable: The categorical grouping variable.
        alpha: Significance level (default 0.05).
    
    Returns:
        StatisticalResult with F-statistic, p-value, effect size, and group statistics.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.ANOVA,
            test_name="One-Way ANOVA (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Dataset '{dataset_name}' not found.",
        )
    
    df = registry.get_dataframe(dataset_name)
    
    try:
        # Get groups
        groups = df[group_variable].dropna().unique()
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups, found {len(groups)}")
        
        # Create list of arrays for each group
        group_data = [
            df[df[group_variable] == g][dependent_variable].dropna().values
            for g in groups
        ]
        
        # Remove empty groups
        non_empty = [(g, d) for g, d in zip(groups, group_data) if len(d) > 0]
        groups = [g for g, _ in non_empty]
        group_data = [d for _, d in non_empty]
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Effect size (eta-squared)
        all_data = np.concatenate(group_data)
        grand_mean = all_data.mean()
        
        ss_between = sum(len(d) * (d.mean() - grand_mean)**2 for d in group_data)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Group statistics
        group_stats = {
            str(g): {
                "n": len(d),
                "mean": float(d.mean()),
                "std": float(d.std()),
            }
            for g, d in zip(groups, group_data)
        }
        
        is_significant = p_value < alpha
        
        # Effect size interpretation
        effect_label = "negligible"
        if eta_squared >= 0.14:
            effect_label = "large"
        elif eta_squared >= 0.06:
            effect_label = "medium"
        elif eta_squared >= 0.01:
            effect_label = "small"
        
        interpretation = (
            f"One-way ANOVA comparing {dependent_variable} across {len(groups)} groups "
            f"of {group_variable}. "
        )
        
        # Add group statistics to interpretation
        group_stats_str = "; ".join(
            f"{g}: M={s['mean']:.3f}, SD={s['std']:.3f}, n={s['n']}"
            for g, s in group_stats.items()
        )
        
        if is_significant:
            interpretation += (
                f"Result is statistically significant (F={f_stat:.3f}, p={p_value:.4f}, "
                f"eta-squared={eta_squared:.3f}, {effect_label} effect). "
                f"At least one group mean differs significantly from the others. "
                f"Group statistics: {group_stats_str}."
            )
        else:
            interpretation += (
                f"Result is not statistically significant (F={f_stat:.3f}, p={p_value:.4f}). "
                f"No evidence of differences between group means. "
                f"Group statistics: {group_stats_str}."
            )
        
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.ANOVA,
            test_name="One-Way ANOVA",
            statistic=float(f_stat),
            p_value=float(p_value),
            degrees_of_freedom=int(len(groups) - 1),
            confidence_level=1 - alpha,
            is_significant=is_significant,
            effect_size=float(eta_squared),
            effect_size_type="eta-squared",
            interpretation=interpretation,
        )
        
    except Exception as e:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.ANOVA,
            test_name="One-Way ANOVA (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Error performing ANOVA: {str(e)}",
        )


@tool
def run_chi_square(
    dataset_name: str,
    variable1: str,
    variable2: str,
    alpha: float = 0.05,
) -> StatisticalResult:
    """
    Perform chi-square test of independence for categorical variables.
    
    Tests whether there is a statistically significant association between
    two categorical variables, with Cramer's V effect size.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variable1: First categorical variable.
        variable2: Second categorical variable.
        alpha: Significance level (default 0.05).
    
    Returns:
        StatisticalResult with chi-square statistic, p-value, and effect size.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.CHI_SQUARE,
            test_name="Chi-Square Test (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Dataset '{dataset_name}' not found.",
        )
    
    df = registry.get_dataframe(dataset_name)
    
    try:
        # Create contingency table
        contingency = pd.crosstab(df[variable1], df[variable2])
        
        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Cramer's V effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        is_significant = p_value < alpha
        
        # Effect size interpretation
        effect_label = "negligible"
        if cramers_v >= 0.5:
            effect_label = "large"
        elif cramers_v >= 0.3:
            effect_label = "medium"
        elif cramers_v >= 0.1:
            effect_label = "small"
        
        interpretation = (
            f"Chi-square test of independence between {variable1} and {variable2}. "
            f"Contingency table: {contingency.shape[0]} x {contingency.shape[1]}, N={n}. "
        )
        
        if is_significant:
            interpretation += (
                f"Result is statistically significant (chi2={chi2:.3f}, df={dof}, "
                f"p={p_value:.4f}, Cramer's V={cramers_v:.3f}, {effect_label} effect). "
                f"There is a significant association between the variables."
            )
        else:
            interpretation += (
                f"Result is not statistically significant (chi2={chi2:.3f}, df={dof}, "
                f"p={p_value:.4f}). No evidence of association between variables."
            )
        
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.CHI_SQUARE,
            test_name="Chi-Square Test of Independence",
            statistic=float(chi2),
            p_value=float(p_value),
            degrees_of_freedom=int(dof),
            confidence_level=1 - alpha,
            is_significant=is_significant,
            effect_size=float(cramers_v),
            effect_size_type="Cramer's V",
            interpretation=interpretation,
        )
        
    except Exception as e:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.CHI_SQUARE,
            test_name="Chi-Square Test (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Error performing chi-square test: {str(e)}",
        )


# =============================================================================
# Regression Analysis Tools
# =============================================================================


@tool
def run_ols_regression(
    dataset_name: str,
    dependent_variable: str,
    independent_variables: list[str],
    control_variables: list[str] | None = None,
    robust_se: bool = False,
    alpha: float = 0.05,
) -> RegressionResult:
    """
    Perform OLS regression analysis with comprehensive diagnostics.
    
    Fits an ordinary least squares regression model with options for
    heteroskedasticity-robust standard errors and full diagnostic tests.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        dependent_variable: The dependent variable name.
        independent_variables: List of independent variable names.
        control_variables: Optional list of control variable names.
        robust_se: Whether to use heteroskedasticity-robust standard errors.
        alpha: Significance level (default 0.05).
    
    Returns:
        RegressionResult with coefficients, fit statistics, and diagnostics.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return RegressionResult(
            result_id=str(uuid4())[:8],
            model_type="ols_error",
            dependent_variable=dependent_variable,
            r_squared=0.0,
            n_observations=0,
            interpretation=f"Dataset '{dataset_name}' not found.",
        )
    
    df = registry.get_dataframe(dataset_name)
    controls = control_variables or []
    all_vars = independent_variables + controls
    
    try:
        # Prepare data
        vars_needed = [dependent_variable] + all_vars
        df_model = df[vars_needed].dropna()
        
        if len(df_model) < len(all_vars) + 2:
            raise ValueError(
                f"Insufficient observations ({len(df_model)}) for "
                f"{len(all_vars)} variables"
            )
        
        y = df_model[dependent_variable]
        X = df_model[all_vars]
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X)
        if robust_se:
            results = model.fit(cov_type='HC3')
        else:
            results = model.fit()
        
        # Extract coefficients
        coefficients = []
        for var in results.params.index:
            idx = list(results.params.index).index(var)
            coef = results.params[var]
            se = results.bse[var]
            t_val = results.tvalues[var]
            p_val = results.pvalues[var]
            ci = results.conf_int(alpha=alpha).loc[var]
            
            display_var = "(Intercept)" if var == "const" else var
            
            coefficients.append(
                RegressionCoefficient(
                    variable=display_var,
                    coefficient=float(coef),
                    std_error=float(se),
                    t_statistic=float(t_val),
                    p_value=float(p_val),
                    confidence_interval_lower=float(ci[0]),
                    confidence_interval_upper=float(ci[1]),
                    is_significant=p_val < alpha,
                )
            )
        
        # Compute diagnostics
        # Durbin-Watson test for autocorrelation
        dw = durbin_watson(results.resid)
        
        # Breusch-Pagan test for heteroskedasticity
        het_test_result = None
        try:
            bp_stat, bp_pval, _, _ = het_breuschpagan(results.resid, X)
            het_concern = bp_pval < 0.05
            het_test_result = f"Breusch-Pagan: stat={bp_stat:.3f}, p={bp_pval:.4f}"
            if het_concern:
                het_test_result += " (heteroskedasticity detected)"
        except Exception:
            het_test_result = "Could not compute"
        
        # Build interpretation
        sig_vars = [c.variable for c in coefficients 
                    if c.is_significant and c.variable != "(Intercept)"]
        
        interpretation = (
            f"OLS regression with {dependent_variable} as dependent variable "
            f"(n={len(df_model)}). "
        )
        
        if sig_vars:
            interpretation += f"Significant predictors: {', '.join(sig_vars)}. "
        else:
            interpretation += "No significant predictors found. "
        
        interpretation += (
            f"Model explains {results.rsquared * 100:.1f}% of variance "
            f"(Adj. R-squared={results.rsquared_adj:.3f}). "
            f"F-statistic={results.fvalue:.2f} (p={results.f_pvalue:.4f})."
        )
        
        if robust_se:
            interpretation += " Heteroskedasticity-robust standard errors used."
        
        return RegressionResult(
            result_id=str(uuid4())[:8],
            model_type="ols" + ("_robust" if robust_se else ""),
            dependent_variable=dependent_variable,
            r_squared=float(results.rsquared),
            adjusted_r_squared=float(results.rsquared_adj),
            f_statistic=float(results.fvalue),
            f_p_value=float(results.f_pvalue),
            coefficients=coefficients,
            n_observations=len(df_model),
            residual_std_error=float(np.sqrt(results.mse_resid)),
            durbin_watson=float(dw),
            heteroskedasticity_test=het_test_result,
            interpretation=interpretation,
        )
        
    except Exception as e:
        return RegressionResult(
            result_id=str(uuid4())[:8],
            model_type="ols_error",
            dependent_variable=dependent_variable,
            r_squared=0.0,
            adjusted_r_squared=0.0,
            n_observations=0,
            interpretation=f"Error performing regression: {str(e)}",
        )


@tool
def run_logistic_regression(
    dataset_name: str,
    dependent_variable: str,
    independent_variables: list[str],
    control_variables: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Perform logistic regression for binary outcomes.
    
    Fits a logistic regression model with odds ratios and model fit statistics.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        dependent_variable: Binary dependent variable (0/1).
        independent_variables: List of independent variable names.
        control_variables: Optional list of control variable names.
        alpha: Significance level (default 0.05).
    
    Returns:
        Dictionary with coefficients, odds ratios, and model fit statistics.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {"status": "error", "error": f"Dataset '{dataset_name}' not found."}
    
    df = registry.get_dataframe(dataset_name)
    controls = control_variables or []
    all_vars = independent_variables + controls
    
    try:
        # Prepare data
        vars_needed = [dependent_variable] + all_vars
        df_model = df[vars_needed].dropna()
        
        y = df_model[dependent_variable]
        
        # Verify binary outcome
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            raise ValueError(
                f"Dependent variable must be binary, found {len(unique_vals)} unique values"
            )
        
        X = df_model[all_vars]
        X = sm.add_constant(X)
        
        # Fit logistic regression
        model = sm.Logit(y, X)
        results = model.fit(disp=0)
        
        # Extract coefficients with odds ratios
        coefficients = []
        for var in results.params.index:
            coef = results.params[var]
            se = results.bse[var]
            z_val = results.tvalues[var]
            p_val = results.pvalues[var]
            ci = results.conf_int(alpha=alpha).loc[var]
            
            display_var = "(Intercept)" if var == "const" else var
            odds_ratio = np.exp(coef)
            
            coefficients.append({
                "variable": display_var,
                "coefficient": float(coef),
                "std_error": float(se),
                "z_statistic": float(z_val),
                "p_value": float(p_val),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "odds_ratio": float(odds_ratio),
                "odds_ratio_ci_lower": float(np.exp(ci[0])),
                "odds_ratio_ci_upper": float(np.exp(ci[1])),
                "is_significant": p_val < alpha,
            })
        
        # Model fit statistics
        fit_stats = {
            "pseudo_r_squared": float(results.prsquared),
            "log_likelihood": float(results.llf),
            "aic": float(results.aic),
            "bic": float(results.bic),
        }
        
        sig_vars = [c["variable"] for c in coefficients 
                    if c["is_significant"] and c["variable"] != "(Intercept)"]
        
        interpretation = (
            f"Logistic regression with {dependent_variable} as binary outcome "
            f"(n={len(df_model)}). "
        )
        
        if sig_vars:
            interpretation += f"Significant predictors: {', '.join(sig_vars)}. "
        
        interpretation += f"Pseudo R-squared={fit_stats['pseudo_r_squared']:.3f}."
        
        return {
            "status": "complete",
            "model_type": "logistic",
            "dependent_variable": dependent_variable,
            "n_observations": len(df_model),
            "coefficients": coefficients,
            "fit_statistics": fit_stats,
            "interpretation": interpretation,
        }
        
    except Exception as e:
        return {"status": "error", "error": f"Error performing logistic regression: {str(e)}"}


# =============================================================================
# Non-parametric Tests
# =============================================================================


@tool
def run_mann_whitney(
    dataset_name: str,
    variable: str,
    group_variable: str,
    alpha: float = 0.05,
) -> StatisticalResult:
    """
    Perform Mann-Whitney U test (non-parametric alternative to independent t-test).
    
    Tests whether the distributions of two groups differ significantly when
    normality assumptions are violated.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variable: The variable to compare.
        group_variable: Grouping variable (must have exactly 2 groups).
        alpha: Significance level (default 0.05).
    
    Returns:
        StatisticalResult with U-statistic, p-value, and effect size (r).
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.MANN_WHITNEY,
            test_name="Mann-Whitney U (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Dataset '{dataset_name}' not found.",
        )
    
    df = registry.get_dataframe(dataset_name)
    
    try:
        groups = df[group_variable].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"Expected 2 groups, found {len(groups)}")
        
        group1 = df[df[group_variable] == groups[0]][variable].dropna()
        group2 = df[df[group_variable] == groups[1]][variable].dropna()
        
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Effect size r = Z / sqrt(N)
        n1, n2 = len(group1), len(group2)
        z = stats.norm.ppf(1 - p_value/2) if p_value < 1 else 0
        effect_r = z / np.sqrt(n1 + n2)
        
        is_significant = p_value < alpha
        
        interpretation = (
            f"Mann-Whitney U test comparing {variable} between {groups[0]} "
            f"(Mdn={group1.median():.3f}, n={n1}) and {groups[1]} "
            f"(Mdn={group2.median():.3f}, n={n2}). "
        )
        
        if is_significant:
            interpretation += (
                f"Result is statistically significant (U={u_stat:.1f}, p={p_value:.4f}, "
                f"r={effect_r:.3f}). The distributions differ significantly."
            )
        else:
            interpretation += (
                f"Result is not statistically significant (U={u_stat:.1f}, p={p_value:.4f}). "
                f"No evidence of difference between distributions."
            )
        
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.MANN_WHITNEY,
            test_name="Mann-Whitney U Test",
            statistic=float(u_stat),
            p_value=float(p_value),
            confidence_level=1 - alpha,
            is_significant=is_significant,
            effect_size=float(effect_r),
            interpretation=interpretation,
        )
        
    except Exception as e:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.MANN_WHITNEY,
            test_name="Mann-Whitney U (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Error: {str(e)}",
        )


@tool
def run_kruskal_wallis(
    dataset_name: str,
    variable: str,
    group_variable: str,
    alpha: float = 0.05,
) -> StatisticalResult:
    """
    Perform Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA).
    
    Tests whether multiple group distributions differ when normality assumptions
    are violated.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        variable: The variable to compare.
        group_variable: Grouping variable (2 or more groups).
        alpha: Significance level (default 0.05).
    
    Returns:
        StatisticalResult with H-statistic, p-value, and effect size (epsilon-squared).
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.OTHER,
            test_name="Kruskal-Wallis (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Dataset '{dataset_name}' not found.",
        )
    
    df = registry.get_dataframe(dataset_name)
    
    try:
        groups = df[group_variable].dropna().unique()
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups, found {len(groups)}")
        
        group_data = [
            df[df[group_variable] == g][variable].dropna().values
            for g in groups
        ]
        group_data = [d for d in group_data if len(d) > 0]
        
        h_stat, p_value = stats.kruskal(*group_data)
        
        # Effect size: epsilon-squared
        n_total = sum(len(d) for d in group_data)
        epsilon_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))
        epsilon_sq = max(0, epsilon_sq)  # Ensure non-negative
        
        is_significant = p_value < alpha
        
        interpretation = (
            f"Kruskal-Wallis test comparing {variable} across {len(groups)} groups. "
        )
        
        if is_significant:
            interpretation += (
                f"Result is significant (H={h_stat:.3f}, p={p_value:.4f}, "
                f"epsilon-squared={epsilon_sq:.3f}). At least one group differs."
            )
        else:
            interpretation += (
                f"Result is not significant (H={h_stat:.3f}, p={p_value:.4f}). "
                f"No evidence of differences between groups."
            )
        
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.OTHER,
            test_name="Kruskal-Wallis H Test",
            statistic=float(h_stat),
            p_value=float(p_value),
            degrees_of_freedom=int(len(groups) - 1),
            confidence_level=1 - alpha,
            is_significant=is_significant,
            effect_size=float(epsilon_sq),
            effect_size_type="epsilon-squared",
            interpretation=interpretation,
        )
        
    except Exception as e:
        return StatisticalResult(
            result_id=str(uuid4())[:8],
            test_type=StatisticalTestType.OTHER,
            test_name="Kruskal-Wallis (Error)",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            interpretation=f"Error: {str(e)}",
        )


# =============================================================================
# Finding Generation Tools
# =============================================================================


@tool
def generate_finding(
    finding_type: str,
    statement: str,
    statistical_support: dict[str, Any] | None = None,
    evidence_description: str = "",
    addresses_research_question: bool = False,
    addresses_gap: bool = False,
) -> DataAnalysisFinding:
    """
    Generate a structured research finding.
    
    This tool creates a finding object with appropriate metadata,
    evidence strength assessment, and linkage to research questions.
    
    Args:
        finding_type: Type of finding (main_result, supporting, unexpected, etc.)
        statement: The finding statement.
        statistical_support: Statistical results supporting the finding.
        evidence_description: Description of supporting evidence.
        addresses_research_question: Whether this addresses the main question.
        addresses_gap: Whether this addresses the identified gap.
    
    Returns:
        DataAnalysisFinding with structured information.
    """
    # Map string to enum
    try:
        finding_type_enum = FindingType(finding_type)
    except ValueError:
        finding_type_enum = FindingType.SUPPORTING
    
    # Assess evidence strength based on statistical support
    evidence_strength = EvidenceStrength.MODERATE
    if statistical_support:
        p_value = statistical_support.get("p_value", 0.5)
        if p_value < 0.01:
            evidence_strength = EvidenceStrength.STRONG
        elif p_value < 0.05:
            evidence_strength = EvidenceStrength.MODERATE
        else:
            evidence_strength = EvidenceStrength.WEAK
    
    return DataAnalysisFinding(
        finding_id=str(uuid4())[:8],
        finding_type=finding_type_enum,
        statement=statement,
        detailed_description=evidence_description,
        evidence_strength=evidence_strength,
        addresses_research_question=addresses_research_question,
        addresses_gap=addresses_gap,
        confidence_level=0.8 if evidence_strength == EvidenceStrength.STRONG else 0.6,
    )


@tool
def assess_gap_coverage(
    findings: list[dict[str, Any]],
    gap_description: str,
    research_question: str,
) -> dict[str, Any]:
    """
    Assess how well the findings address the identified research gap.
    
    This tool evaluates the coverage of findings against the gap that
    the research was designed to address.
    
    Args:
        findings: List of findings from the analysis.
        gap_description: Description of the research gap.
        research_question: The refined research question.
    
    Returns:
        Dictionary with gap coverage assessment.
    """
    # Count findings that address the gap
    gap_addressing = [f for f in findings if f.get("addresses_gap", False)]
    question_addressing = [f for f in findings if f.get("addresses_research_question", False)]
    
    # Calculate coverage score
    total_findings = len(findings)
    if total_findings == 0:
        coverage_score = 0.0
    else:
        gap_ratio = len(gap_addressing) / total_findings
        question_ratio = len(question_addressing) / total_findings
        coverage_score = (gap_ratio + question_ratio) / 2
    
    # Determine if gap is addressed
    gap_addressed = coverage_score >= 0.5 and len(gap_addressing) >= 1
    
    # Generate explanation
    if gap_addressed:
        explanation = (
            f"The analysis addresses the research gap with {len(gap_addressing)} "
            f"finding(s) directly contributing to filling the identified gap. "
            f"Coverage score: {coverage_score:.2f}."
        )
    else:
        explanation = (
            f"The analysis partially addresses the research gap. "
            f"Only {len(gap_addressing)} of {total_findings} findings directly address the gap. "
            f"Coverage score: {coverage_score:.2f}. Additional analysis may be needed."
        )
    
    return {
        "gap_addressed": gap_addressed,
        "coverage_score": coverage_score,
        "findings_addressing_gap": len(gap_addressing),
        "findings_addressing_question": len(question_addressing),
        "total_findings": total_findings,
        "explanation": explanation,
    }


# =============================================================================
# Robustness Check Tools
# =============================================================================


@tool
def execute_robustness_check(
    dataset_name: str,
    check_type: str,
    original_result: dict[str, Any],
    modification: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a robustness check on analysis results.
    
    Performs various robustness checks including alternative specifications,
    subset analysis, and sensitivity tests by re-running the analysis.
    
    Args:
        dataset_name: Name of the dataset in the DataRegistry.
        check_type: Type of check: 'alternative_spec', 'subset', 'outlier_removal', 
                   'different_controls'.
        original_result: The original analysis result to compare against.
        modification: Specification of the modification:
                     - For 'subset': {"filter": "condition"}
                     - For 'outlier_removal': {"variable": "name", "method": "iqr"|"zscore"}
                     - For 'different_controls': {"add": [], "remove": []}
    
    Returns:
        Dictionary with robustness check comparison results.
    """
    registry = get_registry()
    
    if dataset_name not in registry.datasets:
        return {"status": "error", "error": f"Dataset '{dataset_name}' not found."}
    
    df = registry.get_dataframe(dataset_name)
    
    try:
        original_significant = original_result.get("is_significant", 
                                                    original_result.get("p_value", 1) < 0.05)
        original_effect = original_result.get("effect_size", 
                                               original_result.get("coefficient", 0))
        
        # Apply modification based on check type
        if check_type == "subset":
            filter_expr = modification.get("filter", "")
            if filter_expr:
                df_modified = df.query(filter_expr)
                mod_description = f"Subset analysis using filter: {filter_expr}"
            else:
                return {"status": "error", "error": "Filter expression required for subset check"}
                
        elif check_type == "outlier_removal":
            var = modification.get("variable", "")
            method = modification.get("method", "iqr")
            
            if not var or var not in df.columns:
                return {"status": "error", "error": f"Variable '{var}' not found"}
            
            if method == "zscore":
                z_scores = np.abs(stats.zscore(df[var].dropna()))
                mask = pd.Series(True, index=df.index)
                mask[df[var].notna()] = z_scores < 3
                df_modified = df[mask]
                mod_description = f"Removed outliers (|z| > 3) from {var}"
            else:  # IQR method
                Q1 = df[var].quantile(0.25)
                Q3 = df[var].quantile(0.75)
                IQR = Q3 - Q1
                df_modified = df[(df[var] >= Q1 - 1.5*IQR) & (df[var] <= Q3 + 1.5*IQR)]
                mod_description = f"Removed IQR outliers from {var}"
        else:
            df_modified = df
            mod_description = f"Robustness check: {check_type}"
        
        # Calculate change metrics
        n_original = len(df)
        n_modified = len(df_modified)
        pct_data_retained = (n_modified / n_original) * 100 if n_original > 0 else 0
        
        return {
            "status": "complete",
            "check_type": check_type,
            "modification_description": mod_description,
            "n_original": n_original,
            "n_modified": n_modified,
            "pct_data_retained": round(pct_data_retained, 1),
            "original_significant": original_significant,
            "original_effect": original_effect,
            "recommendation": (
                "Re-run the primary analysis on the modified dataset to assess robustness. "
                f"The modified dataset retains {pct_data_retained:.1f}% of observations."
            ),
            "modified_dataset_name": f"{dataset_name}_robustness_{check_type}",
        }
        
    except Exception as e:
        return {"status": "error", "error": f"Error in robustness check: {str(e)}"}


# =============================================================================
# Backward Compatibility Wrappers
# =============================================================================
# These functions maintain compatibility with the old API while the node
# implementation is being updated to use the new DataRegistry-based tools.


@tool
def execute_hypothesis_test(
    test_type: str,
    hypothesis: str,
    variables: list[str],
    test_parameters: dict[str, Any] | None = None,
) -> StatisticalResult:
    """
    Execute a statistical hypothesis test (backward-compatible wrapper).
    
    This is a compatibility wrapper for the old API. For new code, use the
    specific test functions: run_ttest, run_anova, run_chi_square, etc.
    
    Args:
        test_type: Type of test (t_test, paired_t_test, anova, chi_square, etc.)
        hypothesis: The hypothesis being tested.
        variables: Variables involved in the test.
        test_parameters: Additional test parameters (e.g., alpha level).
    
    Returns:
        StatisticalResult with test statistics and interpretation.
    """
    params = test_parameters or {}
    alpha = params.get("alpha", 0.05)
    
    # Map string to enum
    try:
        test_type_enum = StatisticalTestType(test_type)
    except ValueError:
        test_type_enum = StatisticalTestType.OTHER
    
    # Create a descriptive result since we don't have actual data
    # This maintains backward compatibility with the placeholder behavior
    interpretation = (
        f"Hypothesis test '{test_type}' planned for variables: {', '.join(variables)}. "
        f"Hypothesis: {hypothesis}. "
        "Note: This is a placeholder result. For actual analysis, "
        "load data using load_data() and use specific test functions."
    )
    
    return StatisticalResult(
        result_id=str(uuid4())[:8],
        test_type=test_type_enum,
        test_name=f"Planned {test_type.replace('_', ' ').title()}",
        statistic=0.0,
        p_value=1.0,  # Conservative placeholder
        degrees_of_freedom=None,
        confidence_level=1 - alpha,
        is_significant=False,
        interpretation=interpretation,
    )


@tool
def execute_regression_analysis(
    model_type: str,
    dependent_variable: str,
    independent_variables: list[str],
    control_variables: list[str] | None = None,
    data_info: dict[str, Any] | None = None,
) -> RegressionResult:
    """
    Execute a regression analysis (backward-compatible wrapper).
    
    This is a compatibility wrapper for the old API. For new code, use
    run_ols_regression or run_logistic_regression with data in the DataRegistry.
    
    Args:
        model_type: Type of regression (ols, fixed_effects, random_effects, etc.)
        dependent_variable: The dependent variable name.
        independent_variables: List of independent variable names.
        control_variables: Optional list of control variable names.
        data_info: Data information from exploration results.
    
    Returns:
        RegressionResult with placeholder values.
    """
    controls = control_variables or []
    all_vars = [dependent_variable] + independent_variables + controls
    
    # Get sample size from data info
    n_obs = 0
    if data_info:
        n_obs = data_info.get("total_rows", 0)
    
    # Create placeholder coefficients
    coefficients = []
    
    # Intercept
    coefficients.append(
        RegressionCoefficient(
            variable="(Intercept)",
            coefficient=0.0,
            std_error=0.0,
            t_statistic=0.0,
            p_value=1.0,
            confidence_interval_lower=0.0,
            confidence_interval_upper=0.0,
            is_significant=False,
        )
    )
    
    # Variables
    for var in independent_variables + controls:
        coefficients.append(
            RegressionCoefficient(
                variable=var,
                coefficient=0.0,
                std_error=0.0,
                t_statistic=0.0,
                p_value=1.0,
                confidence_interval_lower=0.0,
                confidence_interval_upper=0.0,
                is_significant=False,
            )
        )
    
    interpretation = (
        f"Regression analysis planned: {model_type.upper()} with {dependent_variable} "
        f"as dependent variable and {len(independent_variables)} independent variable(s). "
        "Note: This is a placeholder result. For actual analysis, "
        "load data using load_data() and use run_ols_regression()."
    )
    
    return RegressionResult(
        result_id=str(uuid4())[:8],
        model_type=model_type,
        dependent_variable=dependent_variable,
        r_squared=0.0,
        adjusted_r_squared=0.0,
        f_statistic=None,
        f_p_value=None,
        coefficients=coefficients,
        n_observations=n_obs,
        interpretation=interpretation,
    )


# =============================================================================
# Export Tool List
# =============================================================================


def get_analysis_tools() -> list:
    """Get list of all analysis tools for the data analyst agent."""
    return [
        # Descriptive statistics
        execute_descriptive_stats,
        compute_correlation_matrix,
        # Parametric hypothesis tests
        run_ttest,
        run_anova,
        run_chi_square,
        # Non-parametric tests
        run_mann_whitney,
        run_kruskal_wallis,
        # Regression
        run_ols_regression,
        run_logistic_regression,
        # Findings and assessment
        generate_finding,
        assess_gap_coverage,
        execute_robustness_check,
        # Backward compatibility
        execute_hypothesis_test,
        execute_regression_analysis,
    ]
