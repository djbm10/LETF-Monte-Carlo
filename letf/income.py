"""
Income trajectory simulation module.

Provides Monte Carlo simulation of income growth with realistic career dynamics,
including promotions, job changes, layoffs, and retirement transitions.

Minimal dependencies to avoid circular imports.
"""

import numpy as np
from typing import Dict, Optional


def simulate_income_trajectory(base_income, years, num_simulations=50,
                              career_stage='mid', aggressive=True, seed=None):
    """
    Monte Carlo simulation of income growth with realistic career dynamics.

    Args:
        base_income: Starting annual income (e.g., $150,000)
        years: Number of years to simulate
        num_simulations: Number of income paths to generate
        career_stage: 'early' (20s-30s), 'mid' (30s-40s), 'late' (40s-50s)
        aggressive: If True, uses more optimistic growth assumptions
        seed: Random seed for reproducibility

    Returns:
        dict with keys:
            'p10', 'p25', 'p50', 'p75', 'p90': Income trajectories at percentiles
            'mean': Average trajectory
            'all_paths': All simulated paths (for analysis)
    """
    rng = np.random.default_rng(seed)

    # ========================================================================
    # CAREER STAGE PARAMETERS
    # ========================================================================
    career_params = {
        'early': {  # 20s-early 30s: Rapid growth, high volatility
            'base_growth': 0.06 if aggressive else 0.05,
            'growth_volatility': 0.08,
            'promotion_rate': 0.25,  # 25% chance per year
            'promotion_boost': (0.15, 0.25),  # 15-25% raise
            'job_change_rate': 0.15,  # 15% chance (high mobility)
            'job_change_boost': (0.10, 0.30),  # 10-30% raise on job change
            'layoff_rate': 0.03,
            'layoff_penalty': 0.20,
            'recovery_years': 1.5
        },
        'mid': {  # 30s-40s: Steady growth, moderate volatility
            'base_growth': 0.04 if aggressive else 0.03,
            'growth_volatility': 0.05,
            'promotion_rate': 0.15,  # 15% chance per year
            'promotion_boost': (0.12, 0.20),  # 12-20% raise
            'job_change_rate': 0.08,  # 8% chance (more stable)
            'job_change_boost': (0.08, 0.20),  # 8-20% raise
            'layoff_rate': 0.02,
            'layoff_penalty': 0.15,
            'recovery_years': 2.0
        },
        'late': {  # 40s-50s: Slower growth, low volatility, near peak
            'base_growth': 0.025 if aggressive else 0.02,
            'growth_volatility': 0.03,
            'promotion_rate': 0.08,  # 8% chance (fewer opportunities)
            'promotion_boost': (0.08, 0.15),  # 8-15% raise
            'job_change_rate': 0.04,  # 4% chance (rare)
            'job_change_boost': (0.05, 0.15),  # 5-15% raise
            'layoff_rate': 0.015,
            'layoff_penalty': 0.12,
            'recovery_years': 2.5
        }
    }

    params = career_params[career_stage]

    # ========================================================================
    # RUN SIMULATIONS
    # ========================================================================
    all_paths = []

    for sim in range(num_simulations):
        income_path = [base_income]
        income = base_income
        recovering_from_layoff = 0  # Counter for recovery years

        for year in range(years):
            # Base growth (inflation + merit increases)
            annual_growth = params['base_growth']

            # Add random volatility (bonuses, cost of living adjustments, etc.)
            random_variation = rng.normal(0, params['growth_volatility'])
            annual_growth += random_variation

            # ================================================================
            # CAREER EVENTS (mutually exclusive)
            # ================================================================
            event_roll = rng.random()

            if recovering_from_layoff > 0:
                # In recovery mode - accelerated catch-up growth
                catch_up_boost = 0.10  # Extra 10% during recovery
                annual_growth += catch_up_boost
                recovering_from_layoff -= 1

            elif event_roll < params['layoff_rate']:
                # LAYOFF - rare but impactful
                annual_growth -= params['layoff_penalty']
                recovering_from_layoff = int(params['recovery_years'])

            elif event_roll < params['layoff_rate'] + params['promotion_rate']:
                # PROMOTION - significant raise
                promotion_raise = rng.uniform(*params['promotion_boost'])
                annual_growth += promotion_raise

            elif event_roll < params['layoff_rate'] + params['promotion_rate'] + params['job_change_rate']:
                # JOB CHANGE - often leads to higher pay
                job_change_raise = rng.uniform(*params['job_change_boost'])
                annual_growth += job_change_raise

            # Apply growth
            income *= (1 + annual_growth)

            # Floor: income can't drop below 50% of base (safety net / severance)
            income = max(income, base_income * 0.5)

            # Ceiling: realistic income cap (nobody goes from $150k -> $10M in 20 years)
            # Cap at 5x starting income for conservative estimate
            income = min(income, base_income * 5)

            income_path.append(income)

        all_paths.append(income_path)

    # ========================================================================
    # CALCULATE PERCENTILES
    # ========================================================================
    all_paths = np.array(all_paths)

    result = {
        'p10': np.percentile(all_paths, 10, axis=0),
        'p25': np.percentile(all_paths, 25, axis=0),
        'p50': np.percentile(all_paths, 50, axis=0),
        'p75': np.percentile(all_paths, 75, axis=0),
        'p90': np.percentile(all_paths, 90, axis=0),
        'mean': np.mean(all_paths, axis=0),
        'all_paths': all_paths
    }

    return result


def get_year_income(income_trajectory, year):
    """
    Get income for a specific year from trajectory.

    Args:
        income_trajectory: Output from simulate_income_trajectory (use 'p50' for median)
        year: Year index (0-based)

    Returns:
        Income for that year
    """
    if year >= len(income_trajectory):
        # Beyond trajectory - use last year with inflation
        years_beyond = year - len(income_trajectory) + 1
        return income_trajectory[-1] * (1.02 ** years_beyond)

    return income_trajectory[year]
