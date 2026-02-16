"""Diagnostic script to isolate where simulated returns go wrong."""
import sys
sys.path.insert(0, '.')
import numpy as np

# Suppress all the prints from module-level code
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from LETF34_analysis import (
        simulate_joint_returns_t, calibrate_joint_return_model,
        simulate_regime_path_semi_markov, compute_letf_return_correct,
        compute_daily_financing_series, calculate_daily_borrow_cost,
        generate_tracking_error_ar1, simulate_etf_returns_from_layers,
        simulate_latent_stress_state, map_underlying_series_for_asset,
        generate_fat_tailed_returns, predict_borrow_spread_series,
        ASSETS, SIM_ENGINE_MODE, N_REGIMES
    )

rng = np.random.default_rng(42)
sim_years = 30
sim_days = sim_years * 252

print("="*70)
print("DIAGNOSTIC: Tracing simulation pipeline layer by layer")
print("="*70)
print(f"SIM_ENGINE_MODE = {SIM_ENGINE_MODE}")
print(f"sim_days = {sim_days} ({sim_years} years)")

# === LAYER 0: Simple regime path (80% low vol) ===
regime_path = np.zeros(sim_days, dtype=int)
# 20% high vol
high_vol_days = rng.choice(sim_days, size=int(sim_days*0.2), replace=False)
regime_path[high_vol_days] = 1
print(f"\nRegime: {np.mean(regime_path==0)*100:.0f}% low vol, {np.mean(regime_path==1)*100:.0f}% high vol")

# === LAYER 1: Test joint return model output ===
print("\n--- LAYER 1: Joint Return Model (simulate_joint_returns_t) ---")

# Create a simple calibrated model manually with known-good params
# Using historical SPY: ~0.04%/day mean, ~1%/day std
test_model = {
    'assets': ['SPY_Ret', 'QQQ_Ret', 'TLT_Ret'],
    'regimes': {
        0: {
            'mu': np.array([0.08/252, 0.10/252, 0.03/252]),  # ~8%, 10%, 3% annual
            'cov': np.diag([0.16**2/252, 0.24**2/252, 0.12**2/252]),
            'nu': 5.0,
            'garch_alpha': 0.06, 'garch_beta': 0.90,
            'dcc_a': 0.02, 'dcc_b': 0.95,
        },
        1: {
            'mu': np.array([-0.05/252, -0.08/252, 0.05/252]),  # Neg in high vol
            'cov': np.diag([0.28**2/252, 0.42**2/252, 0.16**2/252]),
            'nu': 4.0,
            'garch_alpha': 0.09, 'garch_beta': 0.86,
            'dcc_a': 0.04, 'dcc_b': 0.90,
        }
    }
}

generated = simulate_joint_returns_t(sim_days, regime_path, test_model, rng)
for asset_key in ['SPY_Ret', 'QQQ_Ret', 'TLT_Ret']:
    rets = generated[asset_key]
    cum = np.prod(1 + rets)
    cagr = cum**(1/sim_years) - 1
    daily_mean = np.mean(rets)
    daily_std = np.std(rets)
    annual_vol = daily_std * np.sqrt(252)
    print(f"  {asset_key}: daily_mean={daily_mean*10000:.2f}bps, daily_std={daily_std*10000:.1f}bps, "
          f"annual_vol={annual_vol*100:.1f}%, cum_return={cum:.2f}x, CAGR={cagr*100:.2f}%")

# === LAYER 2: Test SPY as 1x ETF (no leverage costs) ===
print("\n--- LAYER 2: SPY through compute_letf_return_correct (1x, no borrow) ---")
spy_underlying = generated['SPY_Ret']
spy_config = ASSETS['SPY']
expense = spy_config['expense_ratio']
spy_etf_rets = np.array([
    compute_letf_return_correct(spy_underlying[t], 1.0, 0, expense, 0)
    for t in range(sim_days)
])
cum_spy = np.prod(1 + spy_etf_rets)
cagr_spy = cum_spy**(1/sim_years) - 1
print(f"  SPY ETF: cum={cum_spy:.2f}x, CAGR={cagr_spy*100:.2f}%")
print(f"  Annual expense drag: {expense*100:.2f}%")

# === LAYER 3: Test TQQQ with financing ===
print("\n--- LAYER 3: TQQQ financing costs ---")
# Simulate reasonable risk-free rates
risk_free = np.where(regime_path == 0, 0.045, 0.015)  # 4.5% low vol, 1.5% high vol
daily_borrow = np.array([
    calculate_daily_borrow_cost(3.0, risk_free[t], 0.0075)
    for t in range(sim_days)
])
print(f"  Mean daily borrow cost: {np.mean(daily_borrow)*10000:.2f} bps")
print(f"  Annual borrow cost: {np.mean(daily_borrow)*252*100:.2f}%")

qqq_underlying = generated['QQQ_Ret']
tqqq_rets = np.array([
    compute_letf_return_correct(qqq_underlying[t], 3.0, 0, 0.0086, daily_borrow[t])
    for t in range(sim_days)
])
cum_tqqq = np.prod(1 + tqqq_rets)
cagr_tqqq = cum_tqqq**(1/sim_years) - 1
print(f"  TQQQ (no tracking error): cum={cum_tqqq:.4f}x, CAGR={cagr_tqqq*100:.2f}%")

# === LAYER 4: Test tracking error magnitude ===
print("\n--- LAYER 4: Tracking error magnitude ---")
vix = np.where(regime_path == 0, 15.0, 35.0)
te = generate_tracking_error_ar1(
    sim_days, regime_path, vix, spy_underlying,
    base_te=0.0002, df_param=5, rng=rng
)
print(f"  TE daily mean: {np.mean(te)*10000:.2f} bps")
print(f"  TE daily std: {np.std(te)*10000:.1f} bps")
print(f"  TE daily |mean|: {np.mean(np.abs(te))*10000:.2f} bps")
print(f"  TE min/max: {np.min(te)*100:.4f}% / {np.max(te)*100:.4f}%")

# What does TE do to cumulative return?
spy_with_te = (1 + spy_etf_rets) * (1 + te) - 1
cum_with_te = np.prod(1 + spy_with_te)
cagr_with_te = cum_with_te**(1/sim_years) - 1
print(f"  SPY with TE: cum={cum_with_te:.2f}x, CAGR={cagr_with_te*100:.2f}%")
print(f"  TE impact on CAGR: {(cagr_with_te - cagr_spy)*100:.2f}%")

# === LAYER 5: Test stress state jumps ===
print("\n--- LAYER 5: Stress state jump impact ---")
# Test with default stress model
stress_model = {
    'regimes': {
        0: {'liq_mu': 0.10, 'liq_phi': 0.90, 'liq_sigma': 0.08,
            'credit_mu': 0.05, 'credit_phi': 0.88, 'credit_sigma': 0.07,
            'jump_base_prob': 0.001, 'jump_scale': 0.010},
        1: {'liq_mu': 0.35, 'liq_phi': 0.90, 'liq_sigma': 0.08,
            'credit_mu': 0.25, 'credit_phi': 0.88, 'credit_sigma': 0.07,
            'jump_base_prob': 0.004, 'jump_scale': 0.018}
    }
}
stress_state = simulate_latent_stress_state(sim_days, regime_path, stress_model, vix, rng)
jumps = stress_state['jump']
print(f"  Jump days: {np.sum(jumps > 0)} out of {sim_days} ({np.mean(jumps > 0)*100:.1f}%)")
print(f"  Mean jump size (when occurs): {np.mean(jumps[jumps > 0])*100:.2f}%")
print(f"  Total jump drag (annual): {np.sum(jumps)/sim_years*100:.2f}%")

# Apply jumps to SPY returns (as done in simulate_single_path_fixed)
spy_after_jumps = np.clip(spy_underlying - jumps, -0.95, 3.0)
cum_after_jumps = np.prod(1 + spy_after_jumps)
cagr_after_jumps = cum_after_jumps**(1/sim_years) - 1
print(f"  SPY before jumps: CAGR={cagr_spy*100:.2f}%")
print(f"  SPY after jumps: CAGR={cagr_after_jumps*100:.2f}%")
print(f"  Jump impact on CAGR: {(cagr_after_jumps - cagr_spy)*100:.2f}%")

# === LAYER 6: Full pipeline for SPY ===
print("\n--- LAYER 6: Full pipeline test (SPY, 1x leverage) ---")
spy_config_full = ASSETS['SPY']
spy_full_rets = simulate_etf_returns_from_layers(
    asset='SPY',
    config=spy_config_full,
    underlying=spy_after_jumps,  # After jumps like the real code
    regime_path=regime_path,
    vix=vix,
    risk_free_annual=risk_free,
    sim_id=42,
    funding_model=None,  # No institutional funding model
    tracking_residual_model=None,
    stress_state=stress_state
)
cum_full = np.prod(1 + spy_full_rets)
cagr_full = cum_full**(1/sim_years) - 1
print(f"  SPY full pipeline: cum={cum_full:.2f}x, CAGR={cagr_full*100:.2f}%")

# Compare: What if we skip jumps?
print("\n--- LAYER 7: SPY full pipeline WITHOUT stress jumps ---")
spy_no_jumps = simulate_etf_returns_from_layers(
    asset='SPY',
    config=spy_config_full,
    underlying=spy_underlying,  # NO jumps
    regime_path=regime_path,
    vix=vix,
    risk_free_annual=risk_free,
    sim_id=42,
    funding_model=None,
    tracking_residual_model=None,
    stress_state=None  # No stress state
)
cum_no_jumps = np.prod(1 + spy_no_jumps)
cagr_no_jumps = cum_no_jumps**(1/sim_years) - 1
print(f"  SPY without jumps/stress: cum={cum_no_jumps:.2f}x, CAGR={cagr_no_jumps*100:.2f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Raw SPY returns (joint model):     CAGR = {(np.prod(1+spy_underlying)**(1/sim_years)-1)*100:+.2f}%")
print(f"  SPY after stress jumps:            CAGR = {cagr_after_jumps*100:+.2f}%")
print(f"  SPY ETF (expenses only):           CAGR = {cagr_spy*100:+.2f}%")
print(f"  SPY ETF + tracking error:          CAGR = {cagr_with_te*100:+.2f}%")
print(f"  SPY full pipeline (with stress):   CAGR = {cagr_full*100:+.2f}%")
print(f"  SPY full pipeline (no stress):     CAGR = {cagr_no_jumps*100:+.2f}%")
print(f"\n  Expected SPY CAGR: ~+7-10%")
