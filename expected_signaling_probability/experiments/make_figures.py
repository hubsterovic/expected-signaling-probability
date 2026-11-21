"""Regenerate every paper figure and table from the run cache (computing anything missing)."""
from expected_signaling_probability.experiments import fig_asymmetric, fig_ratio, fig_symmetric, table_exponents

if __name__ == "__main__":
    fig_symmetric.main()
    fig_ratio.main()
    fig_asymmetric.main()
    table_exponents.main()
