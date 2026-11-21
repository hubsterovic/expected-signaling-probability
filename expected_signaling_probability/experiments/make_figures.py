"""Regenerate every paper figure and table from the run cache (computing anything missing)."""
from expected_signaling_probability.experiments import fig_asymmetric, fig_ratio, fig_symmetric, table_exponents
from expected_signaling_probability.experiments.common import PAPER_GLOBAL_OPS

if __name__ == "__main__":
    # main text: BCSZ only
    fig_symmetric.main()
    fig_ratio.main_bcsz()
    fig_asymmetric.main()
    # supplementary: all ensembles
    fig_symmetric.main(global_ops=PAPER_GLOBAL_OPS, suffix="_all")
    fig_ratio.main(suffix="_all")
    fig_asymmetric.main(global_ops=PAPER_GLOBAL_OPS, suffix="_all")
    table_exponents.main()
