import itertools
from backtester import run_backtest
from logger import init_logger

logger = init_logger("grid_tuner.log")


def grid_search():
    volume_spike_range = [1.3, 1.5, 1.7]
    ml_confidence_range = [0.5, 0.6, 0.7]
    pyramid_configs = [
        {"high": 0.4, "medium": 0.25, "low": 0.15},
        {"high": 0.5, "medium": 0.3, "low": 0.2},
    ]

    results = []
    for vol_thr, ml_thr, pyr in itertools.product(
        volume_spike_range, ml_confidence_range, pyramid_configs
    ):
        metrics = run_backtest(
            volume_spike_threshold=vol_thr,
            ml_confidence_threshold=ml_thr,
            pyramid_levels=pyr,
        )
        logger.info(f"VOL={vol_thr} ML={ml_thr} PYR={pyr} => {metrics}")
        results.append((vol_thr, ml_thr, pyr, metrics))
    return results


if __name__ == "__main__":
    grid_search()

