import numpy as np
from line_profiler import LineProfiler
from memory_profiler import profile

import meta_learning
import signals


def main() -> None:
    lp = LineProfiler()
    lp.add_function(signals.rolling_mean)
    if hasattr(meta_learning, "retrain_meta_learner"):
        lp.add_function(meta_learning.retrain_meta_learner)
    lp.enable_by_count()
    signals.rolling_mean(np.random.rand(1000), 20)
    lp.print_stats()


if __name__ == "__main__":
    main()
