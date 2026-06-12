"""Plot presence frequency maps aligned with a cost-map log.

This script is the cost-log-aligned wrapper for the Fig.10-style visualization.
It starts from a *_cost_map_logs.pickle path, infers the matching episode and
step, and then visualizes the corresponding raw_pred_image ensemble.
"""

from scripts.analysis.plot_cost_log_presence_score_maps import main


if __name__ == "__main__":
    main()
