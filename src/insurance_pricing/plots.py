import matplotlib.pyplot as plt
import numpy as np

from insurance_pricing import metrics

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def calibration_and_lorenz(y, exposure, predictions: dict, n_groups: int = 10):
    fig, (ax_cal, ax_lor) = plt.subplots(1, 2, figsize=(11, 4.5))

    for i, (name, mu) in enumerate(predictions.items()):
        style = {"color": "black", "linestyle": LINESTYLES[i % 4], "marker": MARKERS[i % 4]}

        table = metrics.calibration_table(y, mu, exposure, n_groups)
        ax_cal.plot(table["predicted_freq"], table["observed_freq"], label=name, markersize=4, **style)

        if np.isnan(np.asarray(mu, dtype=float)).any():
            continue
        x, lorenz = metrics.lorenz_curve(y, mu, exposure)
        g = metrics.gini(y, mu, exposure)
        ax_lor.plot(x, lorenz, label=f"{name} (Gini = {g:.3f})", color="black", linestyle=LINESTYLES[i % 4])

    lim = ax_cal.get_xlim()[1], ax_cal.get_ylim()[1]
    top = max(lim)
    ax_cal.plot([0, top], [0, top], color="gray", linewidth=0.8)
    ax_cal.set_xlabel("predicted frequency")
    ax_cal.set_ylabel("observed frequency")
    ax_cal.set_title(f"Calibration ({n_groups} equal-exposure groups)")
    ax_cal.legend(frameon=False)

    ax_lor.plot([0, 1], [0, 1], color="gray", linewidth=0.8)
    ax_lor.set_xlabel("share of exposure")
    ax_lor.set_ylabel("share of claims")
    ax_lor.set_title("Lorenz curves")
    ax_lor.legend(frameon=False)

    fig.tight_layout()
    return fig