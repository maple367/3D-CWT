"""Plot a finite-mode envelope exported by liang_70um.py.

Install the optional plotting dependency with ``python -m pip install '.[plot]'``.
The figure shows the slowly varying envelope, not the microscopic optical field.
"""
import argparse
from pathlib import Path

import numpy as np


def plot_result(directory, output, *, mode=0):
    """Write a near-field map and center-line profiles for one exported mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory, output = Path(directory), Path(output)
    with np.load(directory / "fields.npz", allow_pickle=False) as data:
        centers = data["centers"]
        eigenvalues = data["eigenvalues"]
        x = data["x_centers_um"]
        y = data["y_centers_um"]
        xf = data["x_faces_um"]
        yf = data["y_faces_um"]
        if centers.ndim != 4 or centers.shape[1:] != (4, len(y), len(x)):
            raise ValueError("Expected centers with shape (mode, 4, Ny, Nx)")
        if not 0 <= mode < len(centers) or len(eigenvalues) != len(centers):
            raise ValueError("mode must index an exported eigenpair")
        intensity = np.sum(np.abs(centers[mode]) ** 2, axis=0)
        eigenvalue = complex(eigenvalues[mode])
    if not np.all(np.isfinite(intensity)) or intensity.max() <= 0:
        raise ValueError("The exported intensity must be finite and nonzero")
    if len(xf) != len(x) + 1 or len(yf) != len(y) + 1:
        raise ValueError("Face coordinates must enclose the cell centers")
    lx, ly = float(xf[-1] - xf[0]), float(yf[-1] - yf[0])
    scaled = intensity / intensity.max()
    # Linear interpolation through cell centers, including even-sized grids.
    horizontal = np.array([np.interp((yf[0]+yf[-1])/2, y, scaled[:, i])
                           for i in range(len(x))])
    vertical = np.array([np.interp((xf[0]+xf[-1])/2, x, scaled[i, :])
                         for i in range(len(y))])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "figure.dpi": 160, "savefig.dpi": 180}):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3),
                                 gridspec_kw={"width_ratios": [1.05, 1]},
                                 layout="constrained")
        image = axes[0].pcolormesh(xf, yf, scaled, shading="flat", cmap="inferno",
                                   vmin=0, vmax=1, rasterized=True)
        axes[0].set(xlabel="x (µm)", ylabel="y (µm)",
                    title="Finite-cavity envelope", xlim=(xf[0], xf[-1]),
                    ylim=(yf[0], yf[-1]), aspect="equal")
        axes[0].set_xticks(np.linspace(xf[0], xf[-1], 5))
        axes[0].set_yticks(np.linspace(yf[0], yf[-1], 5))
        fig.colorbar(image, ax=axes[0], label="Intensity / peak", shrink=.9)
        axes[1].plot(x-xf[0], horizontal, color="#1f77b4", lw=2,
                     label="Along x, at y = Ly/2")
        axes[1].plot(y-yf[0], vertical, color="#d95f02", lw=1.8,
                     ls="--", label="Along y, at x = Lx/2")
        axes[1].set(xlabel="Position (µm)", ylabel="Intensity / peak",
                    title="Center-line profiles", xlim=(0, max(lx, ly)),
                    ylim=(0, 1.08))
        axes[1].grid(alpha=.2)
        axes[1].legend(frameon=False, loc="upper right", fontsize=9)
        axes[1].text(.04, .95,
                     f"δL = {eigenvalue.real*lx:.5f}\nαL = {eigenvalue.imag*lx:.5f}",
                     transform=axes[1].transAxes, va="top",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
        fig.suptitle(f"{lx:g} × {ly:g} µm² PCSEL · circular air holes", fontsize=14)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, metadata={"Software": "Matplotlib"})
        plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing fields.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", type=int, default=0, help="zero-based exported mode index")
    args = parser.parse_args(argv)
    output = args.output or args.directory / "selected_mode.png"
    plot_result(args.directory, output, mode=args.mode)
    print(f"Wrote {output.name}")


if __name__ == "__main__":
    main()
