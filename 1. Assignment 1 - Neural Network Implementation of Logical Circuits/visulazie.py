import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
#  PART 2.1 - MULTI-INPUT LOGIC GATE CLASSIFICATION
# ------------------------------------------------------------------------------


def print_truth_table_21(X, y, preds):
    nin = X.shape[1]
    col_labels = [chr(65 + k) for k in range(nin)]
    hdr = ("  " + "  ".join(f"{c:>5}" for c in col_labels)
           + "    Target   Pred   Correct")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i in range(len(X)):
        row_str = "  ".join(f"{int(v):>5}" for v in X[i])
        ok = "yes" if preds[i] == y[i] else " NO"
        print(f"  {row_str}    {int(y[i]):>6}  {int(preds[i]):>5}   {ok}")


def plot_mse_21(ax, mse_log, perf_ep, title, is_bip):
    color = "steelblue" if not is_bip else "darkorange"
    ax.semilogy(mse_log, lw=1.5, color=color, label="MSE")
    if perf_ep:
        ax.axvline(perf_ep, color="green", ls="--", lw=1.3,
                   label=f"100% acc @ ep {perf_ep}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def print_comparison_table_21(TASKS, REPRESENTATIONS, results):
    print("\n" + "=" * 70)
    print("  CONVERGENCE SPEED COMPARISON")
    print("=" * 70)
    print(f"  {'Function':<38}  {'Repr':<9}  Accuracy  First-100%-Epoch")
    print("  " + "-" * 68)

    for fi, (fname, _, _) in enumerate(TASKS):
        for ri, (rep, _) in enumerate(REPRESENTATIONS):
            acc, pe = results[(fi, ri)]
            pe_str = str(pe) if pe else "---"
            print(f"  {fname:<38}  {rep:<9}  {acc * 100:>7.1f}%  {pe_str:>16}")


#  ------------------------------------------------------------------------------
#  PART 2.2 - HOPFIELD NETWORK: SR LATCH
#  ------------------------------------------------------------------------------
def print_weights_and_biases_22(hopfield_net):
    print(f"\n  Weight matrix W:\n{hopfield_net.W}")
    print("\n  Bias vectors for each SR condition:")
    for S, R, name in [(1, -1, "Set"),
                       (-1, 1, "Reset"),
                       (-1, -1, "Hold"),
                       (1,  1, "Invalid")]:
        print(
            f"    {name:9s} (S={S:+2d}, R={R:+2d}): x = {hopfield_net.get_external_signal(S, R)}")


def print_summary_22(func_valid, noise_valid):
    nfv, nft = sum(func_valid),  len(func_valid)
    nnv, nnt = sum(noise_valid), len(noise_valid)
    print(
        f"  Functional tests: {nfv}/{nft} = {100*nfv/nft:.1f}% converged to valid SR state")
    print(
        f"  Noise tests     : {nnv}/{nnt} = {100*nnv/nnt:.1f}% converged to valid SR state")
    print(
        f"  Overall         : {nfv+nnv}/{nft+nnt} = {100*(nfv+nnv)/(nft+nnt):.1f}%")


def plot_energy_landscape_22(hopfield_net):
    all_states = [np.array([1., -1.]),
                  np.array([-1.,  1.]),
                  np.array([1.,  1.]),
                  np.array([-1., -1.])]
    slabels = ["[+1,-1]\n(Set)", "[-1,+1]\n(Reset)",
               "[+1,+1]\n(Inv+)", "[-1,-1]\n(Inv-)"]
    sr_plot = [(1, -1, "Set"), (-1, 1, "Reset"),
               (-1, -1, "Hold"), (1,  1, "Invalid")]
    colors_bar = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig_e, ax_e = plt.subplots(figsize=(11, 5))
    xp = np.arange(len(all_states))
    bw = 0.2
    for k, (S, R, cname) in enumerate(sr_plot):
        energies = [hopfield_net.calc_energy(y, S, R) for y in all_states]
        ax_e.bar(xp + k * bw, energies, bw,
                 label=f"{cname} (S={S:+},R={R:+})",
                 color=colors_bar[k], alpha=0.85)

    ax_e.set_xticks(xp + 1.5 * bw)
    ax_e.set_xticklabels(slabels, fontsize=11)
    ax_e.axhline(0, color="k", lw=0.8, ls="--")
    ax_e.set_ylabel("Hopfield Energy  E(s)", fontsize=11)
    ax_e.set_title(
        "Energy Landscape: Each State Under Each SR Condition\n"
        "(stable attractors have the lowest energy)",
        fontsize=12, fontweight="bold"
    )
    ax_e.legend(fontsize=9, loc="upper right")
    ax_e.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("part22_energy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  [Figure saved: part22_energy.png]")
