"""Genera las figuras principales del EDA en imagenes_eda/.

Uso: python scripts/eda_figures.py
"""
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "imagenes_eda"
TARGET, FAMILY, ID = "label", "attack_cat", "id"

# Paleta categórica validada (CVD-safe): slot 1 = Normal / train, slot 2 = Ataque / test
C1, C2 = "#2a78d6", "#eb6834"
INK2 = "#52514e"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlelocation": "left",
    "axes.titlesize": 12, "font.size": 10,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  ", OUT.relative_to(ROOT) / name)


def label_text(s):
    return s.map({0: "Normal", 1: "Ataque"})


def fig_clases(train, test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    x = np.arange(2)
    for i, (lab, col, name) in enumerate([(0, C1, "Normal"), (1, C2, "Ataque")]):
        vals = [(df[TARGET] == lab).mean() * 100 for df in (train, test)]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=col, label=name)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=9, color=INK2, padding=2)
    ax.set_xticks(x, [f"train\n{len(train):,}", f"test\n{len(test):,}"])
    ax.set_ylabel("% de filas")
    ax.set_ylim(0, 80)
    ax.set_title("Proporción normal vs. ataque")
    ax.legend(frameon=False)

    ax = axes[1]
    fam = pd.DataFrame({"train": train[FAMILY].value_counts(normalize=True),
                        "test": test[FAMILY].value_counts(normalize=True)}).mul(100)
    fam = fam.drop("Normal").sort_values("train")
    y = np.arange(len(fam))
    ax.barh(y + 0.2, fam["train"], height=0.38, color=C1, label="train")
    ax.barh(y - 0.2, fam["test"], height=0.38, color=C2, label="test")
    ax.set_yticks(y, fam.index)
    ax.set_xlabel("% del total de filas del archivo")
    ax.set_title("Familias de ataque")
    counts = train[FAMILY].value_counts()
    for yi, f in zip(y, fam.index):
        ax.text(fam.loc[f, "train"] + 0.3, yi + 0.2, f"{counts[f]:,}", va="center", fontsize=8, color=INK2)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("1 · Clases: el ataque es mayoritario y su proporción cambia entre archivos",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "01_clases.png")


def fig_id(train, test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (name, df) in zip(axes, [("train", train), ("test", test)]):
        b = df.groupby(pd.cut(df[ID], 50, labels=False))[TARGET].mean() * 100
        ax.fill_between(b.index, b.values, color=C2, alpha=0.25, linewidth=0)
        ax.plot(b.index, b.values, color=C2, linewidth=2)
        ax.set_title(f"{name}: % de ataques por bloque de id")
        ax.set_xlabel("bloque de id (1 bloque = 2 % del archivo)")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("% ataque")
    fig.suptitle("2 · El id es el número de fila de un archivo ordenado por clase → se elimina",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "02_id_ordenado.png")


def fig_duplicados(train, test, feats):
    h_tr = pd.util.hash_pandas_object(train[feats], index=False)
    h_te = pd.util.hash_pandas_object(test[feats], index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    rate = (train.assign(dup=h_tr.duplicated(keep=False))
            .groupby(FAMILY)["dup"].mean().mul(100).sort_values())
    cols = [C1 if f == "Normal" else C2 for f in rate.index]
    bars = ax.barh(rate.index, rate.values, color=cols, height=0.6)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK2, padding=3)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% de filas que tienen al menos una copia idéntica (train)")
    ax.set_title("Filas repetidas por familia")

    ax = axes[1]
    dup_tr = h_tr.duplicated().mean() * 100
    seen = h_te.isin(set(h_tr)).mean() * 100
    vals = [dup_tr, h_te.duplicated().mean() * 100, seen]
    names = ["copias dentro\nde train", "copias dentro\nde test", "filas de test\nya vistas en train"]
    bars = ax.bar(names, vals, color="#8a8984", width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, color=INK2, padding=2)
    ax.set_ylabel("% de filas del archivo")
    ax.set_ylim(0, 50)
    ax.set_title("Repeticiones y solapamiento")
    fig.suptitle("3 · 42 % de train son copias y 10 % del test ya está en train → validación agrupada",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "03_duplicados.png")


def fig_ttl(train):
    ct = pd.crosstab(train["sttl"], label_text(train[TARGET]))
    ct = ct[ct.sum(axis=1) > 100][["Normal", "Ataque"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(ct))
    b1 = ax.bar(x, ct["Normal"], color=C1, width=0.6, label="Normal")
    b2 = ax.bar(x, ct["Ataque"], bottom=ct["Normal"], color=C2, width=0.6, label="Ataque",
                edgecolor="white", linewidth=1.5)
    ax.set_xticks(x, [f"sttl = {v}" for v in ct.index])
    ax.set_ylabel("filas (train)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    for xi, (n, a) in enumerate(ct.values):
        ax.text(xi, n + a + 1500, f"{a / (n + a):.0%} ataque", ha="center", fontsize=9, color=INK2)
    ax.legend(frameon=False)
    ax.set_title("4 · sttl (TTL origen→destino) casi separa las clases: artefacto del laboratorio",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "04_sttl_atajo.png")


def fig_distribuciones(train):
    show = [("sttl", "TTL origen→destino"), ("ct_state_ttl", "estado + TTL"),
            ("dload", "bits/s destino→origen"), ("sbytes", "bytes origen→destino"),
            ("smean", "tamaño medio paquete origen"), ("rate", "paquetes/s")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (c, desc) in zip(axes.ravel(), show):
        v = np.log1p(train[c].clip(lower=0))
        bins = np.histogram_bin_edges(v, 40)
        for lab, col, name in [(0, C1, "Normal"), (1, C2, "Ataque")]:
            ax.hist(v[train[TARGET] == lab], bins=bins, color=col, alpha=0.55, density=True, label=name)
        ax.set_title(f"{c} · {desc}", fontsize=10)
        ax.set_xlabel("log(1 + x)")
        ax.set_yticks([])
    axes[0, 0].legend(frameon=False)
    fig.suptitle("5 · Variables más asociadas a la clase (train): muy asimétricas → log1p",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "05_distribuciones.png")


def fig_categoricas(train, test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    s = (train.groupby("service")[TARGET].agg(["mean", "size"])
         .sort_values("mean"))
    bars = ax.barh([f"{i}  (n={n:,})" for i, n in zip(s.index, s["size"])], s["mean"] * 100,
                   color=C2, height=0.6)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK2, padding=3)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% de ataques")
    ax.set_title("service ('-' = no identificado, 54 % de filas)")

    ax = axes[1]
    p = train["proto"].value_counts()
    top = pd.concat([p.head(6), pd.Series({f"otros {len(p) - 6}": p.iloc[6:].sum()})])
    bars = ax.barh(top.index[::-1], top.values[::-1] / len(train) * 100, color=C1, height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color=INK2, padding=3)
    ax.set_xlim(0, 55)
    ax.set_xlabel("% de filas (train)")
    unseen = sorted(set(test["state"]) - set(train["state"]))
    ax.set_title(f"proto: {len(p)} valores, cola larga")
    fig.suptitle(f"6 · Categóricas: cola larga en proto; state {unseen} aparece solo en test",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "06_categoricas.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(RAW / "UNSW_NB15_training-set.csv", encoding="utf-8-sig")
    test = pd.read_csv(RAW / "UNSW_NB15_testing-set.csv", encoding="utf-8-sig")
    feats = [c for c in train.columns if c not in (ID, TARGET, FAMILY)]
    print("Figuras:")
    fig_clases(train, test)
    fig_id(train, test)
    fig_duplicados(train, test, feats)
    fig_ttl(train)
    fig_distribuciones(train)
    fig_categoricas(train, test)


if __name__ == "__main__":
    main()
