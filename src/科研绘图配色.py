"""Shared plotting style for mathematical modeling figures."""

from matplotlib.colors import LinearSegmentedColormap


科研配色 = {
    "红": "#C02635",
    "珊瑚": "#F58966",
    "深蓝": "#00609D",
    "浅蓝": "#7EABCE",
    "橙": "#F5A86E",
    "浅黄": "#F4D291",
}

科研调色板 = [
    科研配色["红"],
    科研配色["橙"],
    科研配色["深蓝"],
    科研配色["浅蓝"],
    科研配色["珊瑚"],
    科研配色["浅黄"],
]

科研连续色带 = LinearSegmentedColormap.from_list(
    "research_continuous",
    [
        科研配色["深蓝"],
        科研配色["浅蓝"],
        "#FFFFFF",
        科研配色["浅黄"],
        科研配色["珊瑚"],
        科研配色["红"],
    ],
)

科研顺序色带 = LinearSegmentedColormap.from_list(
    "research_sequential",
    [
        "#FFFFFF",
        科研配色["浅黄"],
        科研配色["橙"],
        科研配色["浅蓝"],
        科研配色["深蓝"],
    ],
)


def 设置科研绘图风格(plt, sns, grid=True):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=科研调色板)
    sns.set_theme(
        style="whitegrid" if grid else "white",
        font="SimHei",
        palette=科研调色板,
        rc={
            "axes.edgecolor": "#D9D9D9",
            "grid.color": "#EAEAEA",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )
