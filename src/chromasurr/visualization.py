import json, matplotlib.pyplot as plt

def plot_sobol(path="sobol.json", order="ST"):
    data = json.load(open(path))
    keys, vals = zip(*data[order].items())
    plt.bar(keys, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"Sobol {order}")
    plt.tight_layout()
    plt.show()
