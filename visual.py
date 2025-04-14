from prov4ml.utils.prov_getters import *
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from configs.paths import CACHE_PATH


file_path = f"{CACHE_PATH}/prov/IBM_outliers_13/provgraph_IBM_outliers.json"
start_at = 0


data = json.load(open(file_path))
print(get_metrics(data))
indices = get_metric(data, metric="Indices_Context.TRAINING", sort_by="time")["value"]
df1 = get_metric(data, "Loss_Context.TRAINING", start_at=start_at, sort_by="time")
is_outlier = get_metric(data, "Outlier_Context.TRAINING", start_at=start_at, sort_by="time")["value"]
is_outlier = [any(eval(o)) for o in is_outlier]
print(pd.Series(is_outlier).value_counts())

# epochs = df1["epoch"]
# df1.to_csv("loss_noshuffle.csv")


sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
color = 'tab:blue'
sns.lineplot(data=df1, x="time", y="value", color=color)
plt.title("Loss at each step of the training (red lines are outliers)")
plt.ylabel("Loss (MSE)")
plt.xlabel("Batch")
# plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# for position in df1[is_outlier]["time"]:
#     plt.axvline(x=position, color='red', linestyle='--')  # Customize the line color and style

plt.tight_layout()
plt.savefig(f"loss_noshuffle.png")
plt.close()


# df1 = get_metric(data, "Backward_time_Context.TRAINING", start_at=start_at, sort_by="time")
# # df1 = df1[df1.index % 2 == 1]
# sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
# sns.lineplot(data=df1, x="time", y="value", color=color, ci=None)
# plt.title("Backward_time")
# # plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# print(len(df1), len(is_outlier))
# for position in df1[is_outlier]["time"]:
#     plt.axvline(x=position, color='red', linestyle='--')  # Customize the line color and style

# plt.tight_layout()
# plt.savefig(f"Backward_time_Context.TRAINING.png")
# plt.close()

# df1 = get_metric(data, "Step_time_Context.TRAINING", start_at=start_at, sort_by="time")
# df1["epoch"] = epochs
# sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
# sns.lineplot(data=df1, x="time", y="value", color=color)
# for position in df1[is_outlier]["time"]:
#     plt.axvline(x=position, color='red', linestyle='--')  # Customize the line color and style

# # plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# plt.title("Step_time_Context")
# plt.tight_layout()
# plt.savefig(f"Step_time_Context.TRAINING.png")
# plt.close()


# df1 = get_metric(data, "gpu_power_usage_Context.TRAINING", start_at=start_at, sort_by="time")
# df1["epoch"] = epochs
# sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
# sns.lineplot(data=df1, x="time", y="value", color=color)
# plt.title("gpu_power_usage_Context.TRAINING")
# # plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# for position in df1[is_outlier]["time"]:
#     plt.axvline(x=position, color='red', linestyle='--')  # Customize the line color and style


# plt.tight_layout()
# plt.savefig(f"gpu_power_usage_Context.TRAINING.png")
# plt.close()

# df1 = get_metric(data, "gpu_usage_Context.TRAINING", start_at=start_at, sort_by="time")
# df1["epoch"] = epochs
# sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
# sns.lineplot(data=df1, x="time", y="value", color=color)
# # plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# plt.title("gpu_usage_Context.TRAINING")
# plt.tight_layout()
# plt.savefig(f"gpu_usage_Context.TRAINING.png")
# plt.close()

# df1 = get_metric(data, "Gpu_power_nvidia_Context.TRAINING", start_at=start_at, sort_by="time")
# df1["epoch"] = epochs
# sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
# sns.lineplot(data=df1, x="time", y="value", color=color)
# # plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# plt.title("Gpu_power_nvidia_Context.TRAINING")
# for position in df1[is_outlier]["time"]:
#     plt.axvline(x=position, color='red', linestyle='--')  # Customize the line color and style


# plt.tight_layout()
# plt.savefig(f"Gpu_power_nvidia_Context.TRAINING.png")
# plt.close()


# df1 = get_metric(data, "Gpu_energy_nvidia_Context.TRAINING", start_at=start_at, sort_by="time")
# df1["epoch"] = epochs
# sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
# sns.lineplot(data=df1, x="time", y="value", color=color)
# # plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# plt.title("Gpu_energy_nvidia_Context.TRAINING")
# for position in df1[is_outlier]["time"]:
#     plt.axvline(x=position, color='red', linestyle='--')  # Customize the line color and style


# plt.tight_layout()
# plt.savefig(f"Gpu_energy_nvidia_Context.TRAINING.png")
# plt.close()


# df1 = get_metric(data, "Gpu_usage_nvidia_Context.TRAINING", start_at=start_at, sort_by="time")
# df1["epoch"] = epochs
# df1["value"] = df1["value"].diff().fillna(0.0)
# sns.set_theme(rc={'figure.figsize': (19.7, 8.27)})
# sns.lineplot(data=df1, x="time", y="value", color=color)
# # plt.xticks(df1["time"], ["No" if not o else "Yes" for o in is_outlier], rotation=90)
# plt.title("Gpu_usage_nvidia_Context.TRAINING")
# for position in df1[is_outlier]["time"]:
#     plt.axvline(x=position, color='red', linestyle='--')  # Customize the line color and style


# plt.tight_layout()
# plt.savefig(f"Gpu_usage_nvidia_Context.TRAINING.png")
# plt.close()
