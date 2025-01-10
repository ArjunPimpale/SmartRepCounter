import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display


df = pd.read_pickle(r"../../data/interim/resampled_data.pkl")

# plotting the data


# cnt = 1
# for label in df['label'].unique():
#     subset = df[df['label'] == label]
#     ax = plt.subplot(6, 1, cnt)
#     ax.plot(subset[:100]['gyro_x'].reset_index(drop=True), label=label)
#     ax.set_title(f"Label: {label}")  # Add a title with the label variable
#     cnt += 1

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["gyro_x"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


mpl.rcParams["figure.figsize"] = (20, 5)
mpl.style.use("ggplot")
mpl.rcParams["figure.dpi"] = 100


category_df = df.query("label == 'ohp'").query("candidate == 'A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_z"].plot(legend=True)

label = "squat"
category = "sitting"

participant_df = df.query(f"label == '{label}'").sort_values("candidate").reset_index()


fig, ax = plt.subplots()
participant_df.groupby(["candidate"])[["acc_x", "acc_y", "acc_z"]].plot()
ax.set_xlabel("index")
ax.set_ylabel("acc_y")
plt.legend()


labels = df["label"].unique()
participants = df["candidate"].unique()

#for accelerometer data for each participant and label
for label in labels:
    for candidate in participants:
        participant_df = (
            df.query(f"label == '{label}'")
            .query(f"candidate == '{candidate}'")
            .reset_index()
        )
        if len(participant_df) > 0:
            fig, ax = plt.subplots()
            participant_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            plt.title(f"{label},{candidate}")
            plt.legend()
            
            
#for gyrocope data for each participant and label

for label in labels:
    for candidate in participants:
        participant_df = (
            df.query(f"label == '{label}'")
            .query(f"candidate == '{candidate}'")
            .reset_index()
        )
        if len(participant_df) > 0:
            fig, ax = plt.subplots()
            participant_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax)
            plt.title(f"{label},{candidate}")
            plt.legend()



#saving the plots

for label in labels:
    for candidate in participants:
        participant_df = (
            df.query(f"label == '{label}'")
            .query(f"candidate == '{candidate}'")
            .reset_index()
        )

        if len(participant_df) > 0:
            fig, ax = plt.subplots(nrows = 2,figsize = (20,10),sharex=True)
            participant_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])
            participant_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            ax[0].legend(loc = 'upper center',shadow = True,ncol = 3,bbox_to_anchor = (0.5,1.2))
            ax[1].legend(loc = 'upper center',shadow = True,ncol = 3,bbox_to_anchor = (0.5,1.2))
            plt.title(f"{label},{candidate}")
            ax[1].set_xlabel("index")
          
            plt.savefig(f"../../reports/figures/{label}_{candidate}.png")