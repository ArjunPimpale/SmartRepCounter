import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display



df = pd.read_pickle(r'../../data/interim/resampled_data.pkl')

#plotting the data


# cnt = 1
# for label in df['label'].unique():
#     subset = df[df['label'] == label]
#     ax = plt.subplot(6, 1, cnt)
#     ax.plot(subset[:100]['gyro_x'].reset_index(drop=True), label=label)
#     ax.set_title(f"Label: {label}")  # Add a title with the label variable
#     cnt += 1

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

for label in df['label'].unique():
    subset = df[df['label'] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['gyro_x'].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
    
    
    

    
    
mpl.rcParams['figure.figsize'] = (15, 5)
mpl.style.use('ggplot')
mpl.rcParams['figure.dpi'] = 100