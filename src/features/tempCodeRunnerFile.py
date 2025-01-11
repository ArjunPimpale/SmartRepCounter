import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


df = pd.read_pickle(r"../../data/interim/resampled_data.pkl")