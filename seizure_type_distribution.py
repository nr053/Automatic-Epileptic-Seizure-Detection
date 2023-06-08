"""Display the distribution of seizure types across each of the three data splits (train/dev/eval)"""

import pandas as pd
import glob
import os
from path import Path

df_train = pd.Dataframe()

for f in glob.glob(Path.machine_path + "train/" + "**/*.csv"):
    df = pd.read_csv(f, header=6)
    


