import pandas as pd
import glob
from tqdm import tqdm

df_eval = pd.DataFrame()


for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/eval" + "/**/*.csv", recursive=True)):
    df = pd.read_csv(f, header=5)
    df_eval = pd.concat([df_eval, df])


