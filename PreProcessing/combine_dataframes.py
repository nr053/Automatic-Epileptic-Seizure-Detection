import pandas as pd
from parent.path import Path

path_to_df = Path.repo + '/PreProcessing'

df0 = pd.read_csv(Path.repo + '/TrainingEpochs/train.csv')
df1 = pd.read_csv(Path.repo + '/TrainingEpochs/train1.csv')
df2 = pd.read_csv(Path.repo + '/TrainingEpochs/train2.csv')
df3 = pd.read_csv(Path.repo + '/TrainingEpochs/train3.csv')
df4 = pd.read_csv(Path.repo + '/TrainingEpochs/train4.csv')
df5 = pd.read_csv(Path.repo + '/TrainingEpochs/train5.csv')
df6 = pd.read_csv(Path.repo + '/TrainingEpochs/train6.csv')


df_combined = pd.concat([df0,df1,df2,df3,df4,df5,df6])
df_combined.to_csv('/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/TrainingEpochs/train_patients_with_seizures.csv')