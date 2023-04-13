"""Find the

1. number of patients
2. number of recordings
3. number of seizures
4. ictal duration

for each sampling frequency"""



import glob
from path import Path
import mne 
import pandas as pd

patients_250 = set()
patients_256 = set()
patients_400 = set()
patients_512 = set()
patients_1000 = set()

sessions_250 = set()
sessions_256 = set()
sessions_400 = set()
sessions_512 = set()
sessions_1000 = set()


recordings_250 = 0
recordings_256 = 0
recordings_400 = 0
recordings_512 = 0
recordings_1000 = 0

seizures_250 = 0
seizures_256 = 0
seizures_400 = 0
seizures_512 = 0
seizures_1000 = 0

ict_dur_250 = 0
ict_dur_256 = 0
ict_dur_400 = 0
ict_dur_512 = 0
ict_dur_1000 = 0

duration_250 = 0
duration_256 = 0
duration_400 = 0
duration_512 = 0
duration_1000 = 0


for f in glob.glob(Path.machine_path + '**/*.edf', recursive=True):
    seizure_duration = 0
    total_duration = 0
    print(f)
    data = mne.io.read_raw_edf(f, infer_types=True)
    
    patient_name = f.split("/")[9]
    session = f.split("/")[9] + '/' + f.split("/")[10]
    csv_file = f.removesuffix('edf') + 'csv_bi'    
    df = pd.read_csv(csv_file, header=5)
    num_seizures = df['label'].str.count('seiz').sum()

    df_duration = pd.read_csv(csv_file, nrows=1, skiprows=2, names=['a'])  #read the file duration row into another dataframe
    df_duration['a'] = df_duration['a'].astype("string")    #convert file duration field to a string
    str = df_duration.loc[0].at['a']    #extract file duration string
    str_list = str.split() #split the string into a str list
    total_file_duration = float(str_list[3])    #extract file duration as a float
    total_duration += total_file_duration

    if df['label'].eq("seiz").any():
            df['duration'] = df['stop_time'] - df['start_time']
            seizure_duration = df['duration'].sum()

    if data.info['sfreq'] == 250:
          recordings_250 += 1
          patients_250.add(patient_name)
          sessions_250.add(session)
          seizures_250 += num_seizures
          ict_dur_250 += seizure_duration
          duration_250 += total_duration

    if data.info['sfreq'] == 256:
          recordings_256 += 1
          patients_256.add(patient_name)
          sessions_256.add(session)
          seizures_256 += num_seizures
          ict_dur_256 += seizure_duration
          duration_256 += total_duration

    if data.info['sfreq'] == 400:
          recordings_400 += 1
          patients_400.add(patient_name)
          sessions_400.add(session)
          seizures_400 += num_seizures
          ict_dur_400 += seizure_duration
          duration_400 += total_duration

    if data.info['sfreq'] == 512:
          recordings_512 += 1
          patients_512.add(patient_name)
          sessions_512.add(session)
          seizures_512 += num_seizures
          ict_dur_512 += seizure_duration
          duration_512 += total_duration
    
    if data.info['sfreq'] == 1000:
          recordings_1000 += 1
          patients_1000.add(patient_name)
          sessions_1000.add(session)
          seizures_1000 += num_seizures
          ict_dur_1000 += seizure_duration
          duration_1000 += total_duration

#print(patients_250)
#print(patients_256)
#print(patients_400)
#print(patients_512)
#print(patients_1000)

print("Number of patients at 250 Hz: ", len(patients_250))
print("Number of sessions at 250 Hz: ", len(sessions_250))
print("Number of recordings at 250 Hz: ", recordings_250)
print("Number of seizures recorded at 250Hz: ", seizures_250)
print("Duration of seizures recorded at 250Hz: ", ict_dur_250)
print("Duration of recordings at 250 Hz: ", duration_250)

print("\nNumber of patients at 256 Hz: ", len(patients_256))
print("Number of sessions at 256 Hz: ", len(sessions_256))
print("Number of recordings at 256 Hz: ", recordings_256)
print("Number of seizures recorded at 256Hz: ", seizures_256)
print("Duration of seizures recorded at 256Hz: ", ict_dur_256)
print("Duration of recordings at 256 Hz: ", duration_256 )

print("\nNumber of patients at 400 Hz: ", len(patients_400))
print("Number of sessions at 400 Hz: ", len(sessions_400))
print("Number of recordings at 400 Hz: ", recordings_400)
print("Number of seizures recorded at 400 Hz: ", seizures_400)
print("Duration of seizures recorded at 400 Hz: ", ict_dur_400)
print("Duration of recordings at 400 Hz: ", duration_400)

print("\nNumber of patients at 512 Hz: ", len(patients_512))
print("Number of sessions at 512 Hz: ", len(sessions_512))
print("Number of recordings at 512 Hz: ", recordings_512)
print("Number of seizures recorded at 512 Hz: ", seizures_512)
print("Duration of seizures recorded at 512 Hz: ", ict_dur_512)
print("Duration of recordings at 512 Hz: ", duration_512)

print("\nNumber of patients at 1000 Hz: ", len(patients_1000))
print("Number of sessions at 1000 Hz: ", len(sessions_1000))
print("Number of recordings at 1000 Hz: ", recordings_1000)
print("Number of seizures recorded at 1000 Hz: ", seizures_1000)
print("Duration of seizures recorded at 1000 Hz: ", ict_dur_1000)
print("Duration of recordings at 1000 Hz: ", duration_1000)

print("\nTotal number of patients: ", len(patients_250) + len(patients_256) + len(patients_400) + len(patients_512) + len(patients_1000))
print("Total number of sessions: ", len(sessions_250) + len(sessions_256) + len(sessions_400) + len(sessions_512) + len(sessions_1000))
print("Total number of recordings: ", recordings_250 + recordings_256 + recordings_400 + recordings_512 + recordings_1000)
print("Total number of seizures: ", seizures_250 + seizures_256 + seizures_400 + seizures_512 + seizures_1000)
print("Total duration of seizures: ", ict_dur_250 + ict_dur_256 + ict_dur_400 + ict_dur_512 + ict_dur_1000)
print("Total duration of recordings: ", duration_250 + duration_256 + duration_400 + duration_512 + duration_1000)

#intersection = set.intersection(patients_250, patients_256, patients_400, patients_512, patients_1000)
#print("patients sampled at multiple frequencies: ", intersection)

#intersec_250_256 = set.intersection(patients_250,patients_256)
#intersec_250_400 = set.intersection(patients_250,patients_400)
#intersec_250_512 = set.intersection(patients_250,patients_512)
#intersec_250_1000 = set.intersection(patients_250,patients_1000)

#print("\nIntersections with 250 Hz: ", intersec_250_256, intersec_250_400, intersec_250_512, intersec_250_1000)