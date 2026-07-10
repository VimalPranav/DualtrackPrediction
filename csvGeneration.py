import os
import random
import pandas as pd

base_dir = "/home/user/Desktop/ULTRASOUND/Dataset"

subjects = sorted([
    folder
    for folder in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, folder))
])

print(f"Found {len(subjects)} subjects")


random.seed(42)
random.shuffle(subjects)

train_subjects = set(subjects[:40])
val_subjects = set(subjects[40:])

print(f"Training subjects   : {len(train_subjects)}")
print(f"Validation subjects : {len(val_subjects)}")


rows = []

for subject in sorted(subjects):

    subject_dir = os.path.join(base_dir, subject)

    split = "train" if subject in train_subjects else "val"

    for file in sorted(os.listdir(subject_dir)):

        if not file.endswith(".h5"):
            continue

        sweep_path = os.path.join(subject_dir, file).replace("\\", "/")

        sweep_id = f"{subject}_{os.path.splitext(file)[0]}"

        rows.append({
            "sweep_id": sweep_id,
            "raw_tus_rec_sweep_path": sweep_path,
            "split": split
        })

# Create DataFrame

df = pd.DataFrame(rows)

# Add index column explicitly
df.insert(0, "index", range(len(df)))

# Save without pandas index
df.to_csv("tusrec_input.csv", index=False)

print(f"\nCreated CSV with {len(df)} scans.")

print(df.head())

print("\nTrain scans :", (df["split"] == "train").sum())
print("Val scans   :", (df["split"] == "val").sum())