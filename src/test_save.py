import pandas as pd
import numpy as np
import time

sequences = ["ACGT" * 9 for _ in range(1000)]
dbps = [f"DBP_{i}" for i in range(100)]

print("Creating dataframe...")
df = pd.DataFrame({
    "DNA probe": np.repeat(sequences, len(dbps)),
    "Protein": np.tile(dbps, len(sequences)),
    "Affinity": np.random.rand(len(sequences) * len(dbps))
})
print("Dataframe shape:", df.shape)

t0 = time.time()
unique_seqs = df["DNA probe"].unique()
encoded_str = {seq: str(np.random.rand(36, 4).tolist()) for seq in unique_seqs}
df["DNA_onehot"] = df["DNA probe"].map(encoded_str)
print("Map to string time:", time.time() - t0)

t0 = time.time()
df.to_csv("test_output.csv", index=False)
print("to_csv time:", time.time() - t0)

import os
print("File size (MB):", os.path.getsize("test_output.csv") / (1024*1024))
