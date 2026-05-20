import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load just a small sample to keep output readable
ref = pd.read_csv("data/reference.csv").drop("target", axis=1, errors="ignore").head(100)

import sqlite3
conn = sqlite3.connect("logs/predictions.db")
curr = pd.read_sql("SELECT * FROM predictions WHERE week = 4", conn)
conn.close()

curr = curr.drop(["prediction", "confidence", "timestamp", "week", "target"], axis=1, errors="ignore").head(100)

# Run report
report   = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=ref, current_data=curr)

# Get dict
try:
    d = snapshot.as_dict()
except AttributeError:
    d = snapshot.dict()

# Print full structure — paste this output and share with Claude
print(json.dumps(d, indent=2, default=str))