import pandas as pd
from q3_train import predict_day

# bridge bike counts for a sample day, in this case a sunday (17-Apr)
sample_day = pd.Series({
    "Brooklyn Bridge":     4126,
    "Manhattan Bridge":    7565,
    "Queensboro Bridge":   9028,
    "Williamsburg Bridge": 5823,
    "Total":               26542
})

predicted_dow = predict_day(sample_day)
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

print("Predicted day of week:", day_names[predicted_dow])