import pandas as pd
from q2_train import predict_same_day

# Example: pretend tomorrow’s forecast is 83 °F / 65 °F, no precip, etc.
tomorrow = pd.Series({
    "High Temp": 77,
    "Low Temp":  73.9,
    "Precipitation": 1.08,
    # temporal extras (code expects these):
    "Month": 7,           # july
    "DayOfWeek": 6,       # 0=Mon … 6=Sun ; here: Wednesday
    "IsWeekend": 1
})

print("Predicted bikes:", int(predict_same_day(tomorrow)))