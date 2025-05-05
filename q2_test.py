import pandas as pd
from q2_train import predict_same_day

tomorrow = pd.Series({
    "High Temp": 82.9,
    "Low Temp":  66.9,
    "Precipitation": 0.4,
    # temporal extras (assumed access because forecast requires day info)
    "Month": 6,           # 0=Jan, 11=Dec
    "DayOfWeek": 5,       # 0=Mon, 6=Sun
    "IsWeekend": 1
})

print("Predicted bikes:", int(predict_same_day(tomorrow)))