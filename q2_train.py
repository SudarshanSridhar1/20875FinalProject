import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path

# hyperparams
CSV_PATH   = Path("nyc_bicycle_counts_2016.csv")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR         = 1e-3
EPOCHS     = 150
TRAIN_FRAC   = 0.85     # 85/15 train/test split
AUG_FACTOR   = 4        # 4x train data with noise to reduce overfitting
NOISE_STD_FRAC = 0.15   # noise for data augmentation
SEED       = 42
torch.manual_seed(SEED); 
np.random.seed(SEED)
SAVE_PATH  = "q2_cnn.pth"

# model features
FEATURE_COLS = ["High Temp", "Low Temp", "Precipitation", "Month", "DayOfWeek", "IsWeekend"]

#create dataloaders
train_df, val_df, test_df = load_data()
scaler = StandardScaler()

train_ds = BikeDataset(train_df, scaler, fit=True)
val_ds   = BikeDataset(val_df,   scaler, fit=False)
test_ds  = BikeDataset(test_df,  scaler, fit=False)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False)
test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False)

model     = WeatherCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_rmse  = float("inf")
wait       = 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_mae, tr_rmse, tr_acc = run_epoch(model, train_dl, criterion, optimizer)
    va_loss, va_mae, va_rmse, va_acc = run_epoch(model, val_dl,   criterion)

    if va_rmse < best_rmse:
        best_rmse = va_rmse
        torch.save(model.state_dict(), SAVE_PATH)
        wait = 0
    else:
        wait += 1

    print(f"[{epoch:02d}/{EPOCHS}] "f"val RMSE {va_rmse:8.1f} | val Acc {va_acc:5.1f}% | "f"train RMSE {tr_rmse:8.1f}")

# run on test set, get rmse and acc values
model.load_state_dict(torch.load(SAVE_PATH))
_, te_mae, te_rmse, te_acc = run_epoch(model, test_dl, criterion)
print(f"\nTEST SET: RMSE {te_rmse:7.1f} | Accuracy {te_acc:5.1f}%")

# cleans up to a float
def _to_num(series):
    return (series.astype(str).str.replace(",", "").replace({"T": 0, "": np.nan, " ": np.nan}).astype(float))

# parses dates despite incomplete info
def _safe_parse_dates(series: pd.Series, default_year=2024):
    txt = series.astype(str).str.strip()
    needs_year = ~txt.str.contains(r"\d{4}")
    txt.loc[needs_year] = txt.loc[needs_year] + f" {default_year}"
    dates = pd.to_datetime(txt, format="%m/%d/%Y", errors="coerce")
    mask = dates.isna()
    if mask.any():           # fallback: 01‑Jan‑2024
        dates.loc[mask] = pd.to_datetime(txt.loc[mask], format="%d-%b %Y", errors="coerce")
    return dates

# returns noisy copies of original data
def _augment_weather(df: pd.DataFrame, mul: int):
    if mul <= 1:
        return pd.DataFrame(columns=df.columns)

    num_copies = mul - 1
    weather_cols = ["High Temp", "Low Temp", "Precipitation"]

    # compute per‑column std once
    col_std = df[weather_cols].std()

    aug_frames = []
    for _ in range(num_copies):
        noisy = df.copy(deep=True)
        for col in weather_cols:
            noise = np.random.normal(
                loc=0.0,
                scale=NOISE_STD_FRAC * col_std[col],
                size=len(df)
            )
            noisy[col] += noise

        # keep temps realistic
        noisy["High Temp"] = noisy["High Temp"].clip(lower=-10, upper=120)
        noisy["Low Temp"]  = noisy["Low Temp"].clip(lower=-20, upper=100)
        # precipitation can’t be negative
        noisy["Precipitation"] = noisy["Precipitation"].clip(lower=0)

        aug_frames.append(noisy)

    return pd.concat(aug_frames, ignore_index=True)

#clean and load data into dataframe
def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    bridges = ["Brooklyn Bridge", "Manhattan Bridge","Queensboro Bridge", "Williamsburg Bridge"]
    for b in bridges:
        df[b] = _to_num(df[b])

    if "Total" not in df.columns:
        df["Total"] = df[bridges].sum(axis=1)
    else:
        df["Total"] = _to_num(df["Total"])

    for c in ["High Temp", "Low Temp", "Precipitation"]:
        df[c] = _to_num(df[c])

    df["Date"] = _safe_parse_dates(df["Date"])
    df = df.dropna(subset=["Date"])

    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    df["target"] = df["Total"]
    df = df.dropna().reset_index(drop=True)

    # split randomly according to hyperparams
    train_df, test_df = train_test_split(
        df, test_size=1-TRAIN_FRAC, random_state=SEED, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # data augmentation performed on training data
    if AUG_FACTOR > 1:
        aug_df = _augment_weather(train_df, AUG_FACTOR)
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 15 % of training data used for validation
    val_split = 0.15
    n_train   = int(len(train_df) * (1 - val_split))
    val_df    = train_df.iloc[n_train:].reset_index(drop=True)
    train_df  = train_df.iloc[:n_train].reset_index(drop=True)

    return train_df, val_df, test_df

# dataset for CNN
class BikeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, scaler: StandardScaler, fit=False):
        if fit:
            scaler.fit(frame[FEATURE_COLS])
        self.x = torch.tensor(
            scaler.transform(frame[FEATURE_COLS]),
            dtype=torch.float32
        )
        self.y = torch.tensor(frame["target"].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):  
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# CNN
class WeatherCNN(nn.Module):
    def __init__(self, in_features=len(FEATURE_COLS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1))

# training epoch with debug output
def run_epoch(model, loader, criterion, opt=None):
    model.train(opt is not None)
    losses, preds, gts = [], [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        if opt:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        preds.append(out.detach().cpu())
        gts.append(y.detach().cpu())

    preds = torch.cat(preds); gts = torch.cat(gts)

    # calculate metrics
    mae  = mean_absolute_error(gts, preds)
    rmse = np.sqrt(mean_squared_error(gts, preds))

    # convert to numpy for accuracy calculations
    gts_np   = gts.numpy().flatten()
    preds_np = preds.numpy().flatten()

    # avoid divide by zero errors
    nonzero  = gts_np != 0 
    mape = np.mean(np.abs((gts_np[nonzero] - preds_np[nonzero]) / gts_np[nonzero])) * 100.0
    acc  = 100.0 - mape
    return np.mean(losses), mae, rmse, acc

#forecast to bike total helper
def predict_same_day(weather_row: pd.Series):
    model = WeatherCNN()
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    x = torch.tensor(
        scaler.transform(weather_row[FEATURE_COLS].values.reshape(1, -1)),
        dtype=torch.float32, device=DEVICE
    )
    with torch.no_grad():
        return model(x).cpu().item()