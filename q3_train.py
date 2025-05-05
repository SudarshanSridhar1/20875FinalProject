import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

# hyperparams
CSV_PATH = Path("nyc_bicycle_counts_2016.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 120
TRAIN_FRAC = 0.85 # 85/15 train/test split
AUG_FACTOR = 2 # 4x train data with noise to reduce overfitting
NOISE_STD_FRAC = 0.15 # noise for data augmentation
SEED = 42
torch.manual_seed(SEED); 
np.random.seed(SEED)
SAVE_PATH = "q3_cnn_weights.pth"
# model features
FEATURE_COLS = ["Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge", "Total"]

# cleans up to a float
def _to_num(series):
    return (series.astype(str).str.replace(",", "").replace({"T": 0, "": np.nan, " ": np.nan}).astype(float))

# parses dates despite incomplete info
def _safe_parse_dates(series: pd.Series, default_year=2016):
    txt = series.astype(str).str.strip()
    needs_year = ~txt.str.contains(r"\d{4}")
    txt.loc[needs_year] = txt.loc[needs_year] + f" {default_year}"
    dates = pd.to_datetime(txt, format="%m/%d/%Y", errors="coerce")
    mask = dates.isna()
    if mask.any(): # fallback: 01‑Jan‑2016
        dates.loc[mask] = pd.to_datetime(txt.loc[mask], format="%d-%b %Y", errors="coerce")
    return dates

# returns noisy copies of original data
def _augment_weather(df: pd.DataFrame, mul: int):
    if mul <= 1:
        return pd.DataFrame(columns=df.columns)

    num_copies = mul - 1
    traffic_cols = ["Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge", "Total"]

    # compute per‑column std once
    col_std = df[traffic_cols].std()

    aug_frames = []
    for _ in range(num_copies):
        noisy = df.copy(deep=True)
        for col in traffic_cols:
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

    df["target"] = df["DayOfWeek"]
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
        self.y = torch.tensor(frame["target"].values, dtype=torch.long)

    def __len__(self):  
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# CNN
class TrafficCNN(nn.Module):
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
            nn.Linear(64, 7)
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1))

# training epoch with debug output
def run_epoch(model, loader, criterion, opt=None):
    model.train(opt is not None)
    total_correct = 0
    total_samples = 0
    losses = []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x) # shape is (B, 7)
        loss = criterion(out, y)
        if opt:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    acc = 100.0 * total_correct / total_samples
    return np.mean(losses), acc

#bike traffic to day helper
def predict_day(row: pd.Series):
    model = TrafficCNN(in_features=len(FEATURE_COLS))
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    x = torch.tensor(
        scaler.transform(row[FEATURE_COLS].values.reshape(1, -1)),
        dtype=torch.float32,
        device=DEVICE
    )
    with torch.no_grad():
        logits = model(x)
        predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class

#create dataloaders for training
train_df, val_df, test_df = load_data()
scaler = StandardScaler()

train_ds = BikeDataset(train_df, scaler, fit=True)
val_ds = BikeDataset(val_df, scaler, fit=False)
test_ds = BikeDataset(test_df, scaler, fit=False)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

model = TrafficCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
wait = 0
train_accs = []
va_accs = []

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer)
    train_accs.append(tr_acc)
    va_loss, va_acc = run_epoch(model, val_dl, criterion)
    va_accs.append(va_acc)

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        torch.save(model.state_dict(), SAVE_PATH)
        wait = 0
    else:
        wait += 1

    print(f"[{epoch:02d}/{EPOCHS}] val Loss {va_loss:8.3f} | val Acc {va_acc:5.1f}% | train Acc {tr_acc:5.1f}%")

# run on test set, get rmse and acc values
model.load_state_dict(torch.load(SAVE_PATH))
_, te_acc = run_epoch(model, test_dl, criterion)
print(f"\nTEST SET: Accuracy {te_acc:5.1f}%")
# plot training and validation accuracy
plt.figure(figsize=(8, 4))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(va_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy per Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()