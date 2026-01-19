import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import wilcoxon
import time
import os
import random
import matplotlib.pyplot as plt

# ==========================================
# 0. REPRODUCIBILITY SETUP
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Global Seed Set to {seed} ---")

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def get_datasets(freq, time_step=60):
    print(f"--- Loading UCI dataset with frequency: {freq} and time_step: {time_step} ---")

    file_path = 'household_power_consumption.txt'

    if not os.path.exists(file_path):
        print(f"!!! File not found at {file_path}. Please update the path. !!!")
        return None, None, None, None

    df = pd.read_csv(file_path, sep=';', low_memory=False)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.set_index('DateTime')
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    power_column = 'Global_active_power'

    df = df[[power_column]].dropna()
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Resample
    df_resampled = df[power_column].resample(freq).mean().to_frame()
    df_resampled.ffill(inplace=True)
    df_resampled.bfill(inplace=True)

    data = df_resampled.values
    
    # Split: 60% Train, 20% Val, 20% Test
    train_size = int(len(data) * 0.6)
    val_size = int(len(data) * 0.2)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    if len(train_data) < time_step + 1:
        print("!!! Not enough data. !!!")
        return None, None, None, None

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    scaled_train = scaler.transform(train_data)
    scaled_val = scaler.transform(val_data)
    scaled_test = scaler.transform(test_data)

    def create_sequences(data, time_step):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_sequences(scaled_train, time_step)
    X_val, y_val = create_sequences(scaled_val, time_step)
    X_test, y_test = create_sequences(scaled_test, time_step)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("!!! Not enough data to create sequences. !!!")
        return None, None, None, None

    X_train = X_train.reshape(-1, time_step, 1)
    X_val = X_val.reshape(-1, time_step, 1)
    X_test = X_test.reshape(-1, time_step, 1)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    return train_ds, val_ds, test_ds, scaler

# ==========================================
# 2. METRICS & STATISTICS
# ==========================================

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "MAPE": mape, "R2": r2}

def perform_wilcoxon_test(y_true, y_pred_model, y_pred_baseline):
    errors_model = np.abs(y_true - y_pred_model).flatten()
    errors_baseline = np.abs(y_true - y_pred_baseline).flatten()
    try:
        stat, p_value = wilcoxon(errors_model, errors_baseline)
        significant = p_value < 0.05
    except Exception as e:
        p_value = 1.0
        significant = False
    return {"p-value": p_value, "Significant": significant}

def plot_actual_vs_predicted(y_true, y_pred, freq_name, save_dir, model_id):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(12, 6))
    limit = min(200, len(y_true))
    plt.plot(y_true[:limit], label="Actual", color='black', alpha=0.7)
    plt.plot(y_pred[:limit], label="Predicted", color='red', linestyle='--')
    plt.title(f"{model_id} - {freq_name} (First {limit} steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"{model_id}_{freq_name}_comparison.png"))
    plt.close()

# ==========================================
# 3. DLINEAR MODEL ARCHITECTURE
# ==========================================

class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len=1, enc_in=1, kernel_size=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # --- Auto-adjust Kernel ---
        kernel_size = min(kernel_size, seq_len)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(3, kernel_size)
        
        self.decompsition = SeriesDecomp(kernel_size=kernel_size)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        self.channels = enc_in

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1).squeeze(-1)

# ==========================================
# 4. TRAINING LOGIC (Standardized)
# ==========================================

def train_dlinear(model, train_loader, val_loader, epochs, lr, device, early_stop_patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_rmse = np.inf
    best_state = None
    patience_counter = 0
    train_hist, val_hist = [], []
    
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb).squeeze(-1)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_hist.append(np.mean(batch_losses))

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb).squeeze(-1)
                preds.append(out.cpu().numpy())
                trues.append(yb.cpu().numpy())

        if len(preds) > 0:
            val_rmse = np.sqrt(mean_squared_error(np.concatenate(trues), np.concatenate(preds)))
            val_hist.append(val_rmse**2) 

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
    
    train_time = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, train_hist, val_hist, train_time, best_rmse

# ==========================================
# 5. MAIN RUNNER (Fair Comparison)
# ==========================================

DLINEAR_GRID = [
    {"lr": 1e-3, "batch_size": 64},
    {"lr": 5e-4, "batch_size": 64},
]

def run_dlinear_fair_comparison(train_ds, val_ds, test_ds, scaler, device, freq_name, save_dir, seq_len):
    
    KERNEL_SIZE = 25 

    # --- PHASE 1: HYPERPARAMETER SEARCH (Train on Train, Val on Val) ---
    print(f"\n--- Phase 1: Grid Search for {freq_name} (SeqLen: {seq_len}) ---")
    
    best_val_rmse = np.inf
    best_cfg = {}

    for i, cfg in enumerate(DLINEAR_GRID):
        # Adjust batch size for Weekly to prevent poor convergence
        bs = 16 if freq_name == 'Weekly' else cfg["batch_size"]
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)
        
        model = DLinear(
            seq_len=seq_len, 
            pred_len=1, 
            enc_in=1, 
            kernel_size=KERNEL_SIZE
        )
        
        # Train (Search phase: 50 epochs)
        model, _, _, _, val_rmse = train_dlinear(
            model, train_loader, val_loader, 
            epochs=50, lr=cfg["lr"], device=device
        )
        
        print(f"[{i+1}/{len(DLINEAR_GRID)}] Config: {cfg} (Adj Batch: {bs}) -> Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_cfg = cfg.copy()
            best_cfg['batch_size'] = bs # Save the actual used batch size

    print(f"Best Config Found: {best_cfg}")

    # --- PHASE 2: RETRAINING (Train on Train+Val) ---
    print(f"\n--- Phase 2: Retraining Best Model on Combined Data ---")
    
    full_train_ds = ConcatDataset([train_ds, val_ds])
    full_train_loader = DataLoader(full_train_ds, batch_size=best_cfg["batch_size"], shuffle=True)
    
    # Dummy val loader
    val_loader_dummy = DataLoader(val_ds, batch_size=best_cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=best_cfg["batch_size"])

    final_model = DLinear(
        seq_len=seq_len, 
        pred_len=1, 
        enc_in=1, 
        kernel_size=KERNEL_SIZE
    )

    # Train (Retrain phase: 100 epochs)
    final_model, train_hist, val_hist, train_time, _ = train_dlinear(
        final_model, full_train_loader, val_loader_dummy, 
        epochs=100, lr=best_cfg["lr"], device=device, early_stop_patience=15
    )

    # --- PHASE 3: FINAL TEST EVALUATION ---
    print(f"\n--- Phase 3: Final Test Evaluation ---")
    
    final_model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = final_model(xb.to(device)).squeeze(-1)
            preds_list.append(out.cpu().numpy())
            trues_list.append(yb.cpu().numpy())

    preds_flat = np.concatenate(preds_list).reshape(-1, 1)
    trues_flat = np.concatenate(trues_list).reshape(-1, 1)
    
    # Inverse Scale
    preds_scaled = scaler.inverse_transform(preds_flat).flatten()
    trues_scaled = scaler.inverse_transform(trues_flat).flatten()
    
    # Metrics
    metrics = calculate_metrics(trues_scaled, preds_scaled)
    
    # Naive Baseline for Comparison
    naive_preds = np.roll(trues_scaled, 1)
    naive_preds[0] = trues_scaled[0] 
    naive_metrics = calculate_metrics(trues_scaled, naive_preds)
    
    wilcoxon_res = perform_wilcoxon_test(trues_scaled, preds_scaled, naive_preds)
    
    params = sum(p.numel() for p in final_model.parameters())

    # Plotting
    plot_actual_vs_predicted(trues_scaled, preds_scaled, freq_name, save_dir, "DLinear")
    plt.figure()
    plt.plot(train_hist, label="Train Loss (Combined)")
    plt.title(f"DLinear Learning Curve - {freq_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"learning_curve_dlinear_{freq_name}.png"))
    plt.close()

    # --- FINAL PRINTOUT (FORMATTED) ---
    print(f"\n{'='*20} RESULTS FOR: UCI_{freq_name} {'='*20}")
    print(f"Best Hyperparameters: {best_cfg}")
    print(f"Parameter Count: {params}")
    print(f"Final Retraining Time (s): {train_time:.2f}")

    print(f"\nMetrics on Test Set:")
    print(f"{'Metric':<10} | {'DLinear':<15} | {'Naive':<15}")
    print(f"{'-'*10}-|-{'-'*15}-|-{'-'*15}")

    for k in ["RMSE", "MSE", "MAE", "MAPE", "R2"]:
        print(f"{k:<10} | {metrics[k]:<15.4f} | {naive_metrics[k]:<15.4f}")

    sig_msg = "(Significant)" if wilcoxon_res['Significant'] else "(Not Significant)"
    print(f"\nWilcoxon Test p-value: {wilcoxon_res['p-value']:.6f} {sig_msg}")
    print(f"Plot saved to {save_dir}/DLinear_{freq_name}_comparison.png")
    print(f"{'='*20} FINISHED {'='*20}\n")

# ==========================================
# 6. EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    
    set_seed(42)
    save_dir = "./dlinear_results"
    
    FREQUENCIES_TO_RUN = {
        'Minutely': 'T', 'Hourly': 'H', 'Daily': 'D', 'Weekly': 'W'
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for freq_name, freq_code in FREQUENCIES_TO_RUN.items():
        exp_name = f"UCI_{freq_name}"
        print(f"\n{'='*20} STARTING EXPERIMENT: {exp_name} {'='*20}")

        if freq_code == 'W':
            time_step = 8
        elif freq_code == 'D':
            time_step = 30
        else:
            time_step = 60

        train_ds, val_ds, test_ds, scaler = get_datasets(freq_code, time_step=time_step)
        
        if train_ds is None: 
            print(f"Skipping {freq_name} due to data loading issues.")
            continue

        run_dlinear_fair_comparison(
            train_ds, val_ds, test_ds,
            scaler,
            device,
            freq_name,
            save_dir,
            seq_len=time_step
        )
