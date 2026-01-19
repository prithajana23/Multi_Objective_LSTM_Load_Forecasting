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
import math
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

    # --- IMPORTANT: Update this path to your UCI dataset file ---
    file_path = 'household_power_consumption.txt'
    # ----------------------------------------------------------------

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

    # Resample to the requested frequency
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
    try:
        # Wilcoxon requires 1D arrays
        stat, p_value = wilcoxon(np.abs(y_true - y_pred_model).flatten(), np.abs(y_true - y_pred_baseline).flatten())
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
# 3. TRANSFORMER MODEL
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        x = self.embedding(x) * self.scale
        x = self.pos_encoder(x)
        x = self.encoder(x)
        # Predict using the last time step
        return self.fc(x[:, -1, :]).squeeze(-1)

# ==========================================
# 4. TRAINING LOGIC (FIXED)
# ==========================================
def train_model(model, train_loader, val_loader, epochs, lr, device, early_stop_patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_rmse = np.inf
    best_state = None
    patience_counter = 0
    train_hist, val_hist = [], []

    # 1. START TIMER
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb) 
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
                out = model(xb)
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
    
    # 2. STOP TIMER
    train_time = time.time() - start_time
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # 3. RETURN 5 ITEMS (Consistent with logic)
    return model, train_hist, val_hist, train_time, best_rmse

# ==========================================
# 5. MAIN RUNNER (FIXED UNPACKING)
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = "./transformer_results"
    os.makedirs(save_dir, exist_ok=True)

    FREQUENCIES_TO_RUN = {
        'Minutely': 'T', 'Hourly': 'H', 'Daily': 'D', 'Weekly': 'W'
    }

    # Grid for Hyperparameter Search
    TRANSFORMER_GRID = [
        {"d_model": 32, "layers": 1, "nhead": 2, "lr": 5e-4, "batch": 32, "dropout": 0.05},
        {"d_model": 64, "layers": 2, "nhead": 4, "lr": 1e-4, "batch": 64, "dropout": 0.10},
    ]

    for freq_name, freq_code in FREQUENCIES_TO_RUN.items():
        exp_name = f"UCI_{freq_name}"
        print(f"\n{'='*20} STARTING EXPERIMENT: {exp_name} {'='*20}")

        # Set time step based on frequency
        if freq_code == 'W':
            time_step = 8
        elif freq_code == 'D':
            time_step = 30
        else:
            time_step = 60

        # Load Dataset
        train_ds, val_ds, test_ds, scaler = get_datasets(freq_code, time_step=time_step)
        if train_ds is None: continue

        # --- PHASE 1: GRID SEARCH (Train on 60%, Validate on 20%) ---
        print("\n--- Phase 1: Grid Search (50 Epochs) ---")
        best_val_rmse = np.inf
        best_cfg = {}

        for cfg in TRANSFORMER_GRID:
            train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=cfg["batch"])

            model = TransformerModel(
                input_size=1,
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["layers"],
                dropout=cfg["dropout"]
            )

            # Train for 50 Epochs
            # FIXED: Now unpacking 5 values
            model, _, val_hist, _, _ = train_model(model, train_loader, val_loader, epochs=50, lr=cfg["lr"], device=device)
            
            if len(val_hist) > 0:
                current_val_rmse = np.sqrt(min(val_hist))
                print(f"Config {cfg} -> Val RMSE: {current_val_rmse:.4f}")

                if current_val_rmse < best_val_rmse:
                    best_val_rmse = current_val_rmse
                    best_cfg = cfg

        print(f"Best Config Found: {best_cfg}")

        # --- PHASE 2: RETRAINING (Train on 80% [Train + Val], Test on 20%) ---
        print("\n--- Phase 2: Retraining on Train + Val (100 Epochs) ---")
        
        full_train_ds = ConcatDataset([train_ds, val_ds])
        full_train_loader = DataLoader(full_train_ds, batch_size=best_cfg["batch"], shuffle=True)
        # Dummy Val Loader (just for the function signature)
        val_loader_monitor = DataLoader(val_ds, batch_size=best_cfg["batch"])
        test_loader = DataLoader(test_ds, batch_size=best_cfg["batch"])

        final_model = TransformerModel(
            input_size=1,
            d_model=best_cfg["d_model"],
            nhead=best_cfg["nhead"],
            num_layers=best_cfg["layers"],
            dropout=best_cfg["dropout"]
        )

        # Retrain for 100 Epochs
        # FIXED: Now unpacking 5 values
        final_model, t_hist, v_hist, t_time, _ = train_model(
            final_model, 
            full_train_loader, 
            val_loader_monitor, 
            epochs=100, 
            lr=best_cfg["lr"], 
            device=device,
            early_stop_patience=15  # Use 15 for retraining consistency
        )
        
        final_params = sum(p.numel() for p in final_model.parameters())

        # --- PHASE 3: EVALUATION ---
        final_model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = final_model(xb.to(device)).squeeze()
                preds_list.append(out.cpu().numpy())
                trues_list.append(yb.cpu().numpy())

        preds = scaler.inverse_transform(np.concatenate(preds_list).reshape(-1,1)).ravel()
        trues = scaler.inverse_transform(np.concatenate(trues_list).reshape(-1,1)).ravel()

        metrics = calculate_metrics(trues, preds)

        # Naive Baseline
        naive_preds = np.roll(trues, 1)
        naive_preds[0] = trues[0]
        wilcoxon_res = perform_wilcoxon_test(trues, preds, naive_preds)
        # --- ADD THIS TO TRANSFORMER ---
        plt.figure()
        plt.plot(t_hist, label="Train Loss (Combined)")
        plt.title(f"Transformer Learning Curve - {freq_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"learning_curve_transformer_{freq_name}.png"))
        plt.close()

        # --- FINAL PRINTOUT ---
        print(f"\n{'='*20} RESULTS FOR: {exp_name} {'='*20}")
        print(f"Best Hyperparameters: {best_cfg}")
        print(f"Parameter Count: {final_params}")
        print(f"Final Retraining Time (s): {t_time:.2f}")

        print(f"\nMetrics on Test Set:")
        print(f"{'Metric':<10} | {'Transformer':<15} | {'Naive':<15}")
        print(f"{'-'*10}-|-{'-'*15}-|-{'-'*15}")

        naive_metrics = calculate_metrics(trues, naive_preds)
        for k in ["RMSE", "MSE", "MAE", "MAPE", "R2"]:
            print(f"{k:<10} | {metrics[k]:<15.4f} | {naive_metrics[k]:<15.4f}")

        sig_msg = "(Significant)" if wilcoxon_res['Significant'] else "(Not Significant)"
        print(f"\nWilcoxon Test p-value: {wilcoxon_res['p-value']:.6f} {sig_msg}")

        # Plotting
        plot_actual_vs_predicted(trues, preds, freq_name, save_dir, "Transformer")
        print(f"Plot saved to {save_dir}/Transformer_{freq_name}_comparison.png")
        print(f"{'='*20} FINISHED {'='*20}\n")
