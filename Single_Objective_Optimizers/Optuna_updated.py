import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import time
import os
import optuna

# --- Metrics ---
def calculate_metrics(y_true, y_pred):
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    y_true_mape, y_pred_mape = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true_mape != 0
    if np.any(non_zero_mask):
        metrics['mape'] = np.mean(np.abs((y_true_mape[non_zero_mask] - y_pred_mape[non_zero_mask]) / y_true_mape[non_zero_mask])) * 100
    else:
        metrics['mape'] = 0.0
    return metrics

def plot_actual_vs_predicted(test_trues, test_preds, save_dir, freq_label):
    plt.figure(figsize=(11, 5))
    steps = np.arange(len(test_trues))
    plt.plot(steps, test_trues, label="Actual", linewidth=2)
    plt.plot(steps, test_preds, label="Predicted", linewidth=2)
    plt.title(f"Actual vs Predicted - {freq_label}")
    plt.xlabel("Time Step")
    plt.ylabel("Global_active_power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(save_dir, f'actual_vs_predicted_{freq_label.lower()}.png')
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_dir, freq_label):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Learning Curve for Best Model - {freq_label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(save_dir, f'learning_curve_{freq_label.lower()}.png')
    plt.savefig(filename, dpi=200)
    plt.close()

def get_datasets_for_frequency(file_path, freq_code, time_step):
    try:
        df = pd.read_csv(file_path, sep=';', low_memory=False)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None, None, None

    try:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df = df.set_index('DateTime')
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        df.replace('?', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df = df.dropna()
        if df.empty: return None, None, None, None
        
        df = df[['Global_active_power']].astype(float)
        df_resampled = df.resample(freq_code).mean()
        df_resampled.ffill(inplace=True)
        df_resampled.bfill(inplace=True)
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None, None, None

    n = len(df_resampled)
    if n < time_step * 2: return None, None, None, None

    train_end, val_end = int(n * 0.64), int(n * 0.80)
    train_df, val_df, test_df = df_resampled.iloc[:train_end], df_resampled.iloc[train_end:val_end], df_resampled.iloc[val_end:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    def create_dataset(data, time_step):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step, 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_scaled, time_step)
    X_val, y_val = create_dataset(val_scaled, time_step)
    X_test, y_test = create_dataset(test_scaled, time_step)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0: return None, None, None, None

    X_train, X_val, X_test = X_train.reshape(-1, time_step, 1), X_val.reshape(-1, time_step, 1), X_test.reshape(-1, time_step, 1)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return train_ds, val_ds, test_ds, scaler

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda', early_stop_patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    start_time = time.time()
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        batch_train_losses = [loss.item() for xb, yb in train_loader if (optimizer.zero_grad(), preds := model(xb.to(device)), loss := criterion(preds, yb.to(device)), loss.backward(), optimizer.step(), True)]
        train_losses.append(np.mean(batch_train_losses))

        model.eval()
        val_preds, val_trues, batch_val_losses = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device))
                val_preds.append(preds.cpu().numpy())
                val_trues.append(yb.cpu().numpy())
                batch_val_losses.append(criterion(preds, yb.to(device)).item())
        
        val_losses.append(np.mean(batch_val_losses))
        val_preds, val_trues = np.concatenate(val_preds), np.concatenate(val_trues)
        val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience: break
                
    train_time = time.time() - start_time
    if best_model_state: model.load_state_dict(best_model_state)
    return model, best_val_rmse, train_time, train_losses, val_losses

if __name__ == "__main__":
    # --- ADDED FOR FAIRNESS: Set random seeds ---
    np.random.seed(42)
    torch.manual_seed(42)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    FREQUENCIES_TO_RUN = {'Minutely': 'T', 'Hourly': 'H', 'Daily': 'D', 'Weekly': 'W'}
    TIME_STEPS = {'Minutely': 60, 'Hourly': 60, 'Daily': 30, 'Weekly': 8}
    bounds = [[0, 1]] * 5
    file_path = 'household_power_consumption.txt'

    for freq_name, freq_code in FREQUENCIES_TO_RUN.items():
        print(f"\n{'='*25} RUNNING OPTUNA FOR {freq_name} FREQUENCY {'='*25}")
        
        train_ds, val_ds, test_ds, scaler = get_datasets_for_frequency(
            file_path, freq_code, TIME_STEPS[freq_name]
        )
        
        if train_ds is None:
            print(f"Skipping {freq_name} due to preprocessing error or insufficient data.")
            continue

        def objective(trial):
            # Define hyperparameter search space
            hidden_size = trial.suggest_int('hidden_size', 32, 128)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            
            if len(train_loader) == 0 or len(val_loader) == 0: return float('inf')

            model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            
            # The fitness function is the validation RMSE from training
            _, val_rmse, _, _, _ = train_model(
                model, train_loader, val_loader, epochs=50, lr=lr, device=DEVICE, early_stop_patience=10
            )
            return val_rmse if np.isfinite(val_rmse) else float('inf')

        # Create a seeded sampler for reproducibility
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Run optimization with a 25-trial budget to match other algorithms
        study.optimize(objective, n_trials=25, show_progress_bar=True)
        
        best_params = study.best_params
        best_rmse = study.best_value

        if not np.isfinite(best_rmse):
            print(f"Optimization failed for {freq_name}. Skipping.")
            continue

        print("\nRetraining the best model on final hyperparameters...")
        # Create final model with best params
        final_batch_size = best_params['batch_size']
        final_model = LSTMModel(input_size=1, hidden_size=best_params['hidden_size'], 
                                num_layers=best_params['num_layers'], dropout=best_params['dropout'])
        
        train_loader = DataLoader(train_ds, batch_size=final_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=final_batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=final_batch_size, shuffle=False)

        final_model, val_rmse, train_time, train_losses, val_losses = train_model(
            final_model, train_loader, val_loader, epochs=50, lr=best_params['lr'], 
            device=DEVICE, early_stop_patience=10
        )

        final_model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = final_model(xb.to(DEVICE))
                test_preds.append(preds.cpu().numpy())
                test_trues.append(yb.cpu().numpy())
        
        if not test_preds:
            print(f"No predictions were made for {freq_name}. Skipping final evaluation.")
            continue

        test_preds, test_trues = np.concatenate(test_preds).reshape(-1, 1), np.concatenate(test_trues).reshape(-1, 1)
        test_preds_inv, test_trues_inv = scaler.inverse_transform(test_preds).flatten(), scaler.inverse_transform(test_trues).flatten()
        
        metrics = calculate_metrics(test_trues_inv, test_preds_inv)
        param_count = count_parameters(final_model)

        save_dir = os.path.join('results_optuna_fair_50epochs', freq_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--- Results for {freq_name} ---")
        print(f"Best Hyperparameters -> Hidden size: {best_params['hidden_size']}, Layers: {best_params['num_layers']}, Dropout: {best_params['dropout']:.3f}, LR: {best_params['lr']:.2e}, Batch size: {final_batch_size}")
        print(f"Final Model -> Val RMSE: {val_rmse:.5f}, Test RMSE: {metrics['rmse']:.5f}, Training Time: {train_time:.2f}s, Params: {param_count}")

        plot_actual_vs_predicted(test_trues_inv, test_preds_inv, save_dir, freq_name)
        plot_loss_curve(train_losses, val_losses, save_dir, freq_name)
        
        errors_predicted = np.abs(test_trues_inv - test_preds_inv)
        if len(errors_predicted) > 20:
             stat, p_value = wilcoxon(errors_predicted)
             print(f"Wilcoxon test p-value: {p_value:.6f}")
        else:
             print("Not enough samples for Wilcoxon test.")

        print(f"Detailed Test Metrics for {freq_name}:")
        for k, v in metrics.items():
            print(f"  {k.upper():<6}: {v:.4f}")

        print(f"Plots and results saved in '{save_dir}'")

    print("\nAll frequency experiments completed.")
