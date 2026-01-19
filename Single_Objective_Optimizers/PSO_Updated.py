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

# --- Metrics ---
def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    non_zero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0
    return {'rmse': rmse, 'mse': mse, 'mae': mae, 'mape': mape, 'r2': r2}

def plot_actual_vs_predicted(test_trues, test_preds, save_dir):
    plt.figure(figsize=(11, 5))
    steps = np.arange(len(test_trues))
    plt.plot(steps, test_trues, label="Actual", linewidth=2)
    plt.plot(steps, test_preds, label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted - Minutely")
    plt.xlabel("Time Step")
    plt.ylabel("Global_active_power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(save_dir, 'actual_vs_predicted_minutely.png')
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Learning Curve for Best Model - Minutely')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(save_dir, 'learning_curve_minutely.png')
    plt.savefig(filename, dpi=200)
    plt.close()

# --- Dataset loader and preprocessing ---
def get_datasets(file_path, time_step=60):
    df = pd.read_csv(file_path, sep=';')
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.set_index('DateTime')
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df = df.dropna()
    df = df[['Global_active_power']].astype(float)
    df_resampled = df.resample('T').mean()  # Minutely frequency
    df_resampled.ffill(inplace=True)
    df_resampled.bfill(inplace=True)
    n = len(df_resampled)
    train_end, val_end = int(n * 0.64), int(n * 0.80)
    train_df, val_df, test_df = df_resampled.iloc[:train_end], df_resampled.iloc[train_end:val_end], df_resampled.iloc[val_end:]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    def create_dataset(data, ts):
        X, Y = [], []
        for i in range(len(data) - ts):
            X.append(data[i:i + ts, 0])
            Y.append(data[i + ts, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_scaled, time_step)
    X_val, y_val = create_dataset(val_scaled, time_step)
    X_test, y_test = create_dataset(test_scaled, time_step)

    X_train = X_train.reshape(-1, time_step, 1)
    X_val = X_val.reshape(-1, time_step, 1)
    X_test = X_test.reshape(-1, time_step, 1)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return train_ds, val_ds, test_ds, scaler

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def decode_position(x):
    hidden_size = int(32 + x[0] * 96)
    num_layers = int(1 + x[1] * 2)
    num_layers = min(max(num_layers, 1), 3)
    dropout = x[2] * 0.5
    lr = 10 ** (-4 - x[3])
    batch_size = int(32 + x[4] * 96)
    batch_size = min(max(batch_size, 1), 128)
    return hidden_size, num_layers, dropout, lr, batch_size

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda', early_stop_patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_rmse = np.inf
    epochs_no_improve = 0
    best_model_state = None
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        batch_train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        train_losses.append(np.mean(batch_train_losses))

        model.eval()
        val_preds, val_trues, batch_val_losses = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_preds.append(preds.cpu().numpy())
                val_trues.append(yb.cpu().numpy())
                batch_val_losses.append(criterion(preds, yb).item())
        val_losses.append(np.mean(batch_val_losses))
        
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        print(f"Epoch {epoch+1}/{epochs}, val RMSE: {val_rmse:.6f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    train_time = time.time() - start_time
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_rmse, train_time, train_losses, val_losses

def evaluate_particle(position, train_ds, val_ds):
    hs, nl, dr, lr, bs = decode_position(position)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    model = LSTMModel(input_size=1, hidden_size=hs, num_layers=nl, dropout=dr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, val_rmse, train_time, _, _ = train_model(model, train_loader, val_loader,
                                                   epochs=50, lr=lr, device=device,
                                                   early_stop_patience=10)
    param_count = count_parameters(model)
    print(f"Particle eval: RMSE={val_rmse:.6f}, Time={train_time:.2f}s, Params={param_count}")
    return val_rmse

class PSO:
    def __init__(self, problem, bounds, swarm_size=5, max_iter=5,
                 w=0.7, c1=1.5, c2=1.5):
        self.problem = problem
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = bounds.shape[0]
        self.positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (swarm_size, self.dim))
        self.velocities = np.zeros((swarm_size, self.dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([np.inf] * swarm_size)
        self.gbest_position = None
        self.gbest_score = np.inf

    def optimize(self):
        for iter in range(self.max_iter):
            print(f"Iteration {iter+1}/{self.max_iter}")
            for i in range(self.swarm_size):
                score = self.problem(self.positions[i])
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])
        return self.gbest_position, self.gbest_score

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    file_path = 'household_power_consumption.txt'
    time_step = 60

    train_ds, val_ds, test_ds, scaler = get_datasets(file_path, time_step)
    bounds = np.array([[0, 1]] * 5)

    pso = PSO(lambda pos: evaluate_particle(pos, train_ds, val_ds),
              bounds, swarm_size=5, max_iter=5, w=0.7, c1=1.5, c2=1.5)

    best_position, best_rmse = pso.optimize()

    hs, nl, dr, lr, bs = decode_position(best_position)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(input_size=1, hidden_size=hs, num_layers=nl, dropout=dr)
    # Final training with 100 epochs for best hyperparameters
    model, val_rmse, train_time, train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=100, lr=lr, device=device, early_stop_patience=10)

    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            test_preds.append(preds.cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    test_preds = np.concatenate(test_preds).reshape(-1, 1)
    test_trues = np.concatenate(test_trues).reshape(-1, 1)

    test_preds_inv = scaler.inverse_transform(test_preds).flatten()
    test_trues_inv = scaler.inverse_transform(test_trues).flatten()

    metrics = calculate_metrics(test_trues_inv, test_preds_inv)
    param_count = count_parameters(model)

    save_dir = 'results_pso_minutely_100epochs'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nBest hyperparameters found by PSO:")
    print(f"Hidden Size: {hs}, Num Layers: {nl}, Dropout: {dr:.3f}, LR: {lr:.2e}, Batch Size: {bs}")
    print(f"Validation RMSE: {val_rmse:.6f}")
    print(f"Test Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper():<6}: {v:.6f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Total Trainable Parameters: {param_count}")

    plot_actual_vs_predicted(test_trues_inv, test_preds_inv, save_dir)
    plot_loss_curve(train_losses, val_losses, save_dir)
    
    # Wilcoxon test on prediction errors (optional)
    errors_predicted = np.abs(test_trues_inv - test_preds_inv)
    if len(errors_predicted) > 20:
        stat, p_value = wilcoxon(errors_predicted)
        print(f"Wilcoxon test p-value: {p_value:.6f}")
    else:
        print("Not enough samples for Wilcoxon test.")

