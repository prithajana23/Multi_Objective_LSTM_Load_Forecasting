import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import wilcoxon
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Data Loading & Preprocessing for UCI Dataset ---
def get_datasets(freq, time_step=60):
    """
    Loads, preprocesses, and prepares the UCI dataset for a specific time frequency.
    """
    print(f"--- Loading UCI dataset with frequency: {freq} and time_step: {time_step} ---")

    # --- IMPORTANT: Update this path to your UCI dataset file ---
    file_path = 'household_power_consumption.txt'
    # ----------------------------------------------------------------

    if not os.path.exists(file_path):
        print(f"!!! File not found at {file_path}. Please update the path. Skipping this run. !!!")
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

    df_resampled = df[power_column].resample(freq).mean().to_frame()
    df_resampled.ffill(inplace=True)
    df_resampled.bfill(inplace=True)

    data = df_resampled.values
    train_size = int(len(data) * 0.6)
    val_size = int(len(data) * 0.2)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    if len(train_data) < time_step + 1 or len(val_data) < time_step + 1 or len(test_data) < time_step + 1:
        print("!!! Not enough data for splits with the given time_step. Skipping. !!!")
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
        print("!!! Not enough data to create sequences for all sets. Skipping. !!!")
        return None, None, None, None

    X_train = X_train.reshape(-1, time_step, 1)
    X_val = X_val.reshape(-1, time_step, 1)
    X_test = X_test.reshape(-1, time_step, 1)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    return train_ds, val_ds, test_ds, scaler

# --- 2. LSTM Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

# --- 3. Model Training Function ---
def train_model(model, train_loader, val_loader, epochs, lr, device, early_stop_patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_rmse = np.inf
    epochs_no_improve = 0
    best_model_state = None
    start_time = time.time()
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_loss_history.append(epoch_train_loss / len(train_loader))

        model.eval()
        val_preds, val_trues = [], []
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_preds.append(preds.cpu().numpy())
                val_trues.append(yb.cpu().numpy())
                loss = criterion(preds, yb)
                epoch_val_loss += loss.item()
        val_loss_history.append(epoch_val_loss / len(val_loader))

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))

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

    return model, best_val_rmse, train_time, train_loss_history, val_loss_history

# --- 4. MOPSO Optimizer ---
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1])
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.score = [np.inf, np.inf, np.inf] # [RMSE, Time, Params]
        self.best_score = [np.inf, np.inf, np.inf]

class MOPSO:
    def __init__(self, problem_func, bounds, swarm_size=5, max_iter=5, w=0.5, c1=1.5, c2=1.5):
        self.problem_func = problem_func
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm = [Particle(bounds) for _ in range(swarm_size)]
        self.archive = []
        self.convergence_history = []

    def _dominates(self, a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def _update_archive(self):
        combined = self.archive + self.swarm
        non_dominated = []
        for p in combined:
            is_dominated = False
            for other in combined:
                if self._dominates(other.score, p.score):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(p)
        self.archive = sorted(non_dominated, key=lambda p: p.score[0])[:100]

    def _select_leader(self):
        if not self.archive:
            return self.swarm[0].best_position
        idx = np.random.choice(len(self.archive))
        return self.archive[idx].position

    def optimize(self):
        for t in range(self.max_iter):
            print(f"MOPSO Iteration {t + 1}/{self.max_iter}")
            for particle in self.swarm:
                particle.score = self.problem_func(particle.position)
                if self._dominates(particle.score, particle.best_score):
                    particle.best_position = np.copy(particle.position)
                    particle.best_score = particle.score

            self._update_archive()
            if not self.archive: continue
            
            best_rmse_in_iter = min(p.score[0] for p in self.archive)
            self.convergence_history.append(best_rmse_in_iter)
            print(f"Archive size: {len(self.archive)}, Best RMSE in archive: {best_rmse_in_iter:.6f}")

            gbest = self._select_leader()
            
            for particle in self.swarm:
                r1, r2 = np.random.random(2)
                particle.velocity = (
                    self.w * particle.velocity +
                    self.c1 * r1 * (particle.best_position - particle.position) +
                    self.c2 * r2 * (gbest - particle.position)
                )
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, self.bounds[:, 0], self.bounds[:, 1])

        return self.archive, self.convergence_history

# --- 5. Hyperparameter Decoding & Evaluation Wrapper ---
def decode_position(position):
    hidden_size = int(32 + position[0] * (128 - 32))
    num_layers = int(1 + position[1] * (3 - 1))
    dropout = position[2] * 0.5
    lr = 10 ** (-5 + position[3] * ((-3) - (-5)))
    batch_size = int(32 + position[4] * (256 - 32))
    return hidden_size, num_layers, dropout, lr, batch_size

def evaluate_hyperparams(position, train_ds, val_ds, device):
    hidden_size, num_layers, dropout, lr, batch_size = decode_position(position)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    
    if len(train_loader) == 0 or len(val_loader) == 0:
        return [np.inf, np.inf, np.inf]

    _, val_rmse, train_time, _, _ = train_model(model, train_loader, val_loader, epochs=50, lr=lr, device=device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  H={hidden_size}, L={num_layers}, D={dropout:.3f}, LR={lr:.1e}, BS={batch_size} -> RMSE={val_rmse:.6f}, Time={train_time:.2f}s, Params={param_count}")
    return [val_rmse, train_time, param_count]

# --- 6. Metrics & Visualization ---
def calculate_metrics(y_true, y_pred):
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    y_true_mape, y_pred_mape = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true_mape != 0
    metrics['mape'] = np.mean(np.abs((y_true_mape[non_zero_mask] - y_pred_mape[non_zero_mask]) / y_true_mape[non_zero_mask])) * 100
    return metrics

def plot_results(pareto_solutions, convergence, train_hist, val_hist, base_save_dir, exp_name):
    save_dir = os.path.join(base_save_dir, exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scores = np.array([s.score for s in pareto_solutions])
    ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c='b', marker='o')
    ax.set_xlabel('RMSE (Lower is Better)')
    ax.set_ylabel('Training Time (s)')
    ax.set_zlabel('Parameter Count')
    ax.set_title(f'Pareto Front for {exp_name}')
    plt.savefig(os.path.join(save_dir, 'pareto_front.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(convergence, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Best RMSE in Archive')
    plt.title(f'MOPSO Convergence for {exp_name}')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'convergence.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label='Training Loss')
    plt.plot(val_hist, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Learning Curve for Best Model - {exp_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    plt.show()
    plt.close()

# --- 7. Main Experiment Runner ---
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    FREQUENCIES_TO_RUN = {
        'Minutely': 'T', 'Hourly': 'H', 'Daily': 'D', 'Weekly': 'W'
    }
    BOUNDS = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

    for freq_name, freq_code in FREQUENCIES_TO_RUN.items():
        exp_name = f"UCI_{freq_name}"
        print(f"\n{'='*20} STARTING EXPERIMENT: {exp_name} {'='*20}")

        if freq_code == 'W':
            time_step_for_run = 8
        elif freq_code == 'D':
            time_step_for_run = 30
        else:
            time_step_for_run = 60

        train_ds, val_ds, test_ds, scaler = get_datasets(freq_code, time_step=time_step_for_run)
        if train_ds is None: continue

        problem_func = lambda pos: evaluate_hyperparams(pos, train_ds, val_ds, DEVICE)
        # --- Use MOPSO instead of MOGWO ---
        optimizer = MOPSO(problem_func, BOUNDS, swarm_size=5, max_iter=5)
        pareto_solutions, convergence_history = optimizer.optimize()
        
        if not pareto_solutions:
            print(f"!!! MOPSO found no solutions for {exp_name}. Skipping. !!!")
            continue

        best_solution = sorted(pareto_solutions, key=lambda s: s.score[0])[0]
        hs, nl, dr, lr, bs = decode_position(best_solution.position)

        full_train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
        full_train_loader = DataLoader(full_train_ds, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        print(f"\n--- Re-training best model for {exp_name} ---")
        final_model = LSTMModel(input_size=1, hidden_size=hs, num_layers=nl, dropout=dr)
        final_model, _, _, train_hist, val_hist = train_model(final_model, full_train_loader, DataLoader(val_ds, batch_size=bs), epochs=100, lr=lr, device=DEVICE, early_stop_patience=10)

        final_model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = final_model(xb)
                test_preds.append(preds.cpu().numpy())
                test_trues.append(yb.cpu().numpy())
        test_preds = scaler.inverse_transform(np.concatenate(test_preds).reshape(-1, 1)).flatten()
        test_trues = scaler.inverse_transform(np.concatenate(test_trues).reshape(-1, 1)).flatten()
        optimized_metrics = calculate_metrics(test_trues, test_preds)

        print(f"\n--- Training baseline model for {exp_name} ---")
        baseline_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.1)
        baseline_model, _, _, _, _ = train_model(baseline_model, full_train_loader, DataLoader(val_ds, batch_size=128), epochs=100, lr=1e-4, device=DEVICE, early_stop_patience=10)
        
        baseline_model.eval()
        base_preds, base_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = baseline_model(xb)
                base_preds.append(preds.cpu().numpy())
                base_trues.append(yb.cpu().numpy())
        base_preds = scaler.inverse_transform(np.concatenate(base_preds).reshape(-1, 1)).flatten()
        base_trues = scaler.inverse_transform(np.concatenate(base_trues).reshape(-1, 1)).flatten()
        baseline_metrics = calculate_metrics(base_trues, base_preds)

        errors_optimized = np.abs(test_trues - test_preds)
        errors_baseline = np.abs(base_trues - base_preds)
        min_len = min(len(errors_optimized), len(errors_baseline))
        stat, p_value = wilcoxon(errors_optimized[:min_len], errors_baseline[:min_len])

        print(f"\n{'='*20} RESULTS FOR: {exp_name} {'='*20}")
        print(f"\nOptimized Hyperparameters:")
        print(f"  Hidden Size: {hs}, Layers: {nl}, Dropout: {dr:.3f}, LR: {lr:.1e}, Batch Size: {bs}")
        print(f"  Parameter Count: {best_solution.score[2]}")
        print(f"  Training Time (s): {best_solution.score[1]:.2f}")
        
        print("\nMetrics on Test Set:")
        print(f"          | {'Optimized':<15} | {'Baseline':<15}")
        print(f"----------|-----------------|----------------")
        for metric in optimized_metrics:
            print(f"{metric.upper():<10}| {optimized_metrics[metric]:<15.4f} | {baseline_metrics[metric]:<15.4f}")

        print(f"\nWilcoxon Rank-Sum Test (p-value): {p_value:.6f}")
        if p_value < 0.05:
            print("  -> The difference between the optimized and baseline models is statistically significant.")
        else:
            print("  -> The difference is not statistically significant.")

        plot_results(pareto_solutions, convergence_history, train_hist, val_hist, base_save_dir='results', exp_name=exp_name)
        print(f"\nPlots for {exp_name} saved to 'results/{exp_name}/'")
        print(f"{'='*20} FINISHED EXPERIMENT: {exp_name} {'='*20}\n")
