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
import math # For isinf

# --------------------------------------------------------------------------
#
#    UPDATE FILE PATH:
#    Update the 'file_path' variable below to point to your dataset.
# --------------------------------------------------------------------------


# --- 1. Data Loading & Preprocessing ---
def get_datasets(freq, time_step=60):
    print(f"--- Loading dataset with frequency: {freq} and time_step: {time_step} ---")
    
    # --- !!! UPDATE THIS PATH !!! ---
    file_path = 'household_power_consumption.txt'
    # -----------------------------------

    if not os.path.exists(file_path):
        print(f"!!! File not found at {file_path}. Skipping this run. !!!")
        print("!!! Please update the 'file_path' variable in the code. !!!")
        return None, None, None, None

    df = pd.read_csv(file_path, sep=';', low_memory=False)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.set_index('DateTime')
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    power_column = 'Global_active_power'
    df = df[[power_column]].dropna()
    df.ffill(inplace=True); df.bfill(inplace=True) # Fill missing values

    df_resampled = df[power_column].resample(freq).mean().to_frame()
    df_resampled.ffill(inplace=True); df_resampled.bfill(inplace=True) # Fill gaps after resampling

    data = df_resampled.values
    train_size, val_size = int(len(data) * 0.6), int(len(data) * 0.2)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    if len(train_data) < time_step + 1 or len(val_data) < time_step + 1 or len(test_data) < time_step + 1:
        print(f"!!! Not enough data for time_step={time_step}. Skipping. !!!")
        return None, None, None, None

    scaler = MinMaxScaler()
    scaler.fit(train_data) # Fit only on training data
    scaled_train = scaler.transform(train_data)
    scaled_val = scaler.transform(val_data)
    scaled_test = scaler.transform(test_data)

    def create_sequences(data, time_step):
        X, Y = [], []
        # Ensure index range is valid
        if len(data) <= time_step:
             return np.array([]), np.array([])
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_sequences(scaled_train, time_step)
    X_val, y_val = create_sequences(scaled_val, time_step)
    X_test, y_test = create_sequences(scaled_test, time_step)

    # Check if sequence creation resulted in empty arrays
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
         print("!!! Sequence creation resulted in empty sets. Skipping. !!!")
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
        out = self.fc(out)
        return out.squeeze(-1)

def count_parameters(model):
    """Counts the trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 3. Model Training Function (for search phase) ---
def train_model_search(model, train_loader, val_loader, epochs, lr, device, early_stop_patience=5):
    """Trains model, used during hyperparameter search."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_rmse = np.inf
    epochs_no_improve = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device))
                val_preds.append(preds.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        if not val_preds:
            val_rmse = np.inf
        else:
            val_preds = np.concatenate(val_preds).flatten()
            val_trues = np.concatenate(val_trues).flatten()
            if np.any(~np.isfinite(val_preds)):
                 val_rmse = np.inf
            else:
                 val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))

        if not np.isfinite(val_rmse):
             break # Stop early if model diverges

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                # print(f"  -> Early stopping triggered at epoch {epoch+1}")
                break

    train_time = time.time() - start_time
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_val_rmse, train_time, [], [] # Return empty history lists

# --- 4. MOEA/D Optimizer ---
class MOEAD:
    def __init__(self, problem_func, bounds, population_size=5, max_iter=4, n_objs=3, T=3):
        self.problem_func = problem_func
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter # Total generations
        self.n_objs = n_objs
        self.T = min(T, population_size) # Neighborhood size cannot exceed population size

        self.population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, len(bounds)))
        self.weights = self._generate_weight_vectors()
        self.neighbors = self._initialize_neighborhoods()
        self.F = np.full((population_size, n_objs), np.inf) # Stores objective values for each individual
        self.z = np.full(n_objs, np.inf) # Reference point (ideal point)

    def _generate_weight_vectors(self):
        W = np.random.rand(self.population_size, self.n_objs)
        norm = np.linalg.norm(W, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return W / norm

    def _initialize_neighborhoods(self):
        dist = np.linalg.norm(self.weights[:, np.newaxis, :] - self.weights[np.newaxis, :, :], axis=2)
        return np.argsort(dist, axis=1)[:, :self.T]

    def _tchebycheff(self, F_val, weight):
        F_val_finite = np.nan_to_num(F_val, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        z_finite = np.nan_to_num(self.z, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        return np.max(weight * np.abs(F_val_finite - z_finite))

    def _mutation_and_crossover(self, i, k1, k2, k3, F_mut=0.5, CR=0.9):
        """ Performs DE/rand/1 mutation and binomial crossover. """
        x = self.population[i]
        x1, x2, x3 = self.population[k1], self.population[k2], self.population[k3]
        v = x1 + F_mut * (x2 - x3)
        v = np.clip(v, self.bounds[:, 0], self.bounds[:, 1])
        j_rand = np.random.randint(0, len(x))
        mask = np.random.rand(len(x)) < CR
        mask[j_rand] = True
        child = np.where(mask, v, x)
        return child

    def optimize(self):
        print("--- Evaluating initial MOEA/D population ---")
        initial_F_raw = [self.problem_func(self.population[i]) for i in range(self.population_size)]
        self.F = np.array([np.nan_to_num(f, nan=np.inf, posinf=np.inf, neginf=-np.inf) for f in initial_F_raw])

        finite_initial_F = self.F[np.all(np.isfinite(self.F), axis=1)]
        self.z = np.min(finite_initial_F, axis=0) if finite_initial_F.size > 0 else np.full(self.n_objs, np.inf)

        finite_initial_rmses = self.F[:, 0][np.isfinite(self.F[:, 0])]
        initial_best_rmse = np.min(finite_initial_rmses) if finite_initial_rmses.size > 0 else np.inf
        convergence_history = [initial_best_rmse]
        print(f"  Initial Best RMSE: {initial_best_rmse:.6f}")
        n_evals_total = self.population_size

        for gen in range(self.max_iter):
            print(f"MOEA/D Iteration {gen + 1}/{self.max_iter}")
            evals_this_gen = 0
            for i in range(self.population_size):
                current_T = min(self.T, self.population_size)
                if current_T < 3:
                     parents_idx = np.random.choice(np.arange(self.population_size), 3, replace=False)
                     parents_idx = parents_idx[parents_idx != i]
                     if len(parents_idx) < 3:
                         parents_idx = np.random.choice(np.arange(self.population_size), 3, replace=True)
                else:
                    parents_idx = np.random.choice(self.neighbors[i], 3, replace=False)
                k1, k2, k3 = parents_idx[0], parents_idx[1], parents_idx[2]
                child = self._mutation_and_crossover(i, k1, k2, k3)
                child_f_raw = self.problem_func(child)
                child_f = np.nan_to_num(child_f_raw, nan=np.inf, posinf=np.inf, neginf=-np.inf)
                evals_this_gen += 1
                
                # --- !!! THIS LINE IS FIXED !!! ---
                # It now correctly handles [inf, inf, inf] without crashing
                self.z = np.minimum(self.z, child_f)
                # ----------------------------------
                
                for j in self.neighbors[i]:
                    if self._tchebycheff(child_f, self.weights[j]) < self._tchebycheff(self.F[j], self.weights[j]):
                        self.population[j] = child
                        self.F[j] = child_f

            current_finite_rmses = self.F[:, 0][np.isfinite(self.F[:, 0])]
            best_rmse_gen = np.min(current_finite_rmses) if current_finite_rmses.size > 0 else np.inf
            convergence_history.append(best_rmse_gen)
            n_evals_total += evals_this_gen
            print(f"  n_eval={n_evals_total} | Best RMSE in population: {best_rmse_gen:.6f}")

        return self._get_non_dominated_solutions(), convergence_history

    def _dominates(self, p_score, q_score):
        if np.any(~np.isfinite(p_score)): return False
        finite_q = np.nan_to_num(q_score, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        better_or_equal = np.all(p_score <= finite_q)
        strictly_better = np.any(p_score < finite_q)
        return better_or_equal and strictly_better

    def _get_non_dominated_solutions(self):
         finite_mask = np.all(np.isfinite(self.F), axis=1)
         finite_indices = np.where(finite_mask)[0]
         if not finite_indices.size > 0: return []
         finite_F = self.F[finite_indices]
         finite_population = self.population[finite_indices]
         non_dominated_mask = np.ones(len(finite_indices), dtype=bool)
         for i in range(len(finite_indices)):
             for j in range(len(finite_indices)):
                 if i != j and self._dominates(finite_F[j], finite_F[i]):
                     non_dominated_mask[i] = False
                     break
         original_indices = finite_indices[non_dominated_mask]
         class Solution:
             def __init__(self, position, score):
                 self.position = position
                 self.score = score
         return [Solution(self.population[i], self.F[i]) for i in original_indices]


# --- 5. Hyperparameter Decoding & Evaluation Wrapper ---
def decode_position(position):
    """Decodes a normalized position vector into hyperparameters."""
    position = np.clip(position, 0, 1)
    hidden_size = int(round(32 + position[0] * (128 - 32)))
    hidden_size = max(32, min(hidden_size, 128))
    num_layers = int(round(1 + position[1] * (3 - 1)))
    num_layers = max(1, min(num_layers, 3))
    dropout = position[2] * 0.5
    dropout = max(0.0, min(dropout, 0.5))
    lr = 10**(-5 + position[3] * 2)
    lr = max(1e-5, min(lr, 1e-3))
    batch_size = int(round(32 + position[4] * (256 - 32)))
    batch_size = max(32, min(batch_size, 256))
    return hidden_size, num_layers, dropout, lr, batch_size

def evaluate_hyperparams(position, train_ds, val_ds, device):
    """Evaluates hyperparameters, returns objectives [RMSE, Time, Params]."""
    try:
        hs, nl, dr, lr, bs = decode_position(position)
        if not (32 <= hs <= 128 and 1 <= nl <= 3 and 0.0 <= dr <= 0.5 and 1e-5 <= lr <= 1e-3 and 32 <= bs <= 256):
             return [np.inf, np.inf, np.inf]
        if train_ds is None or val_ds is None or len(train_ds) == 0 or len(val_ds) == 0:
             return [np.inf, np.inf, np.inf]
        bs_train = min(bs, len(train_ds))
        bs_val = min(bs, len(val_ds))
        if bs_train == 0 or bs_val == 0: return [np.inf, np.inf, np.inf]

        train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs_val, shuffle=False)

        if len(train_loader) == 0 or len(val_loader) == 0:
            return [np.inf, np.inf, np.inf]

        model = LSTMModel(input_size=1, hidden_size=hs, num_layers=nl, dropout=dr)
        params = count_parameters(model)
        model, val_rmse, train_time, _, _ = train_model_search(
            model, train_loader, val_loader,
            epochs=50, lr=lr, device=device, early_stop_patience=5
        )
        val_rmse = val_rmse if np.isfinite(val_rmse) else np.inf
        train_time = train_time if np.isfinite(train_time) else np.inf
        print(f"  Eval -> H={hs}, L={nl}, D={dr:.3f}, LR={lr:.1e}, BS={bs} | RMSE={val_rmse:.6f}, Time={train_time:.2f}s, Params={params}")
        return [val_rmse, train_time, float(params)]
    except Exception as e:
        print(f"  -> Error during evaluation: {e}. Returning inf.")
        return [np.inf, np.inf, np.inf]


# --- 6. Final Test Function ---
def run_final_test(params_dict, train_ds, val_ds, test_ds, scaler, device):
    """
    Retrains a model with given hyperparameters and evaluates on the test set.
    Returns: dict with all hyperparameters, metrics, time, param count, and predictions.
    """
    hs, nl, dr, lr, bs = params_dict['H'], params_dict['L'], params_dict['D'], params_dict['LR'], params_dict['BS']
    print(f"\n--- Final Test: H={hs}, L={nl}, D={dr:.3f}, LR={lr:.1e}, BS={bs} ---")

    full_train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    bs_final_train = min(bs, len(full_train_ds))
    bs_final_val = min(bs, len(val_ds))
    bs_final_test = min(bs, len(test_ds))

    if bs_final_train == 0 or bs_final_val == 0 or bs_final_test == 0:
         print("  -> ERROR: Effective batch size is zero for final test. Skipping.")
         return None

    full_train_loader = DataLoader(full_train_ds, batch_size=bs_final_train, shuffle=True, drop_last=True)
    val_loader_final = DataLoader(val_ds, batch_size=bs_final_val, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs_final_test, shuffle=False)

    if len(full_train_loader) == 0 or len(val_loader_final) == 0 or len(test_loader) == 0:
        print("  -> ERROR: Empty DataLoader for final test. Skipping.")
        return None

    final_model = LSTMModel(input_size=1, hidden_size=hs, num_layers=nl, dropout=dr)
    param_count = count_parameters(final_model)

    start_time_final = time.time()
    # Use train_model_final and get history
    final_model_trained, train_hist, val_hist = train_model_final( # MODIFIED
        final_model, full_train_loader, val_loader_final,
        epochs=100, lr=lr, device=device, early_stop_patience=10
    )
    final_train_time = time.time() - start_time_final

    final_model_trained.eval()
    test_preds_scaled, test_trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = final_model_trained(xb.to(device))
            test_preds_scaled.append(preds.cpu().numpy())
            test_trues_scaled.append(yb.cpu().numpy())

    if not test_preds_scaled or not test_trues_scaled:
         return None

    test_preds_scaled = np.concatenate(test_preds_scaled).flatten()
    test_trues_scaled = np.concatenate(test_trues_scaled).flatten()
    test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    test_trues = scaler.inverse_transform(test_trues_scaled.reshape(-1, 1)).flatten()

    final_metrics = calculate_metrics(test_trues, test_preds)

    results = {
        'H': hs, 'L': nl, 'D': dr, 'LR': lr, 'BS': bs, 'Params': param_count,
        'Time': final_train_time, **final_metrics,
        'y_pred': test_preds, 'y_true': test_trues, # Store predictions
        'train_hist': train_hist, 'val_hist': val_hist # Store history
    }
    return results

# --- 7. Metrics Calculation & Plotting ---
def calculate_metrics(y_true, y_pred):
    """Calculates standard regression metrics."""
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if not np.all(np.isfinite(y_pred)):
        metrics['rmse'] = np.inf; metrics['mse'] = np.inf; metrics['mae'] = np.inf
        metrics['r2'] = np.nan; metrics['mape'] = np.inf
        return metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    try:
        metrics['r2'] = r2_score(y_true, y_pred)
    except ValueError: # Handle cases where r2_score might fail (e.g., constant true values)
        metrics['r2'] = np.nan
    mask = y_true != 0
    if np.sum(mask) == 0: metrics['mape'] = np.inf
    else: metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return metrics

def plot_results(pareto_solutions, convergence, best_model_train_hist, best_model_val_hist, base_save_dir, exp_name):
    """Generates and saves Pareto, convergence, and learning curve plots."""
    save_dir = os.path.join(base_save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    # Pareto Front Plot
    if pareto_solutions:
        scores = np.array([s.score for s in pareto_solutions if np.all(np.isfinite(s.score))])
        if scores.shape[0] > 0 and scores.shape[1] == 3:
            fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
            finite_mask_plot = np.all(np.isfinite(scores), axis=1)
            if np.any(finite_mask_plot):
                 ax.scatter(scores[finite_mask_plot, 0], scores[finite_mask_plot, 1], scores[finite_mask_plot, 2], c='b', marker='o', label='Pareto Optimal (Search Phase)')
                 ax.set_xlabel('Validation RMSE'); ax.set_ylabel('Search Training Time (s)'); ax.set_zlabel('Parameter Count')
                 ax.set_title(f'Pareto Front (from Search) for {exp_name}'); ax.legend()
                 plt.savefig(os.path.join(save_dir, 'pareto_front_search.png'))
            plt.close(fig)
    # Convergence Plot
    if convergence:
        finite_convergence = [c for c in convergence if np.isfinite(c)]
        if finite_convergence:
             plt.figure(figsize=(10, 6)); plt.plot(finite_convergence, marker='o'); plt.xlabel('Iteration / Generation')
             plt.ylabel('Best Validation RMSE in Population'); plt.title(f'Convergence for {exp_name}'); plt.grid(True)
             plt.savefig(os.path.join(save_dir, 'convergence.png')); plt.close()
    # Learning Curve for the *best* model (retrained)
    if best_model_train_hist and best_model_val_hist:
        plt.figure(figsize=(10, 6)); plt.plot(best_model_train_hist, label='Training Loss (MSE)'); plt.plot(best_model_val_hist, label='Validation Loss (MSE)')
        plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title(f'Learning Curve for Best Retrained Model - {exp_name}'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)
        plt.savefig(os.path.join(save_dir, 'learning_curve_best_model.png')); plt.close()

def plot_actual_vs_predicted(test_trues, test_preds, freq_label, save_dir, model_id="best"):
    """Generates and saves plot comparing actual vs predicted values."""
    plt.figure(figsize=(12, 6)); plt.plot(test_trues, label="Actual", alpha=0.7); plt.plot(test_preds, label="Predicted", alpha=0.7)
    plt.title(f"Actual vs Predicted ({model_id} model) - {freq_label}"); plt.xlabel("Time Step (Test Set)"); plt.ylabel("Global Active Power (kW)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'actual_vs_predicted_{freq_label.lower()}_{model_id}.png')); plt.close()

# --- 8. Final Training Function (Returns History) ---
def train_model_final(model, train_loader, val_loader, epochs, lr, device, early_stop_patience=10):
    """Trains the final model, returns the trained model and history."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_rmse = np.inf
    epochs_no_improve = 0
    best_model_state = model.state_dict()
    train_loss_history, val_loss_history = [], []

    print(f"  Starting final training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); preds = model(xb); loss = criterion(preds, yb); loss.backward(); optimizer.step()
            epoch_train_losses.append(loss.item())
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_preds, val_trues, epoch_val_losses = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device); preds = model(xb); loss = criterion(preds, yb)
                epoch_val_losses.append(loss.item()); val_preds.append(preds.cpu().numpy()); val_trues.append(yb.cpu().numpy())
        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else np.inf
        val_loss_history.append(avg_val_loss)

        if val_preds:
            val_preds = np.concatenate(val_preds).flatten(); val_trues = np.concatenate(val_trues).flatten()
            if np.any(~np.isfinite(val_preds)): current_val_rmse = np.inf; break
            else: current_val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss (MSE): {avg_val_loss:.6f}, Val RMSE: {current_val_rmse:.6f}")
            if current_val_rmse < best_val_rmse:
                best_val_rmse = current_val_rmse; epochs_no_improve = 0; best_model_state = model.state_dict()
                # print(f"    -> New best validation RMSE: {best_val_rmse:.6f}") # Optional: more verbose
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience: print(f"  -> Early stopping triggered at epoch {epoch+1} with best Val RMSE: {best_val_rmse:.6f}"); break
        else: print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, No validation data.")

    print(f"  Finished final training. Best validation RMSE achieved: {best_val_rmse:.6f}")
    model.load_state_dict(best_model_state)
    return model, train_loss_history, val_loss_history # Return history

# --- 9. Main Experiment Runner ---
if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Define frequencies and corresponding time steps
    FREQ_CONFIG = {
        'Minutely': {'code': 'T', 'step': 60}, # 'T' or 'min' for minutely
        'Hourly': {'code': 'H', 'step': 60},   # 'H' or 'h' for hourly
        'Daily': {'code': 'D', 'step': 30},
        'Weekly': {'code': 'W', 'step': 8}    # 'W' for weekly
    }

    # MOEA/D Hyperparameters
    POPULATION_SIZE = 5
    MAX_ITERATIONS = 4 # Total evaluations = POP_SIZE + POP_SIZE * MAX_ITERATIONS
    NEIGHBORHOOD_SIZE = 3 # T parameter

    BOUNDS = np.array([
        [0, 1], # Hidden Size (normalized)
        [0, 1], # Num Layers (normalized)
        [0, 1], # Dropout (normalized)
        [0, 1], # Learning Rate (normalized)
        [0, 1]  # Batch Size (normalized)
    ])

    all_results_tables = {} # Store tables for each frequency

    for freq_name, config in FREQ_CONFIG.items():
        freq_code = config['code']
        time_step_for_run = config['step']
        exp_name = f"UCI_MOEAD_{freq_name}"
        print(f"\n{'='*30} STARTING EXPERIMENT: {exp_name} {'='*30}")

        # Get datasets for the current frequency
        datasets = get_datasets(freq_code, time_step=time_step_for_run)
        if datasets is None:
            print(f"!!! Failed to load data for {freq_name}. Skipping. !!!")
            continue
        train_ds, val_ds, test_ds, scaler = datasets

        # Define the problem function for MOEA/D for this frequency
        problem_func = lambda pos: evaluate_hyperparams(pos, train_ds, val_ds, DEVICE)

        # Initialize and run MOEA/D
        moead = MOEAD(problem_func, BOUNDS,
                      population_size=POPULATION_SIZE,
                      max_iter=MAX_ITERATIONS,
                      n_objs=3, # RMSE, Time, Params
                      T=NEIGHBORHOOD_SIZE)
        pareto_solutions, convergence_history = moead.optimize()

        # --- Post-Optimization Analysis ---
        final_results_list = []
        best_rmse_solution_details = None # To store details of the best RMSE model for plots
        min_rmse_overall = np.inf
        baseline_results = None # To store baseline metrics
        baseline_results_raw = None # To store baseline raw results including predictions

        if not pareto_solutions:
            print(f"!!! MOEA/D found no non-dominated solutions for {exp_name}. Skipping final tests. !!!")
        else:
            print(f"\n--- Found {len(pareto_solutions)} Pareto Optimal Solution(s) for {exp_name} ---")
            print("--- Running Final Tests on Pareto Solutions ---")

            for i, sol in enumerate(pareto_solutions):
                print(f"\nTesting Solution {i+1}/{len(pareto_solutions)}")
                hs, nl, dr, lr, bs = decode_position(sol.position)
                # Use param count from search result if available and finite, otherwise recalculate
                param_count_search = int(sol.score[2]) if len(sol.score) > 2 and np.isfinite(sol.score[2]) else count_parameters(LSTMModel(1, hs, nl, dr))

                params_dict = {'H': hs, 'L': nl, 'D': dr, 'LR': lr, 'BS': bs, 'Params': param_count_search}

                solution_results = run_final_test(params_dict, train_ds, val_ds, test_ds, scaler, DEVICE)

                if solution_results:
                    final_results_list.append(solution_results)
                    # Check if this solution is the new best based on RMSE
                    if np.isfinite(solution_results['rmse']) and solution_results['rmse'] < min_rmse_overall:
                        min_rmse_overall = solution_results['rmse']
                        best_rmse_solution_details = solution_results # Keep track of the best one


            # --- Run Baseline Model ---
            print(f"\n--- Training baseline model for {exp_name} ---")
            baseline_params = {'H': 64, 'L': 2, 'D': 0.1, 'LR': 1e-4, 'BS': 128, 'Params': count_parameters(LSTMModel(1, 64, 2, 0.1))}
            # Capture the raw results including predictions for Wilcoxon
            baseline_results_raw = run_final_test(baseline_params, train_ds, val_ds, test_ds, scaler, DEVICE)
            if baseline_results_raw:
                 # Store metrics separately for easy access
                 baseline_results = {k: v for k, v in baseline_results_raw.items() if k not in ['y_pred', 'y_true', 'train_hist', 'val_hist']}

            # --- Generate and Print Table ---
            if final_results_list:
                final_results_list.sort(key=lambda x: x['rmse'])
                print("\n" + "="*125)
                print(f"                                   Final Metrics for Pareto Front Models ({exp_name})")
                print("="*125)
                # Adjusted header and row format widths slightly
                header    = "| {:<7} | {:<4} | {:<1} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<8} |"
                separator = "|{:-<9}|{:-<6}|{:-<3}|{:-<9}|{:-<9}|{:-<9}|{:-<9}|{:-<9}|{:-<9}|{:-<9}|{:-<9}|{:-<1A0}|"
                row_format = "| {:<7} | {:<4} | {:<1} | {:<7.3f} | {:<7.1e} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<8} |"

                print(header.format("Params", "H", "L", "Dropout", "LR", "BS", "RMSE", "MSE", "MAE", "R2", "MAPE", "Time"))
                print(separator)

                table_content = ""
                for res in final_results_list:
                     # Check for non-finite values before printing
                     rmse_str = f"{res['rmse']:.4f}" if np.isfinite(res['rmse']) else "inf"
                     mse_str = f"{res['mse']:.4f}" if np.isfinite(res['mse']) else "inf"
                     mae_str = f"{res['mae']:.4f}" if np.isfinite(res['mae']) else "inf"
                     r2_str = f"{res['r2']:.4f}" if np.isfinite(res['r2']) else "nan"
                     mape_str = f"{res['mape']:.2f}%" if np.isfinite(res['mape']) else "inf%"
                     time_str = f"{res['Time']:.2f}s" if np.isfinite(res['Time']) else "inf s"

                     # Use the formatted strings in the row
                     line = row_format.format(res['Params'], res['H'], res['L'], res['D'], res['LR'], res['BS'],
                                             rmse_str, mse_str, mae_str, r2_str, mape_str, time_str)
                     print(line)
                     table_content += line + "\n"
                all_results_tables[exp_name] = table_content
                print("="*125)

                # --- Print Baseline and Wilcoxon ---
                if baseline_results and best_rmse_solution_details:
                     print("\n--- Baseline Model Results ---")
                     # Check for inf/nan before printing baseline results

                     # --- CORRECTED CODE ---
                    # This version moves the if/else logic inside the main expression
                     br = baseline_results
                     print(f"  RMSE: {f'{br['rmse']:.4f}' if np.isfinite(br['rmse']) else 'inf'}, "
                           f"MSE: {f'{br['mse']:.4f}' if np.isfinite(br['mse']) else 'inf'}, "
                           f"MAE: {f'{br['mae']:.4f}' if np.isfinite(br['mae']) else 'inf'}, "
                           f"R2: {f'{br['r2']:.4f}' if np.isfinite(br['r2']) else 'nan'}, "
                           f"MAPE: {f'{br['mape']:.2f}%' if np.isfinite(br['mape']) else 'inf%'}")
                    
                     # Wilcoxon Test (Best Optimized vs Baseline)
                     try:
                         # Ensure predictions are valid arrays and exist
                         if 'y_true' in best_rmse_solution_details and 'y_pred' in best_rmse_solution_details and \
                            'y_true' in baseline_results_raw and 'y_pred' in baseline_results_raw:

                             best_y_true = np.array(best_rmse_solution_details['y_true'])
                             best_y_pred = np.array(best_rmse_solution_details['y_pred'])
                             baseline_y_true = np.array(baseline_results_raw['y_true'])
                             baseline_y_pred = np.array(baseline_results_raw['y_pred'])

                             # Check if predictions are finite
                             if np.all(np.isfinite(best_y_pred)) and np.all(np.isfinite(baseline_y_pred)):
                                 errors_optimized = np.abs(best_y_true - best_y_pred)
                                 errors_baseline = np.abs(baseline_y_true - baseline_y_pred)

                                 min_len = min(len(errors_optimized), len(errors_baseline))
                                 if min_len > 0:
                                      # Perform Wilcoxon test only if errors are different
                                      if not np.array_equal(errors_optimized[:min_len], errors_baseline[:min_len]):
                                           stat, p_value = wilcoxon(errors_optimized[:min_len], errors_baseline[:min_len])
                                           print(f"\nWilcoxon Test (Best Optimized vs Baseline) p-value: {p_value:.6f} {'(Significant)' if p_value < 0.05 else '(Not Significant)'}")
                                      else:
                                           print("\nWilcoxon Test skipped: Optimized and Baseline errors are identical.")
                                 else: print("\nWilcoxon Test skipped: Not enough data points.")
                             else: print("\nWilcoxon Test skipped: Non-finite predictions found.")
                         else: print("\nWilcoxon Test skipped: Prediction arrays not found.")

                     except Exception as e: print(f"\nCould not perform Wilcoxon test: {e}")

            else: print("\nNo valid results obtained from final testing.")

        # --- Generate Plots ---
        save_dir = os.path.join('results', exp_name)
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists

        # Plots for optimization process (using search results)
        plot_results(pareto_solutions, convergence_history,
                     best_rmse_solution_details.get('train_hist', []) if best_rmse_solution_details else [],
                     best_rmse_solution_details.get('val_hist', []) if best_rmse_solution_details else [],
                     'results', exp_name)

        # Plot Actual vs Predicted for best optimized and baseline (using final test results)
        if best_rmse_solution_details and 'y_true' in best_rmse_solution_details and 'y_pred' in best_rmse_solution_details:
             plot_actual_vs_predicted(best_rmse_solution_details['y_true'], best_rmse_solution_details['y_pred'],
                                      freq_name, save_dir, model_id="best_optimized")
        if baseline_results_raw and 'y_true' in baseline_results_raw and 'y_pred' in baseline_results_raw:
             plot_actual_vs_predicted(baseline_results_raw['y_true'], baseline_results_raw['y_pred'],
                                      freq_name, save_dir, model_id="baseline")


        print(f"\nPlots and results for {exp_name} saved to 'results/{exp_name}/'")
        print(f"{'='*30} FINISHED EXPERIMENT: {exp_name} {'='*30}\n")
