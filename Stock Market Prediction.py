import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Download Data
data = yf.download("AAPL", start="2015-01-01", end="2023-12-31")

# 2. Manual Technical Indicators Implementation
df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Simple Moving Average
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# Exponential Moving Average
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Relative Strength Index
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

# Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = calculate_sma(data, window)
    rolling_std = data.rolling(window=window).std()
    upper = sma + (rolling_std * num_std)
    lower = sma - (rolling_std * num_std)
    return upper, sma, lower

# Average True Range
def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

# On-Balance Volume
def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

# Apply all indicators
df['SMA_20'] = calculate_sma(df['Close'], 20)
df['EMA_20'] = calculate_ema(df['Close'], 20)
df['RSI_14'] = calculate_rsi(df['Close'], 14)
df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
df['OBV'] = calculate_obv(df['Close'], df['Volume'])
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Daily_Return'].rolling(20).std()

# Lag Features
for lag in [1, 2, 3, 5, 10]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

# Rolling Statistics
for window in [5, 10, 20, 50]:
    df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window).mean()
    df[f'Rolling_Std_{window}'] = df['Close'].rolling(window).std()

# Drop NA values
df = df.dropna()

# 3. Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# 4. Create Sequences for LSTM
def create_sequences(data, seq_length=60, target_col=3):  # Target is 'Close' price
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_col])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 5. PyTorch Dataset
class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 6. LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Initialize model
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 7. Training Setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# 8. Training Loop
num_epochs = 100
best_loss = float('inf')
patience = 15
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * sequences.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * sequences.size(0)
    
    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# 9. Evaluation
def evaluate_model(model, data_loader):
    model.eval()
    actuals = []
    predictions = []
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            outputs = model(sequences).cpu().numpy().flatten()
            targets = targets.numpy().flatten()
            
            predictions.extend(outputs)
            actuals.extend(targets)
    
    return np.array(actuals), np.array(predictions)

train_actual, train_pred = evaluate_model(model, train_loader)
test_actual, test_pred = evaluate_model(model, test_loader)

# Inverse scaling
def inverse_transform_predictions(pred, actual, scaler, n_features):
    dummy = np.zeros((len(pred), n_features))
    dummy[:, 0] = pred
    pred_inv = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = actual
    actual_inv = scaler.inverse_transform(dummy)[:, 0]
    
    return actual_inv, pred_inv

train_actual_inv, train_pred_inv = inverse_transform_predictions(train_pred, train_actual, scaler, scaled_data.shape[1])
test_actual_inv, test_pred_inv = inverse_transform_predictions(test_pred, test_actual, scaler, scaled_data.shape[1])

# Calculate metrics
def print_metrics(actual, predicted, label):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"\n{label} Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

print_metrics(train_actual_inv, train_pred_inv, "Training")
print_metrics(test_actual_inv, test_pred_inv, "Testing")

# 10. Future Prediction
def predict_next_day(model, data, scaler, seq_length=60):
    model.eval()
    last_sequence = data[-seq_length:]
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        prediction = model(sequence_tensor).cpu().numpy()[0][0]
    
    # Create a dummy row for inverse transform
    dummy_row = np.zeros((1, data.shape[1]))
    dummy_row[0, 0] = prediction
    predicted_price = scaler.inverse_transform(dummy_row)[0, 0]
    return predicted_price

next_day_price = predict_next_day(model, scaled_data, scaler)
print(f"\nPredicted Next Day Closing Price: ${next_day_price:.2f}")

# 11. Visualization
plt.figure(figsize=(15, 6))
plt.plot(test_actual_inv, label='Actual Prices')
plt.plot(test_pred_inv, label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()