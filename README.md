# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

### Train Dataset
<img width="611" height="707" alt="image" src="https://github.com/user-attachments/assets/641feae0-d1fd-4e3a-9ba4-455afe657d9c" />

### Test Dataset
<img width="610" height="685" alt="image" src="https://github.com/user-attachments/assets/ecaa926d-60c7-4e1d-b629-7aed2ae835ca" />


## DESIGN STEPS
### STEP 1: 
Load and normalize data, create sequences.

### STEP 2: 
Convert data to tensors and set up DataLoader.


### STEP 3: 
Define the RNN model architecture


### STEP 4: 
Summarize, compile with loss and optimizer.


### STEP 5: 
Train the model with loss tracking.


### STEP 6: 


Predict on test data, plot actual vs. predicted prices.


## PROGRAM

### Name: Malar Mariam S

### Register Number: 212223230118

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv('/content/trainset.csv')
df_test = pd.read_csv('/content/testset.csv')

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    # write your code here
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo
from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model,train_loader,criterion,optimizer,epochs=20):
  train_losses=[]
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch,y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
      total_loss=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
  return train_losses

train_losses = train_model(model,train_loader,criterion,optimizer,epochs=20)

# Plot training loss
print('Name:  Malar Mariam S')
print('Register Number: 212223230118')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:  Malar Mariam S')
print('Register Number: 212223230118')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')

```

### OUTPUT


## Training Loss Over Epochs Plot

<img width="705" height="501" alt="image" src="https://github.com/user-attachments/assets/8dfe6033-829b-46e4-8323-9e73cb9680be" />
<img width="720" height="609" alt="image" src="https://github.com/user-attachments/assets/39c43e7e-06ef-4eab-894e-26427c6a9a1d" />


## True Stock Price, Predicted Stock Price vs time

<img width="744" height="536" alt="image" src="https://github.com/user-attachments/assets/0a0a1a8b-87fa-48e9-9427-c64744412d7b" />

### Predictions
<img width="292" height="60" alt="image" src="https://github.com/user-attachments/assets/065d497a-7131-46dc-978a-2dc0514992d4" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
