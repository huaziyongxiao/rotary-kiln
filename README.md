# rotary-kiln
Adaptive Rotary Kiln Prediction
# Main Function
def main():
    # Data Path
    data_file_path = 'zinc_rotary_kiln_temperature.csv'
    
    #  Load and Preprocess Data
    temperature_data = load_data(data_file_path)
    X, y, scaler = preprocess_data(temperature_data)
    
    #  Split the Training and Testing Sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train the MW-PLS Model
    model = train_model(X_train, y_train)
    
    #  Predict the Temperature of the Test Set
    y_pred = predict_temperature(model, X_test, scaler)
    
    #  Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
if __name__ == "__main__":
    main()
