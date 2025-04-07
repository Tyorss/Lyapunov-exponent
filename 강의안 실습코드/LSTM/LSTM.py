import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib
matplotlib.use('Agg')  # SSH나 원격 환경을 위한 백엔드 설정
import matplotlib.pyplot as plt
import os

# 데이터 다운로드 (삼성전자 주가)
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock['Close'].values.reshape(-1, 1)

# 데이터 전처리
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    
    return np.array(X), np.array(y), scaler

# LSTM 모델 생성
def create_model(look_back):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # 현재 작업 디렉토리 출력
    current_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"현재 작업 디렉토리: {current_dir}")

    # 데이터 다운로드
    ticker = "005930.KS"  # 삼성전자
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    data = get_stock_data(ticker, start_date, end_date)
    
    # 데이터 전처리
    look_back = 60
    X, y, scaler = prepare_data(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 학습/테스트 데이터 분할
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 모델 학습
    model = create_model(look_back)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    # 예측
    predictions = model.predict(X_test)
    
    # 예측 결과 역변환
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform([y_test])
    
    # 결과 시각화
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.T, label='Real Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Stock Price Prediction Results')
    plt.xlabel('Time')
    plt.ylabel('Price (KRW)')
    plt.legend()
    
    # 그래프를 파일로 저장
    save_path = os.path.join(current_dir, 'stock_prediction.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"그래프가 다음 경로에 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()
