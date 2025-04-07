import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# 데이터 불러오기 함수
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

# 로그 수익률 계산 함수
def calculate_log_returns(data):
    return np.log(data / data.shift(1)).dropna()

# R/S 계산 함수
def calculate_RS(data, n):
    N = len(data)
    M = N // n
    if M < 2:
        return np.nan
    RS = []
    for m in range(M):
        sub_data = data[m*n:(m+1)*n]
        mean = np.mean(sub_data)
        Y = np.cumsum(sub_data - mean)
        R = np.max(Y) - np.min(Y)
        S = np.std(sub_data)
        if S == 0:
            continue
        RS.append(R / S)
    return np.mean(RS)

# 허스트 지수 추정 함수
def estimate_hurst_exponent(data, min_n=10, max_n=None):
    if max_n is None:
        max_n = len(data) // 4
    n_values = np.logspace(np.log10(min_n), np.log10(max_n), num=20, dtype=int)
    n_values = np.unique(n_values_UNCONNECTED

    RS_values = []
    for n in n_values:
        RS = calculate_RS(data, n)
        if not np.isnan(RS):
            RS_values.append(RS)
    if len(RS_values) < 2:
        return np.nan, [], []
    log_n = np.log10(n_values[:len(RS_values)])
    log_RS = np.log10(RS_values)
    coeffs = np.polyfit(log_n, log_RS, 1)
    H = coeffs[0]
    return H, log_n, log_RS

# 데이터 Scrambled 함수
def scramble_data(data):
    scrambled = data.copy()
    np.random.shuffle(scrambled)
    return scrambled

# 허스트 지수 그래프 출력 함수
def plot_hurst_exponent(ticker, data, scrambled_data):
    H_unscrambled, log_n_unscrambled, log_RS_unscrambled = estimate_hurst_exponent(data)
    H_scrambled, log_n_scrambled, log_RS_scrambled = estimate_hurst_exponent(scrambled_data)
    
    if np.isnan(H_unscrambled) or np.isnan(H_scrambled):
        print(f"{ticker}에 대한 허스트 지수를 추정할 데이터 포인트가 부족합니다.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(log_n_unscrambled, log_RS_unscrambled, label=f'Unscrambled (H={H_unscrambled:.2f})')
    plt.plot(log_n_scrambled, log_RS_scrambled, label=f'Scrambled (H={H_scrambled:.2f})')
    plt.xlabel('log(n)')
    plt.ylabel('log(R/S)')
    plt.title(f'{ticker}에 대한 허스트 지수')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 결과 해석 출력
    print(f"\n{ticker} 결과 해석:")
    print(f"- Unscrambled H: {H_unscrambled:.2f}")
    if H_unscrambled > 0.55:
        print("  -> 지속성이 존재하며, 트렌드가 강화될 가능성이 높습니다.")
    elif H_unscrambled < 0.45:
        print("  -> 반지속성이 존재하며, 평균으로 회귀하는 경향이 있습니다.")
    else:
        print("  -> 무작위 보행에 가까우며, 장기 의존성이 약합니다.")
    print(f"- Scrambled H: {H_scrambled:.2f} (무작위로 섞인 데이터로, H ≈ 0.5에 가까움은 정상)")

# 메인 실행 코드
tickers = ['SPY', 'QQQ', 'BRK-A', 'BA', 'GE', 'F', 'MRK', 'PFE', 'AAPL', 'IBM', 'AXP', 'GS', 'CAT']
end_date = datetime.today()
start_date = end_date - timedelta(days=40*365)  # 40년 이상 데이터

for ticker in tickers:
    try:
        print(f"\n{ticker} 분석 시작...")
        prices = get_data(ticker, start_date, end_date)
        if len(prices) < 1000:  # 충분한 데이터 포인트 확보
            print(f"{ticker}에 대한 데이터가 충분하지 않습니다. (데이터 포인트: {len(prices)})")
            continue
        log_returns = calculate_log_returns(prices)
        scrambled_log_returns = scramble_data(log_returns.values)
        plot_hurst_exponent(ticker, log_returns.values, scrambled_log_returns)
    except Exception as e:
        print(f"{ticker} 처리 중 오류 발생: {e}")