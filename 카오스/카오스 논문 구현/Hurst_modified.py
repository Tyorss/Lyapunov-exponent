import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import random
import seaborn as sns
import os
from scipy.signal import argrelextrema

# Set the visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create image directory if it doesn't exist
if not os.path.exists('image'):
    os.makedirs('image')

# Function to calculate Hurst Exponent using R/S Analysis
def calculate_hurst(time_series, fixed_lags=None):
    """
    Calculate the Hurst exponent using R/S analysis with fixed lags
    
    Parameters:
    time_series (array): Array of returns
    fixed_lags (list): List of fixed lag values to use
    
    Returns:
    hurst_exponent (float): Estimated Hurst exponent
    lags (array): Array of lags used
    rs_values (array): R/S values for each lag
    cycle (int): Memory cycle in years
    """
    # 데이터 표준화
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    if fixed_lags is None:
        # 2년부터 20년까지 1년 단위로 생성
        fixed_lags = [i for i in range(2, 21)]
    
    lags = []
    rs_values = []
    
    # N값 4년 이하와 초과를 구분하여 저장할 리스트
    rs_values_short = []
    rs_values_long = []
    lags_short = []
    lags_long = []
    
    print("\nR/S 분석 결과:")
    
    # 각 lag에 대한 R/S 값 계산
    for lag in fixed_lags:
        if lag >= len(time_series) // 2:  # 최대 lag를 데이터 길이의 절반으로 제한
            continue
            
        rs_values_for_lag = []
        
        # 오버랩되는 윈도우 사용
        for start in range(0, len(time_series) - lag):
            window = time_series[start:start + lag]
            
            # 누적 편차 계산
            mean = np.mean(window)
            cum_dev = np.cumsum(window - mean)
            
            # R 값 계산 (최대 - 최소)
            r_value = np.max(cum_dev) - np.min(cum_dev)
            
            # S 값 계산 (표준 편차)
            s_value = np.std(window, ddof=1)
            
            # 0 표준편차 처리
            if s_value < 1e-10:
                continue
                
            # R/S 값 계산
            rs_values_for_lag.append(r_value / s_value)
        
        if len(rs_values_for_lag) > 0:
            rs_mean = np.mean(rs_values_for_lag)
            
            # 유효한 결과만 저장
            lags.append(lag)
            rs_values.append(rs_mean)
            
            # N값에 따라 구분하여 저장
            if lag <= 4:
                rs_values_short.append(rs_mean)
                lags_short.append(lag)
            else:
                rs_values_long.append(rs_mean)
                lags_long.append(lag)
            
            print(f"N = {lag}년: R/S = {rs_mean:.3f}")
    
    if len(lags) < 4 or len(rs_values) < 4:
        return None, None, None, None
    
    # 로그-로그 변환
    lags_log = np.log10(lags)
    rs_values_log = np.log10(rs_values)
    
    # 가중치 적용 - 장기 lag의 영향력 조정
    weights = np.ones_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        if lag > 4:
            weights[i] = max(0.2, 4.0/lag)
    
    # 전체 데이터에 대한 허스트 지수 계산
    h_overall, _ = np.polyfit(lags_log, rs_values_log, 1, w=weights)
    
    # N값 4년 이하에 대한 허스트 지수
    h_short = None
    if len(lags_short) >= 4:
        lags_short_log = np.log10(lags_short)
        rs_short_log = np.log10(rs_values_short)
        h_short, _ = np.polyfit(lags_short_log, rs_short_log, 1)
        print(f"\nN ≤ 4년 허스트 지수: {h_short:.3f}")
    
    # N값 4년 초과에 대한 허스트 지수
    h_long = None
    if len(lags_long) >= 4:
        lags_long_log = np.log10(lags_long)
        rs_long_log = np.log10(rs_values_long)
        h_long, _ = np.polyfit(lags_long_log, rs_long_log, 1)
        print(f"N > 4년 허스트 지수: {h_long:.3f}")
        
        if h_short is not None and h_long > h_short:
            print(f"주의: N>4년에서 허스트 지수가 증가했습니다 ({h_short:.3f} → {h_long:.3f})")
    
    # 메모리 사이클 결정
    cycle = determine_cycle_breakpoint(lags, rs_values)
    
    # N≤4년 범위의 H값을 최종 결과로 사용
    final_h = h_short if h_short is not None else h_overall
    
    print(f"\n최종 허스트 지수 (H): {final_h:.3f}")
    if 0.5 < final_h < 1.0:
        print(f"해석: 시계열은 지속성(persistence)을 보이며, 현재 추세가 미래에도 지속될 가능성이 높습니다.")
    elif 0.0 < final_h < 0.5:
        print(f"해석: 시계열은 반지속성(anti-persistence)을 보이며, 평균 회귀 경향이 있습니다.")
    else:
        print(f"해석: 시계열은 랜덤워크(random walk)에 가까운 특성을 보입니다.")
    
    return final_h, lags, rs_values, cycle

def determine_cycle_breakpoint(lags, rs_values):
    """
    기울기 변화가 크게 나타나는 지점(breakpoint)을 찾아 메모리 사이클로 결정
    """
    if len(lags) < 5:
        return None
    
    # 로그 변환
    lags_log = np.log10(lags)
    rs_values_log = np.log10(rs_values)
    
    # 로컬 기울기(local slopes) 계산
    slopes = []
    for i in range(2, len(lags_log)-2):
        # 5점 이동 기울기 (더 안정적)
        x_local = lags_log[i-2:i+3]
        y_local = rs_values_log[i-2:i+3]
        slope, _ = np.polyfit(x_local, y_local, 1)
        slopes.append((lags[i], slope))
    
    if len(slopes) < 3:
        return 4  # 기본값
    
    # 기울기 변화율 계산
    slope_changes = []
    for i in range(1, len(slopes)-1):
        prev_slope = slopes[i-1][1]
        curr_slope = slopes[i][1]
        next_slope = slopes[i+1][1]
        
        # 기울기 변화의 2차 도함수 개념 적용
        change = abs(next_slope - 2*curr_slope + prev_slope)
        slope_changes.append((slopes[i][0], change))
    
    # 기울기 변화가 가장 큰 지점 찾기
    max_change_idx = np.argmax([change for _, change in slope_changes])
    breakpoint = slope_changes[max_change_idx][0]
    
    # 변화점이 4년 주변이면 4년으로 설정
    if 3 <= breakpoint <= 5:
        return 4
    
    # 변화점 설명 출력
    print(f"\n메모리 사이클 변화점(breakpoint): {breakpoint}년")
    
    return breakpoint

# Function to scramble a time series
def scramble_series(time_series):
    """
    Scramble a time series to destroy any memory effect
    
    Parameters:
    time_series (array): Original time series
    
    Returns:
    scrambled_series (array): Scrambled time series
    """
    scrambled_series = time_series.copy()
    # 여러 번 섞어서 더 랜덤하게 만듦
    for _ in range(10):
        np.random.shuffle(scrambled_series)
    return scrambled_series

# Function to analyze a single stock
def analyze_stock(ticker, start_date, end_date, data_frequency='yearly'):
    """
    Download data, calculate returns, and compute Hurst exponent
    
    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date for data
    end_date (str): End date for data
    data_frequency (str): Data frequency ('daily', 'weekly', 'monthly', or 'yearly')
    
    Returns:
    results (dict): Dictionary containing analysis results
    """
    print(f"\n분석 시작: {ticker}")
    
    # Convert frequency to yfinance format
    yf_frequency = '1d'
    if data_frequency == 'weekly':
        yf_frequency = '1wk'
    elif data_frequency == 'monthly':
        yf_frequency = '1mo'
    elif data_frequency == 'yearly':
        yf_frequency = '1y'
    
    try:
        # Ticker 객체 생성
        ticker_obj = yf.Ticker(ticker)
        
        # 기본 정보 확인
        info = ticker_obj.info
        if not info:
            print(f"{ticker}: 종목 정보를 가져올 수 없습니다.")
            return None
            
        print(f"종목명: {info.get('longName', ticker)}")
        
        # 데이터 다운로드
        data = ticker_obj.history(period="max", interval=yf_frequency)
        
        # 지정된 기간으로 필터링
        data = data[start_date:end_date]
        
        if len(data) == 0:
            print(f"{ticker}: 해당 기간에 데이터가 없습니다.")
            return None
            
        # 데이터 확인
        print(f"다운로드된 데이터: {len(data)} 포인트")
        print(f"기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
            
        # Verify we have enough data points
        if len(data) < 20:  # 최소 20년 데이터 필요
            print(f"{ticker}: 데이터 포인트가 부족합니다 (필요: 20년, 현재: {len(data)}년)")
            return None
        
        # Calculate log returns
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()
        
        if len(data) < 20:
            print(f"{ticker}: 수익률 계산 후 데이터 포인트가 부족합니다")
            return None
            
        log_returns = data['Log_Return'].values
        
        # Create scrambled version of the returns
        scrambled_returns = scramble_series(log_returns)
        
        # Calculate Hurst exponent for original series
        h_original, lags_original, rs_values_original, cycle_original = calculate_hurst(log_returns)
        
        if h_original is None:
            print(f"{ticker}: 허스트 지수를 계산할 수 없습니다")
            return None
            
        # Calculate Hurst exponent for scrambled series
        h_scrambled, lags_scrambled, rs_values_scrambled, cycle_scrambled = calculate_hurst(scrambled_returns)
        
        # 메모리 주기 (이미 년 단위)
        memory_period = cycle_original
        
        # 프랙탈 차원 (D = 2 - H)
        fractal_dimension = 2 - h_original
        
        # 논문과 같은 형식으로 결과 출력
        print(f"\n{ticker} Unscramble 결과:")
        print(f"허스트 지수: {h_original:.3f}")
        if memory_period:
            print(f"메모리 주기: {memory_period:.1f}년")
        else:
            print("메모리 주기: Unknown")
        print(f"프랙탈 차원: {fractal_dimension:.3f}")
        
        print(f"\n{ticker} Scramble 결과:")
        print(f"허스트 지수: {h_scrambled:.3f}")
        
        # 추세 지속 확률 계산 (H > 0.5인 경우만)
        if h_original > 0.5:
            trend_probability = (h_original - 0.5) * 2 * 100  # 0.5~1.0 범위를 0~100%로 변환
            print(f"\n추세 지속 확률: {trend_probability:.1f}%")
        else:
            print("\n추세 지속 확률: 0%")
        
        results = {
            'ticker': ticker,
            'data': data,
            'log_returns': log_returns,
            'scrambled_returns': scrambled_returns,
            'h_original': h_original,
            'lags_original': lags_original,
            'rs_values_original': rs_values_original,
            'h_scrambled': h_scrambled,
            'lags_scrambled': lags_scrambled,
            'rs_values_scrambled': rs_values_scrambled,
            'cycle': cycle_original,
            'memory_period': memory_period,
            'fractal_dimension': fractal_dimension,
            'frequency': data_frequency
        }
        
        return results
        
    except Exception as e:
        print(f"{ticker} 분석 중 오류 발생: {str(e)}")
        return None

# Function to plot Hurst exponent analysis
def plot_hurst_analysis(results, save_path=None):
    """
    Plot Hurst exponent analysis results
    
    Parameters:
    results (dict): Dictionary containing analysis results
    save_path (str): Path to save the figure
    """
    ticker = results['ticker']
    h_original = results['h_original']
    h_scrambled = results['h_scrambled']
    lags_original = results['lags_original']
    rs_values_original = results['rs_values_original']
    lags_scrambled = results['lags_scrambled']
    rs_values_scrambled = results['rs_values_scrambled']
    cycle = results['cycle']
    frequency = results['frequency']
    
    # Log-log plot
    plt.figure(figsize=(14, 10))
    
    # Original series
    plt.loglog(lags_original, rs_values_original, 'o-', label=f'Original (H={h_original:.3f})', color='blue')
    
    # Scrambled series
    plt.loglog(lags_scrambled, rs_values_scrambled, 's--', label=f'Scrambled (H={h_scrambled:.3f})', color='red')
    
    # Mark the cycle
    if cycle:
        plt.axvline(x=cycle, color='green', linestyle='--', label=f'Memory Cycle ({cycle} years)')
    
    # Mark N=4 (논문에서의 기준점)
    plt.axvline(x=4, color='purple', linestyle=':', label='N=4 years')
    
    # Fit lines separately for N<=4 and N>4
    lags_short = [lag for lag in lags_original if lag <= 4]
    rs_short = [rs_values_original[lags_original.index(lag)] for lag in lags_short]
    
    lags_long = [lag for lag in lags_original if lag > 4]
    rs_long = [rs_values_original[lags_original.index(lag)] for lag in lags_long]
    
    if lags_short and rs_short:
        h_short, _ = np.polyfit(np.log10(lags_short), np.log10(rs_short), 1)
        x_short = np.logspace(np.log10(min(lags_short)), np.log10(max(lags_short)), 100)
        y_short = np.power(x_short, h_short) * (rs_short[0] / np.power(lags_short[0], h_short))
        plt.loglog(x_short, y_short, '-', label=f'H(N≤4)={h_short:.3f}', color='blue', alpha=0.5)
    
    if lags_long and rs_long:
        h_long, _ = np.polyfit(np.log10(lags_long), np.log10(rs_long), 1)
        x_long = np.logspace(np.log10(min(lags_long)), np.log10(max(lags_long)), 100)
        y_long = np.power(x_long, h_long) * (rs_long[0] / np.power(lags_long[0], h_long))
        plt.loglog(x_long, y_long, '-', label=f'H(N>4)={h_long:.3f}', color='green', alpha=0.5)
    
    plt.title(f'R/S Analysis for {ticker}')
    plt.xlabel('Time Lag (N)')
    plt.ylabel('R/S Value')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프가 저장되었습니다: {save_path}")
    
    plt.close()

def explain_hurst_exponent():
    """
    Print an explanation of the Hurst exponent
    """
    explanation = """
허스트 지수 (Hurst Exponent) 설명:

허스트 지수는 시계열 데이터의 장기 기억 특성을 측정하는 통계적 지표입니다.
이 지수는 영국의 수문학자 해롤드 에드윈 허스트(Harold Edwin Hurst)가 나일강의
수위 변화를 연구하던 중 발견했습니다.

허스트 지수(H)의 범위와 해석:
1. H = 0.5: 완전한 랜덤워크(브라운 운동), 과거와 미래 사이에 상관관계가 없음
2. 0 < H < 0.5: 반지속성(Anti-persistence) 또는 평균 회귀 특성, 상승 이후 하락 가능성이 높음   
3. 0.5 < H < 1: 지속성(Persistence), 현재 추세가 미래에도 지속될 가능성이 높음

R/S 분석 (Rescaled Range Analysis):
허스트 지수를 계산하는 가장 일반적인 방법은 R/S 분석입니다. 이 방법은 다양한
시간 간격에서 시계열의 '범위(Range)'를 '표준편차(Standard deviation)'로 나눈 값의
로그-로그 그래프의 기울기를 계산합니다.

금융시장에서의 의미:
- H가 0.5에 가까우면: 효율적 시장 가설에 부합, 가격 변동이 예측 불가능
- H가 0.5보다 크면: 추세추종 전략이 효과적일 수 있음
- H가 0.5보다 작으면: 평균회귀 전략이 효과적일 수 있음

프랙탈 차원 (Fractal Dimension):
허스트 지수는 프랙탈 차원(D)과 다음 관계가 있습니다: D = 2 - H
프랙탈 차원이 클수록 시계열의 불규칙성이 높아집니다.
"""
    print(explanation)

def main():
    # Explain Hurst exponent
    explain_hurst_exponent()
    
    # Set the list of tickers for analysis
    tickers = ['SPY', 'QQQ', 'BRK-A', 'BA', 'GE', 'F', 'MRK', 'PFE', 'AAPL', 'IBM', 'AXP', 'GS', 'CAT']
    
    # Analysis parameters
    start_date = '1980-01-01'
    end_date = '2025-03-01'
    data_frequency = 'monthly'  # 월별 데이터 사용
    
    # Analyze each ticker
    for ticker in tickers:
        print(f"\n==================================================")
        print(f"{ticker} 분석 시작")
        print(f"==================================================\n")
        
        results = analyze_stock(ticker, start_date, end_date, data_frequency)
        
        if results:
            # Save visualization
            plot_path = f"image/{ticker}_hurst_analysis.png"
            plot_hurst_analysis(results, save_path=plot_path)
            
            print(f"\n==================================================\n")

if __name__ == "__main__":
    main() 