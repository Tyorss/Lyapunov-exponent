import yfinance as yf

# SPY ETF 데이터 가져오기
spy = yf.Ticker("SPY")

# 기본 정보 출력
print("SPY ETF 정보:")
print(f"현재가: ${spy.info['regularMarketPrice']}")
print(f"52주 최고가: ${spy.info['fiftyTwoWeekHigh']}")
print(f"52주 최저가: ${spy.info['fiftyTwoWeekLow']}")
print(f"거래량: {spy.info['regularMarketVolume']}")

# 최근 5일간의 주가 데이터 가져오기
hist = spy.history(period="5d")
print("\n최근 5일간의 주가 데이터:")
print(hist) 