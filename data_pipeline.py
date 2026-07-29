import yfinance as yf
import pandas as pd

def fetch_data(ticker, start="2015-01-01", end=None):
    
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Close"]].rename(columns={"Close": ticker})
    return df

def build_dataset():
    tickers = {
        "NSEI": "^NSEI",
        "BSESN": "^BSESN",
        "INDIAVIX": "^INDIAVIX"
    }

    dfs = []
    for name, symbol in tickers.items():
        print(f"Downloading {name} ({symbol})...")
        df = fetch_data(symbol)
        df.columns = [name]  
        dfs.append(df)


    combined = pd.concat(dfs, axis=1, sort=True)


    combined = combined.dropna()

    return combined

if __name__ == "__main__":
    data = build_dataset()
    print(data.head())
    print(data.tail())
    print(f"\nShape: {data.shape}")
    data.to_csv("raw_market_data.csv")
    print("\nSaved to raw_market_data.csv")