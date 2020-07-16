import os
import argparse

import yfinance as yf


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", type=str, help="Ticker(s) to get data from")
    parser.add_argument("--start", type=str, help="Start date")
    parser.add_argument("--end", type=str, help="End date")
    parser.add_argument("--period", type=str, default="max",
                        help="Period instead of start/end (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd")
    parser.add_argument("--interval", type=str, default="1d",
                        help="Get data by interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)")

    return parser


def save_yahoo_data(tickers: str, start=None, end=None, interval="1d", period="max", save=False, **kwargs):
    for ticker in tickers.split():
        print("Downloading", ticker)
        data = yf.download(ticker, start=start, end=end, period=period, interval=interval, **kwargs)

        if data.empty:
            continue

        if start or end:
            filename = f"{ticker.upper()}_{start}-{end}_{interval}.csv"
        else:
            filename = f"{ticker.upper()}_{period}_{interval}.csv"

        if save:
            data.to_csv("data/" + filename)


if __name__ == "__main__":
    parser = get_parser()
    config, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    save_yahoo_data(config.tickers, start=config.start, end=config.end,
                    period=config.period, interval=config.interval, save=False)
