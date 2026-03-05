import yfinance as yf
import pandas as pd

ndxt = yf.download("^NDXTR", auto_adjust=True)

ndxt.to_pickle("/home/djmann/corrected_cache_v8/ndxt_total_return_local.pkl")