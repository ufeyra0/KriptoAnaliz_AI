import yfinance as yf
import pandas as pd
import os

print("Veri İndirme Merkezi Başlatıldı...")

def veri_cek(periyot, aralik, dosya_adi):
    print(f"--- {dosya_adi} indiriliyor ({periyot}) ---")
    try:
        
        data = yf.download('BTC-USD', period=periyot, interval=aralik, progress=False)
        
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
       
        if 'Close' not in data.columns and 'Adj Close' in data.columns:
             data['Close'] = data['Adj Close']
        data = data[['Open', 'High', 'Low', 'Close']]
        data = data.dropna()
        
        
        data.to_csv(dosya_adi)
        print(f"✅ {dosya_adi} kaydedildi. Satır sayısı: {len(data)}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")


veri_cek(periyot='2y', aralik='1d', dosya_adi='btc_daily.csv')  
veri_cek(periyot='2y', aralik='1h', dosya_adi='btc_hourly.csv') 

print("\n🚀 İndirme tamamlandı! Şimdi veri_uretimi.py çalıştırılabilir.")