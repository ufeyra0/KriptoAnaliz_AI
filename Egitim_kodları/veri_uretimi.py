import mplfinance as mpf
import pandas as pd
import os


PENCERE = 40          
KLASOR_1D = "egitim_1d"
KLASOR_4H = "egitim_4h"


for k in [KLASOR_1D, KLASOR_4H]:
    if not os.path.exists(k):
        os.makedirs(k)

print("Veriler hazırlanıyor...")


try:
    df_daily = pd.read_csv('btc_daily.csv', index_col=0, parse_dates=True)
    df_hourly = pd.read_csv('btc_hourly.csv', index_col=0, parse_dates=True)
except:
    print("CSV dosyaları eksik! Önce veri_indir.py çalıştır.")
    exit()


if df_daily.index.tz is not None:
    df_daily.index = df_daily.index.tz_localize(None)
if df_hourly.index.tz is not None:
    df_hourly.index = df_hourly.index.tz_localize(None)


ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
df_4h = df_hourly.resample('4h').apply(ohlc_dict).dropna()

print(f"Günlük Veri: {len(df_daily)} | 4 Saatlik Veri: {len(df_4h)}")


df_daily['SMA20'] = df_daily['Close'].rolling(window=20).mean()
df_daily = df_daily.dropna()

etiketler = []
hata_sayaci = 0 

print("Veri artırımı devrede! Günde 6 adet resim üretiliyor...")

for tarih in df_4h.index:
    try:
        
        idx_4h = df_4h.index.get_loc(tarih)
        if idx_4h < PENCERE: 
            continue
        
        fourh_window = df_4h.iloc[idx_4h-PENCERE : idx_4h]
        
        
        tarih_gunluk_bitis = tarih.normalize() - pd.Timedelta(days=1)
        
        if tarih_gunluk_bitis not in df_daily.index:
            
            if hata_sayaci < 5:
                print(f"Atlandı: {tarih_gunluk_bitis} tarihi günlük veride yok.")
                hata_sayaci += 1
            continue
            
        idx_d = df_daily.index.get_loc(tarih_gunluk_bitis)
        if idx_d < PENCERE: 
            continue
            
        daily_window = df_daily.iloc[idx_d-PENCERE : idx_d+1]

        
        son_fiyat = daily_window['Close'].iloc[-1]
        son_sma = daily_window['SMA20'].iloc[-1]
        etiket = 1 if son_fiyat > son_sma else 0
        
        
        fname = f"{tarih.strftime('%Y%m%d_%H%M')}.png"
        path_1d = os.path.join(KLASOR_1D, fname)
        path_4h = os.path.join(KLASOR_4H, fname)
        
        
        save_kargs = dict(dpi=40, bbox_inches='tight', pad_inches=0)
        
        mpf.plot(daily_window, type='candle', style='charles', mav=(20,),
                 axisoff=True, volume=False, savefig=dict(fname=path_1d, **save_kargs))
        
        mpf.plot(fourh_window, type='candle', style='yahoo', mav=(5,13),
                 axisoff=True, volume=False, savefig=dict(fname=path_4h, **save_kargs))
        
        etiketler.append([fname, etiket])

    except Exception as e:
        print(f"Beklenmedik Hata: {e}")
        continue 

    if len(etiketler) > 0 and len(etiketler) % 200 == 0:
        print(f"🖼️ {len(etiketler)} adet veri üretildi...")


df_lbl = pd.DataFrame(etiketler, columns=['dosya_adi', 'etiket'])
df_lbl.to_csv('etiketler_dual.csv', index=False)

if len(etiketler) == 0:
    print(" HATA: Hala 0 resim üretildi. Lütfen 'btc_daily.csv' dosyasını açıp tarih formatını kontrol et (YYYY-MM-DD olmalı).")
else:
    print(f" VERİ SETİ HAZIR! Toplam {len(etiketler)} çift resim üretildi.")