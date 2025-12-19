import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import mplfinance as mpf
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import os
from datetime import timedelta


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


st.set_page_config(page_title="Kripto AI Analiz", layout="wide", page_icon="⚡")


st.markdown("""
<style>
    div.stButton > button {
        height: 60px;            /* Buton Yüksekliği */
        font-size: 24px;         /* Yazı Boyutu */
        font-weight: bold;       /* Kalın Yazı */
        border-radius: 10px;     /* Kenar Yuvarlama */
        border: 1px solid #444;  /* Kenarlık Rengi */
        transition: all 0.3s;    /* Animasyon */
    }
    div.stButton > button:hover {
        border-color: #00ff00;   /* Üzerine gelince yeşil çerçeve */
        color: #00ff00;          /* Üzerine gelince yeşil yazı */
    }
</style>
""", unsafe_allow_html=True)


if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC-USD'


@st.cache_resource
def model_yukle():
    try:
        return tf.keras.models.load_model('bitcoin_dual_model.h5')
    except:
        return None
model = model_yukle()


st.title("💸 AI Kripto Analiz Terminali")
st.markdown("Analiz etmek istediğiniz varlığı seçiniz. Veriler anlık olarak **Yahoo Finance** sunucularından çekilir.")


col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    
    if st.button("BITCOIN ₿", use_container_width=True):
        st.session_state.symbol = 'BTC-USD'
        st.rerun()

with col_btn2:
    
    if st.button("ETHEREUM ⟠", use_container_width=True):
        st.session_state.symbol = 'ETH-USD'
        st.rerun()


current_symbol = st.session_state.symbol
coin_name = "Bitcoin" if current_symbol == 'BTC-USD' else "Ethereum"
coin_short = "BTC" if current_symbol == 'BTC-USD' else "ETH"
coin_color = "#F7931A" if current_symbol == 'BTC-USD' else "#627EEA" 


st.sidebar.title(f"⚙️ {coin_short} Ayarları")
st.sidebar.markdown(f"**Seçili Varlık:** <span style='color:{coin_color}; font-size:18px'>{coin_name}</span>", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.header("💼 Cüzdan")
coin_miktari = st.sidebar.number_input(f"Elinizdeki {coin_short}", value=0.000000, format="%.6f", step=0.001)
alis_maliyeti = st.sidebar.number_input("Toplam Maliyet ($)", value=0.0, step=50.0)
st.sidebar.divider()
st.sidebar.caption("Veriler canlıdır. 'Piyasayı Yenile' butonu son fiyatı çeker.")
yenile_butonu = st.sidebar.button("🔄 Piyasayı Yenile", use_container_width=True)


def interaktif_grafik_ciz(df, baslik, cizgiler=[], zoom_start=None, zoom_end=None):
    fig = go.Figure()
    df['DateStr'] = df.index.strftime('%d %b %H:%M')

    fig.add_trace(go.Candlestick(
        x=df['DateStr'], 
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Fiyat'
    ))

    if 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['DateStr'], y=df['SMA20'], 
            line=dict(color='orange', width=1), name='Trend (SMA20)'
        ))

    colors = ['#ff4b4b', '#00ff00', '#29b5e8'] 
    labels = ['Direnç', 'Destek', 'Pivot']
    
    if cizgiler:
        for i, seviye in enumerate(cizgiler):
            fig.add_hline(
                y=seviye, line_width=1, line_dash="dash", 
                line_color=colors[i], annotation_text=labels[i], 
                annotation_position="top right"
            )

    x_range = None
    if zoom_start and zoom_end:
        try:
            start_idx = df.index.searchsorted(zoom_start)
            end_idx = df.index.searchsorted(zoom_end)
            start_idx = max(0, start_idx)
            end_idx = min(len(df)-1, end_idx)
            x_range = [start_idx, end_idx + 5] 
        except:
            x_range = None

    layout_args = dict(
        title=dict(text=baslik, x=0, xanchor='left'),
        yaxis_title='USD', template="plotly_dark",
        height=500, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(type='category', nticks=10, tickangle=-45) 
    )
    
    if x_range:
        layout_args['xaxis_range'] = x_range

    fig.update_layout(**layout_args)
    return fig


def resim_hazirla(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(1, 64, 64, 1)


c1, c2 = st.columns([3, 1])
with c1:
    st.subheader(f"⚡ {coin_name} Analiz Ekranı")
with c2:
    if model is None: st.error("Model Hatası")
    else: st.success(f"Yapay Zeka: {coin_short} Modunda 🟢")

levels = []
anlik_fiyat = 0
direnc, destek = 0, 0
prob = 0.5
trend_yukari = True

with st.spinner(f'{coin_name} için global piyasa verileri taranıyor...'):
    try:
        ticker = yf.Ticker(current_symbol)
        df_daily = ticker.history(period='1y', interval='1d')
        df_hourly = ticker.history(period='1y', interval='1h')
        
        if df_hourly.empty or df_daily.empty:
            st.error("Veri çekilemedi. Lütfen sayfayı yenileyin.")
            st.stop()
            
        df_daily.index = df_daily.index.tz_localize(None)
        df_hourly.index = df_hourly.index.tz_localize(None)
        
        ohlc_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        df_4h = df_hourly.resample('4h').agg(ohlc_rules).dropna()

        df_daily['SMA20'] = df_daily['Close'].rolling(window=20).mean()
        df_4h['SMA20'] = df_4h['Close'].rolling(window=20).mean()
        
        ai_daily = df_daily.iloc[-40:].copy()
        ai_4h = df_4h.iloc[-40:].copy()
        
        focus_start_d = ai_daily.index[0]
        focus_end_d = ai_daily.index[-1]
        focus_start_4 = ai_4h.index[0]
        focus_end_4 = ai_4h.index[-1]
        
        anlik_fiyat = df_hourly['Close'].iloc[-1]
        
        tepe = ai_daily['High'].max()
        dip = ai_daily['Low'].min()
        fark = tepe - dip
        
        direnc = tepe
        destek = dip
        pivot = tepe - (fark * 0.5)
        levels = [direnc, destek, pivot] 
        
        path_d, path_4 = "ai_d.png", "ai_4.png"
        mpf.plot(ai_daily, type='candle', mav=(20,), axisoff=True, volume=False, savefig=path_d)
        mpf.plot(ai_4h, type='candle', mav=(5,13), axisoff=True, volume=False, savefig=path_4)
        
        if model:
            prob = model.predict([resim_hazirla(path_d), resim_hazirla(path_4)])[0][0]
            trend_yukari = prob > 0.5
        else:
            prob, trend_yukari = 0.5, True

    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")
        st.stop()


gosterilecek_guven = prob if trend_yukari else (1 - prob)

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Anlık {coin_short}", f"${anlik_fiyat:,.2f}") 
k2.metric("Ana Direnç", f"${direnc:,.0f}")
k3.metric("Ana Destek", f"${destek:,.0f}")
k4.metric("AI Güven Skoru", f"%{gosterilecek_guven*100:.1f}")

st.divider()

col_g1, col_g2 = st.columns(2)
with col_g1:
    fig_d = interaktif_grafik_ciz(df_daily, f"📅 {coin_name} Günlük Grafik", cizgiler=levels, zoom_start=focus_start_d, zoom_end=focus_end_d)
    st.plotly_chart(fig_d, use_container_width=True, key=f"daily_{coin_short}")

with col_g2:
    fig_4 = interaktif_grafik_ciz(df_4h, f"⏱️ {coin_name} 4-Saatlik Grafik", cizgiler=levels, zoom_start=focus_start_4, zoom_end=focus_end_4)
    st.plotly_chart(fig_4, use_container_width=True, key=f"4h_{coin_short}")

st.markdown(f"### 📊 Yapay Zeka Kararı ({coin_short})")
alt1, alt2 = st.columns([1, 1.5])

with alt1:
    if trend_yukari:
        st.markdown("<h1 style='color:#00ff00;'>YÜKSELİŞ 🚀</h1>", unsafe_allow_html=True)
        st.write(f"AI, {coin_name} grafiğindeki formasyonları analiz etti ve YÜKSELİŞ öngörüyor.")
    else:
        st.markdown("<h1 style='color:#ff4b4b;'>DÜŞÜŞ 📉</h1>", unsafe_allow_html=True)
        st.write(f"AI, {coin_name} grafiğindeki formasyonları analiz etti ve DÜŞÜŞ öngörüyor.")
    st.progress(float(gosterilecek_guven))

with alt2:
    if coin_miktari == 0:
        st.info(f"ℹ️ Portföy hesabı için sol menüden '{coin_short} Adedi' giriniz.")
    else:
        cuzdan_degeri = anlik_fiyat * coin_miktari
        kar_zarar = cuzdan_degeri - alis_maliyeti
        renk = "#00ff00" if kar_zarar >= 0 else "#ff4b4b"
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Cüzdan Değeri", f"${cuzdan_degeri:,.2f}")
        col_p2.markdown(f"**Net Kâr/Zarar:** <span style='color:{renk}; font-size:24px'>${kar_zarar:,.2f}</span>", unsafe_allow_html=True)

    if trend_yukari:
        st.success(f"💡 TAVSİYE: Yön YUKARI. Hedef: **${direnc:,.0f}**")
    else:
        st.error(f"💡 TAVSİYE: Yön AŞAĞI. Destek: **${destek:,.0f}**")