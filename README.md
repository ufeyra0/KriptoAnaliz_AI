# 💸 AI Crypto Analysis Terminal (Yapay Zeka Destekli Kripto Analiz)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

Bu proje, **Bitcoin (BTC)** ve **Ethereum (ETH)** grafiklerini gerçek zamanlı olarak analiz eden, görüntü işleme (OpenCV) 
ve Derin Öğrenme (CNN) tekniklerini kullanarak piyasa yönünü tahmin eden bir web uygulamasıdır.

## 🚀 Proje Hakkında

**Trade Master AI**, finansal piyasalardaki mum (candlestick) grafiklerini birer "görüntü" olarak algılar. 
Geleneksel indikatörlerin aksine, bu sistem grafikleri **görsel olarak analiz eder**.

Proje, hem Günlük (Daily) hem de 4-Saatlik (4H) grafikleri aynı anda işleyerek **Hibrit bir Yapay Zeka Modeli** (Dual-Input CNN)
kullanır ve kullanıcıya Yükseliş (Long) veya Düşüş (Short) yönünde bir güven skoru sunar.

### 🌟 Temel Özellikler

* **Canlı Veri Akışı:** Yahoo Finance API üzerinden anlık veri çekimi.
* **Görüntü İşleme:** Grafikler arka planda oluşturulur, `OpenCV` ile gri tonlamaya çevrilir ve 64x64 boyutunda normalize edilir.
* **Çift Zamanlı Analiz:** Hem kısa vade (4S) hem uzun vade (Günlük) trend analizi.
* **İnteraktif Grafikler:** `Plotly` ile oluşturulmuş, yakınlaştırılabilir profesyonel borsa grafikleri.
* **Portföy Takibi:** Anlık fiyat üzerinden kâr/zarar hesaplama modülü.
* **User-Agent Koruması:** Bot engellemelerine karşı özel oturum yönetimi.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python
* **Arayüz:** Streamlit
* **Yapay Zeka:** TensorFlow / Keras (CNN)
* **Görüntü İşleme:** OpenCV, NumPy
* **Veri Kaynağı:** yfinance
* **Görselleştirme:** Plotly, mplfinance

## 📸 Ekran Görüntüleri

<img width="1892" height="918" alt="image" src="https://github.com/user-attachments/assets/4b300bfb-91ba-43cc-bfb4-49f22a6c8586" />
<img width="1900" height="912" alt="image" src="https://github.com/user-attachments/assets/8774ac17-07ff-454a-9c41-e1786943de94" />



## ⚙️ Kurulum ve Çalıştırma

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Projeyi Klonlayın

git clone [https://github.com/KULLANICI_ADINIZ/proje-isminiz.git](https://github.com/KULLANICI_ADINIZ/proje-isminiz.git)
cd proje-isminiz

### 2. Gerekli Kütüphaneleri Yükleyin
pip install -r requirements.txt

### 3. Uygulamayı Başlatın
python -m streamlit run app.py


### 🧠 Model Mimarisi
Sistem, önceden eğitilmiş bir .h5 modeli kullanır. Model şu adımlardan oluşur:

1.Giriş: 2 Adet Görüntü (Günlük Grafik + 4 Saatlik Grafik).

2.İşleme: Her iki görüntü için ayrı Konvolüsyon (Conv2D) katmanları.

3.Birleştirme: İki koldan gelen verilerin Concatenate ile birleşmesi.

4.Karar: Dense katmanları ve Sigmoid aktivasyon fonksiyonu ile 0-1 arası olasılık hesabı.


### 📂 Dosya Yapısı
app.py: Ana uygulama ve arayüz kodları.

bitcoin_dual_model.h5: Eğitilmiş Yapay Zeka modeli.

requirements.txt: Gerekli kütüphane listesi.
