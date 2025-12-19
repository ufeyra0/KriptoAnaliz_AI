import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


RESIM_BOYUTU = (64, 64)
KLASOR_1D = "egitim_1d"
KLASOR_4H = "egitim_4h"
CSV_DOSYASI = "etiketler_dual.csv"

print("1. Genişletilmiş veri seti yükleniyor...")
df = pd.read_csv(CSV_DOSYASI)


print(f"Toplam Veri Sayısı: {len(df)}")

X_daily = []
X_4h = []
y = []

for index, satir in df.iterrows():
    try:
        p1 = os.path.join(KLASOR_1D, satir['dosya_adi'])
        p2 = os.path.join(KLASOR_4H, satir['dosya_adi'])
        
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None: continue

        img1 = cv2.resize(img1, RESIM_BOYUTU) / 255.0
        img2 = cv2.resize(img2, RESIM_BOYUTU) / 255.0
        
        X_daily.append(img1)
        X_4h.append(img2)
        y.append(satir['etiket'])
    except:
        continue

X_daily = np.array(X_daily).reshape(-1, 64, 64, 1)
X_4h = np.array(X_4h).reshape(-1, 64, 64, 1)
y = np.array(y)


X_train_d, X_test_d, X_train_4, X_test_4, y_train, y_test = train_test_split(
    X_daily, X_4h, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Eğitim İçin Ayrılan: {len(y_train)} adet")
print(f"Test İçin Ayrılan: {len(y_test)} adet")


def create_branch(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    return inputs, x

in_d, out_d = create_branch((64, 64, 1))
in_4, out_4 = create_branch((64, 64, 1))

birlesik = concatenate([out_d, out_4])

z = Dense(256)(birlesik)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Dropout(0.5)(z)

z = Dense(64, activation='relu')(z)
z = Dropout(0.3)(z)
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[in_d, in_4], outputs=output)

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('bitcoin_dual_model.h5', monitor='val_accuracy', save_best_only=True)
]

print("3. Eğitim Başlıyor...")
history = model.fit(
    x=[X_train_d, X_train_4], 
    y=y_train, 
    epochs=40, 
    batch_size=64, 
    validation_data=([X_test_d, X_test_4], y_test),
    callbacks=callbacks
)


best_model = tf.keras.models.load_model('bitcoin_dual_model.h5')
loss, acc = best_model.evaluate([X_test_d, X_test_4], y_test)
print(f"\n🚀 FİNAL BAŞARI (%20 Test Verisiyle): %{acc*100:.2f}")