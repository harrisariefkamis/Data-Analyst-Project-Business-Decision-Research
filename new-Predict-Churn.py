# Prediksi Churn untuk Data Pelanggan Baru
# Kita akan membuat fungsi predict_churn_new_customers yang akan menerima DataFrame berisi data pelanggan baru, melakukan preprocessing yang sama seperti pada data pelatihan, dan kemudian menggunakan model Random Forest yang telah dilatih untuk memprediksi status churn.
import pickle
import pandas as pd

# Memuat model Random Forest yang sudah disimpan
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_rf_model = pickle.load(file)

print(f"Model '{model_filename}' berhasil dimuat.")

def predict_churn_new_customers(new_customer_df, ohe_encoder, trained_feature_columns):
    """
    Memprediksi churn untuk data pelanggan baru menggunakan model Random Forest yang sudah dilatih.

    Args:
        new_customer_df (pd.DataFrame): DataFrame berisi data pelanggan baru.
        ohe_encoder (OneHotEncoder): Objek OneHotEncoder yang telah dilatih pada data training.
        trained_feature_columns (list): Daftar nama kolom fitur yang digunakan saat pelatihan model.

    Returns:
        pd.Series: Prediksi churn (True/False) untuk setiap pelanggan baru.
    """
    # Membuat salinan agar tidak mengubah DataFrame asli
    processed_df = new_customer_df.copy()

    # --- Preprocessing Data Baru (harus sama dengan data pelatihan) ---

    # 1. Konversi kolom tanggal jika ada dan belum dalam format datetime
    # Asumsi kolom 'First_Transaction' dan 'Last_Transaction' ada dan perlu dikonversi
    if 'First_Transaction' in processed_df.columns and not pd.api.types.is_datetime64_any_dtype(processed_df['First_Transaction']):
        processed_df['First_Transaction'] = pd.to_datetime(processed_df['First_Transaction'] / 1000, unit='s', origin='1970-01-01')
    if 'Last_Transaction' in processed_df.columns and not pd.api.types.is_datetime64_any_dtype(processed_df['Last_Transaction']):
        processed_df['Last_Transaction'] = pd.to_datetime(processed_df['Last_Transaction'] / 1000, unit='s', origin='1970-01-01')

    # 2. Ekstraksi Tahun Transaksi
    if 'First_Transaction' in processed_df.columns:
        processed_df['Year_First_Transaction'] = processed_df['First_Transaction'].dt.year
    if 'Last_Transaction' in processed_df.columns:
        processed_df['Year_Last_Transaction'] = processed_df['Last_Transaction'].dt.year

    # 3. Handle Kolom 'Product' menggunakan OneHotEncoder yang sudah dilatih
    if 'Product' in processed_df.columns:
        product_encoded_new = ohe_encoder.transform(processed_df[['Product']])
        product_df_new = pd.DataFrame(product_encoded_new, columns=ohe_encoder.get_feature_names_out(['Product']))
        processed_df = pd.concat([processed_df.drop(columns=['Product']).reset_index(drop=True), product_df_new], axis=1)
    
    # 4. Pilih hanya kolom fitur yang digunakan saat pelatihan
    # Penting: Pastikan urutan kolom sesuai dengan data pelatihan
    # Kita akan membuat DataFrame baru dengan hanya kolom yang diharapkan dan mengisi NaN jika ada kolom yang hilang
    final_features = pd.DataFrame(columns=trained_feature_columns)
    for col in trained_feature_columns:
        if col in processed_df.columns:
            final_features[col] = processed_df[col]
        else:
            # Jika ada kolom yang tidak ada di data baru (misal Product_X jika tidak ada Product X)
            # Isi dengan 0.0 sesuai dengan output OneHotEncoder untuk kategori yang tidak terlihat.
            # Ini penting untuk konsistensi dimensi.
            final_features[col] = 0.0

    # Prediksi menggunakan model yang dimuat
    predictions = loaded_rf_model.predict(final_features)

    return pd.Series(predictions, index=new_customer_df.index)

print("Fungsi `predict_churn_new_customers` telah didefinisikan.")

#Contoh Penggunaan Fungsi Prediksi Churn
#Sekarang kita akan membuat beberapa contoh data pelanggan baru dan menggunakan fungsi yang baru saja kita definisikan untuk memprediksi churn mereka.#
#Perhatikan bahwa kita akan menggunakan objek ohe yang sudah dilatih sebelumnya (dari cell c9c5b060) dan nama-nama kolom fitur dari X yang sudah diproses (dari kernel state).#
# Contoh data pelanggan baru
new_customers_data = {
    'Customer_ID': [100001, 100002, 100003, 100004, 100005],
    'Product': ['Jaket', 'Sepatu', 'Baju', 'Tas', 'Jaket'],
    'First_Transaction': [1580000000000, 1500000000000, 1400000000000, 1590000000000, 1600000000000], # Timestamp dalam ms
    'Last_Transaction': [1600000000000, 1510000000000, 1400000000000, 1595000000000, 1600000000000], # Timestamp dalam ms
    'Average_Transaction_Amount': [500000, 1500000, 200000, 800000, 1200000],
    'Count_Transaction': [2, 10, 1, 5, 3]
}

new_customer_df = pd.DataFrame(new_customers_data)

print("Data pelanggan baru:")
display(new_customer_df)

# Dapatkan nama-nama kolom fitur dari DataFrame X yang digunakan saat pelatihan
# (X adalah DataFrame yang sudah di-one-hot encode dan disiapkan untuk pelatihan)
# Asumsi `X` masih ada di kernel state.
# Jika tidak, kita bisa merekonstruksi daftar kolom:
# trained_feature_columns = ['Average_Transaction_Amount', 'Count_Transaction', 'Year_First_Transaction', 'Year_Last_Transaction', 'Product_Baju', 'Product_Jaket', 'Product_Sepatu', 'Product_Tas']

trained_feature_columns = X.columns.tolist()

# Lakukan prediksi churn
churn_predictions = predict_churn_new_customers(new_customer_df, ohe, trained_feature_columns)

# Tambahkan prediksi ke DataFrame pelanggan baru untuk visualisasi yang lebih baik
new_customer_df['Predicted_Churn'] = churn_predictions

print("\nHasil prediksi churn untuk pelanggan baru:")
display(new_customer_df)

#visualisiasi Predict new
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Customer_ID',
    y='Predicted_Churn',
    data=new_customer_df,
    palette={True: 'salmon', False: 'lightgreen'},
    hue='Predicted_Churn',
    legend=False
)
plt.title('Prediksi Churn untuk Pelanggan Baru')
plt.xlabel('Customer ID')
plt.ylabel('Predicted Churn (True = Churn, False = No Churn)')
plt.yticks([0, 1], ['No Churn', 'Churn'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Ekspor Hasil Prediksi Churn ke File CSV
# Tentukan nama file CSV output
output_filename = 'new_customer_churn_predictions.csv'

# Ekspor DataFrame ke CSV
new_customer_df.to_csv(output_filename, index=False)

print(f"Hasil prediksi churn pelanggan baru telah diekspor ke '{output_filename}'")
print("Lima baris pertama dari file CSV yang diekspor:")
display(pd.read_csv(output_filename).head())
