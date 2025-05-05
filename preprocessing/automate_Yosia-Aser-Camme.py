import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_dataset(file_path, save_path=None):
    """
    Fungsi untuk melakukan preprocessing dataset secara otomatis.
    
    Args:
        file_path (str): Path ke file dataset (CSV).
        save_path (str, optional): Path untuk menyimpan dataset hasil preprocessing. Default None.
    
    Returns:
        pd.DataFrame: Dataset yang telah diproses.
    """
    # Memuat dataset
    dataset = pd.read_csv(file_path, sep=';')
    
    # 1. Mengisi Nilai Kosong (Missing Values)
    if dataset.isnull().values.any():
        print("Dataset memiliki nilai kosong. Mengisi nilai kosong...")
        for col in dataset.columns:
            if dataset[col].dtype in ['float64', 'int64']:
                dataset[col].fillna(dataset[col].mean(), inplace=True)  # Isi dengan rata-rata
            else:
                dataset[col].fillna(dataset[col].mode()[0], inplace=True)  # Isi dengan mode
    
    # 2. Menghapus Data Duplikat
    dataset = dataset.drop_duplicates()
    if dataset.empty:
        raise ValueError("Dataset is empty after removing duplicates.")
    
    # 3. Normalisasi atau Standarisasi Fitur
    numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.empty:
        scaler = StandardScaler()
        dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    else:
        raise ValueError("Dataset is empty after handling missing values or outliers.")
    
    # 4. Deteksi dan Penanganan Outlier
    Q1 = dataset[numeric_columns].quantile(0.25)
    Q3 = dataset[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (dataset[numeric_columns] < (Q1 - 1.5 * IQR)) | (dataset[numeric_columns] > (Q3 + 1.5 * IQR))
    dataset = dataset[~outlier_condition.any(axis=1)]
    if dataset.empty:
        raise ValueError("Dataset is empty after removing outliers.")
    
    # 5. Encoding Data Kategorikal
    for col in dataset.select_dtypes(include=['object']).columns:
        dataset[col] = dataset[col].astype('category').cat.codes
    
    # 6. Binning (Pengelompokan Data)
    if 'T' in dataset.columns:
        bins = [-float('inf'), 0, 10, 20, 30, float('inf')]
        labels = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']
        dataset['Temperature_Binned'] = pd.cut(dataset['T'], bins=bins, labels=labels)
    
    # Menyimpan dataset hasil preprocessing jika save_path diberikan
    if save_path:
        dataset.to_csv(save_path, index=False)
    
    return dataset

def main():
    # Path ke dataset mentah
    raw_dataset_path = r'C:\Users\Shinkai\Pictures\Eksperimen_SML_Yosia-Aser-Camme\Air-Quality_raw.csv'
    
    # Path untuk menyimpan dataset hasil preprocessing
    processed_dataset_path = r'C:\Users\Shinkai\Pictures\Eksperimen_SML_Yosia-Aser-Camme\preprocessing\Air-Quality_preprocessing.csv'
    
    # Memproses dataset
    print("Memulai proses preprocessing dataset...")
    processed_data = preprocess_dataset(raw_dataset_path, save_path=processed_dataset_path)
    print("Proses preprocessing selesai.")
    
    # Menampilkan beberapa baris awal dataset hasil preprocessing
    print("Dataset hasil preprocessing:")
    print(processed_data.head())

if __name__ == "__main__":
    # Menjalankan fungsi utama
    main()