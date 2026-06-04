# heart_disease_preprocessing

Folder ini berisi dataset yang sudah diproses dari tahap eksperimen.

## File yang dihasilkan oleh automate_NamaSiswa.py:
- `X_train.csv` - Fitur training
- `X_test.csv`  - Fitur testing
- `y_train.csv` - Label training
- `y_test.csv`  - Label testing
- `train_preprocessed.csv` - Dataset training lengkap
- `test_preprocessed.csv`  - Dataset testing lengkap
- `scaler.pkl`  - StandardScaler yang sudah di-fit
- `feature_columns.pkl` - Daftar nama kolom fitur

## Cara regenerasi:
```bash
python preprocessing/automate_NamaSiswa.py
```
