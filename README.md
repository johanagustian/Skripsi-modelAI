# Apta AI Model: Collaborative Planning & Optimization

Sistem pembangkit soal adaptif berbasis **Text-to-Text Transfer Transformer (T5)** yang diintegrasikan dengan **Item Response Theory (IRT)** untuk optimasi pengukuran pendidikan.

---

## 🚀 Persiapan Awal

Proyek ini menggunakan **Git LFS (Large File Storage)** untuk mengelola bobot model (*model weights*).

### 1. Clone Repository

```bash
git clone https://github.com/johanagustian/Skripsi-modelAI.git
cd Skripsi-modelAI
```

### 2. Install Git LFS dan Download Model

```bash
git lfs install
git lfs pull
```

### 3. Setup Virtual Environment

```bash
python -m venv venv
```

Aktivasi virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠 Cara Menjalankan Aplikasi

Pastikan file `model.safetensors` sudah tersedia di folder `models/`.

Jika belum tersedia, jalankan:

```bash
git lfs pull
```

Jalankan aplikasi:

```bash
uvicorn main:app --reload --port 8000
```

Akses dokumentasi API melalui:

```text
http://127.0.0.1:8000/docs
```

---

## 📁 Struktur Folder

```text
├── models/       # Bobot model T5 (.safetensors), konfigurasi, tokenizer
├── routers/      # Endpoint API
├── schemas/      # Skema validasi Pydantic
├── services/     # Logika bisnis dan integrasi model
├── main.py       # Entry point aplikasi
└── requirements.txt
```

### Deskripsi Folder

| Folder      | Deskripsi                                               |
| ----------- | ------------------------------------------------------- |
| `models/`   | Berisi bobot model T5, konfigurasi model, dan tokenizer |
| `routers/`  | Endpoint API yang digunakan oleh aplikasi               |
| `schemas/`  | Validasi data menggunakan Pydantic                      |
| `services/` | Logika bisnis dan integrasi model AI                    |

---

## 📦 Finalisasi Dependencies

Untuk memastikan seluruh library yang digunakan dapat direplikasi pada environment lain:

```bash
pip freeze > requirements.txt
```

Commit perubahan:

```bash
git add requirements.txt README.md
git commit -m "Add documentation and requirements"
git push -u origin main
```

---

## 🔧 Teknologi yang Digunakan

* Python
* FastAPI
* Transformers (Hugging Face)
* PyTorch
* Git LFS
* Uvicorn
* Pydantic

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan penelitian dan pengembangan sistem pembelajaran adaptif berbasis AI.
