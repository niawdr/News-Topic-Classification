# Klasifikasi Topik Berita Otomatis Menggunakan Metode LSTM dan Bi-LSTM Untuk Analisis Konten Media   

## Deskripsi  
Proyek ini bertujuan untuk mengembangkan sistem klasifikasi topik berita digital menggunakan metode Long Short-Term Memory (LSTM) dan Bidirectional LSTM (Bi-LSTM).   

Model dikembangkan dan dievaluasi menggunakan dataset berita dari portal **CNN Indonesia**, dengan enam kategori topik:  
1. Ekonomi  
2. Hiburan  
3. Hukum dan Kriminal  
4. Kesehatan  
5. Politik  
6. Teknologi  

Hasil evaluasi menunjukkan bahwa model **Bi-LSTM tanpa FastText** memiliki performa terbaik dengan:  
- **Validation Loss**: 0.4705  
- **Validation Accuracy**: 87.60%  
- **Precision, Recall, dan F1-Score**: 0.88    

## Instalasi  
1. Clone repository ini:  
   ```bash
   git clone https://github.com/niawdr/News-Topic-Classification.git
   cd News-Topic-Classification

