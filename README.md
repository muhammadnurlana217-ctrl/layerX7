# LayerX Studio

Pengubah satu gambar menjadi banyak layer PNG transparan, langsung di browser (Android/desktop). Tanpa Node.js.

## Menjalankan
Hosting folder ini sebagai situs statis (GitHub Pages, Netlify, Cloudflare Pages), atau jalankan server lokal. ES module dan model AI tidak bisa dimuat dari `file://`.
Butuh internet saat pertama kali: Transformers.js (jsDelivr), JSZip (cdnjs), dan model (Hugging Face, ±4 MB q8, lalu di-cache browser).

## Cara kerja
- **Otomatis**: model SegFormer-B0 (ADE20K, 150 kelas) memberi peta area. Tiap area dipecah per bagian terpisah dan dijadikan layer (maks 40, terbesar di bawah).
- **Manual** (opsional): aktifkan, lingkari area dengan jari/mouse untuk membuat layer baru. Untuk memperbaiki hasil AI, hapus layer lalu lingkari ulang.
- **Layer manager**: ganti nama, tampil/sembunyi, urutan, unduh PNG, hapus, preview gabungan dengan kotak transparansi.
- **Export**: ZIP (`layers/layer_01_nama.png`, `composite.png`, `metadata.json`), PNG gabungan, PNG individual.

## Catatan jujur
- Model mengenali kategori umum (dinding, langit, orang, mobil, dll). Objek di luar kategori atau tepi detail bisa tidak sempurna.
- Gambar di atas 2048 px disusutkan agar RAM HP aman. Tidak ada upscale; warna tidak diubah.
- Semua layer berukuran sama dengan gambar kerja; tumpuk di posisi 0,0. Posisi objek ada di `bbox` pada metadata.
- Ini **bukan** file project Alight Motion native. Impor PNG berurutan ke Alight Motion (Media → Gambar), semuanya pada posisi tengah/sama.

## Mode & XML Alight Motion
- **Mode UI/Screenshot** (default): memisahkan area warna rata besar (latar) dan mengelompokkan elemen kecil yang berdekatan (teks, ikon, avatar). Bagian latar yang tertutup elemen diisi warna rata agar elemen bisa digeser.
- **alightmotion.xml** dibuat mengikuti struktur contoh XML dari Alight Motion 5.0 (scene + media + shape `.rect`, tiap layer menempati seluruh kanvas). Alamat file memakai `file://<folder>/layers/<nama>.png`, jadi ekstrak ZIP lebih dulu dan isi kolom folder di aplikasi sesuai lokasinya. Jika Alight Motion meminta media diganti, arahkan ke folder `layers`.
