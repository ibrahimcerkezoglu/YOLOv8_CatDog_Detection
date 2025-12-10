# YOLOv8_CatDog_Detection 🐱🐶

BLG-407 Makine Öğrenmesi dersi için hazırlanmış olan bu projede, **kedi** ve **köpek** sınıflarından oluşan bir veri seti üzerinde **YOLOv8** ile nesne tespiti modeli eğitilmiş ve elde edilen `best.pt` modeli **PyQt5 tabanlı bir masaüstü uygulamada** kullanılmıştır.

## 1. Proje Özeti

- **Amaç:** Kedi ve köpek nesnelerini görüntü üzerinde tespit etmek ve bunu kullanıcıya basit bir PyQt5 arayüzü ile göstermek.
- **Model:** YOLOv8n (Ultralytics)
- **Sınıflar:** `cat`, `dog`
- **Çıktı:** 
  - Eğitim sürecini gösteren `yolo_training.ipynb`
  - Eğitilmiş model ağırlığı `best.pt`
  - PyQt5 GUI uygulaması `gui_app.py`

---

## 2. Klasör / Dosya Yapısı

```text
YOLOv8_CatDog_Detection/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── best.pt
├── gui_app.py
├── yolo_training.ipynb
├── requirements.txt  (opsiyonel)
└── README.md
```
---

## 3. Kullanılan Teknolojiler

Python 3.10

PyTorch + CUDA (NVIDIA GeForce RTX 4060 Laptop GPU)

Ultralytics YOLOv8

OpenCV

Matplotlib

PyQt5

---

## 4. Kurulum

Projeyi klonladıktan sonra aşağıdaki adımlar takip edilerek ortam oluşturulabilir.

# 1) Sanal ortam (örnek: conda)
conda create -n tf_gpu python=3.10
conda activate tf_gpu

# 2) Gerekli paketler
pip install ultralytics==8.3.0 opencv-python matplotlib pyqt5
# PyTorch için (CUDA sürümüne göre) resmi PyTorch sitesindeki komut kullanılmalıdır.

İsteğe bağlı olarak requirements.txt şu içerikle oluşturulabilir:

ultralytics==8.3.0
opencv-python
matplotlib
pyqt5
torch
torchvision

---

## 5. YOLOv8 Eğitim Süreci

Eğitim adımlarının tamamı yolo_training.ipynb dosyasında detaylı şekilde gösterilmiştir.

Özetle:

Gerekli kütüphaneler yüklenir ve ortam kontrol edilir (PyTorch, CUDA vb.).

Veri seti yolu ve data.yaml dosyası ayarlanır.

Hazır yolov8n.pt tabanlı model yüklenir.

Aşağıdaki parametrelerle eğitim yapılır:

```bash
results = model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="cats_dogs_v1",
    project="runs/train",
    patience=10
)
```

**Eğitim sonunda YOLO tarafından üretilen:**

Loss & mAP grafikleri (results.png)

En iyi ağırlık dosyası: runs/train/cats_dogs_v1/weights/best.pt
dosyaları kullanılır ve best.pt proje kök dizinine kopyalanır.

**Eğitim sonrası elde edilen temel metrikler:**

mAP@0.5: ≈ 0.51

mAP@0.5:0.95: ≈ 0.31

**Sınıflar:**

cat mAP ≈ 0.48

dog mAP ≈ 0.54

## 6. PyQt5 GUI Uygulaması

GUI uygulaması gui_app.py dosyasında yer almaktadır.

**6.1. Çalıştırma**

```bash
python gui_app.py
```

**6.2. Arayüz Özellikleri**

Arayüz iki ana panelden oluşur:

Original Image: Kullanıcının seçtiği ham görüntü.

Tagged Image: YOLOv8 modeli ile analiz edilip bounding box çizilmiş çıktı görüntüsü.

Alt kısımda ise şu butonlar yer alır:

Select Image: Bilgisayardan bir görüntü seçer.

Test Image: Seçilen görüntüyü YOLOv8 modeline gönderir, tahminleri alır ve bounding box’ları çizer.

Save Image: Bounding box çizilmiş çıktıyı diske kaydeder.

(Opsiyonel) Video / kamera desteği istenirse aynı mantıkla eklenebilir.

**6.3. Tespit Sonuçları**

Tespit edilen nesne sayısı ve sınıfı arayüz alt kısmında gösterilir.
Örnek: Tespit Sonucu: cat: 1, dog: 2 vb.

