# Turkish License Plate Detection with YOLO11n

Bu proje, YOLO11n kullanılarak Türk plakalarının tespiti üzerine geliştirilmiştir. Model, [Kaggle'dan alınan Turkish License Plate Dataset](https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset/data) ile eğitilmiştir.

## İçerik

- `split_dataset.py`: Verileri eğitim (%80) ve doğrulama (%20) olarak ayırır, YOLO formatına uygun şekilde `images/` ve `labels/` klasörlerine yerleştirir.
- `train_yolo.py`: YOLO11n modelini `dataset/data.yaml` dosyasına göre eğitir.
- `main.py`: Plaka tanıma sisteminin çalışacağı dosya.
- `dataset/`: Eğitim ve doğrulama verilerinin bulunduğu klasör (script tarafından otomatik oluşturulur).
- `dataset/data.yaml`: YOLO config dosyası (script tarafından oluşturulur).


## Veri Seti

Veri seti Kaggle üzerinden temin edilmiştir:  
📎 [https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset/data](https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset/data)

## Eğitim

Modeli eğitmek için aşağıdaki adımları izleyin:

```bash
pip install ultralytics
pip install easyocr
pip install opencv-python-headless
python split_dataset.py
python create_data_yaml.py
python train_yolo.py
python main.py
```
