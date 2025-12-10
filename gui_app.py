import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QFrame,
)


# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def np_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """
    OpenCV (BGR, uint8) görüntüsünü Qt'ye uygun QPixmap'e çevirir.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def count_classes(result) -> str:
    """
    YOLO result objesinden sınıf sayımlarını (cat/dog) çıkarır,
    GUI'de göstermek için string döner.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return "Hiç nesne tespit edilmedi."

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    counts = {}
    for cid in cls_ids:
        cname = names.get(cid, str(cid))
        counts[cname] = counts.get(cname, 0) + 1

    parts = [f"{name}: {cnt}" for name, cnt in counts.items()]
    return ", ".join(parts)


# -----------------------------
# Ana Pencere
# -----------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv8 - Kedi / Köpek Nesne Tespiti (PyQt5 GUI)")
        self.resize(1200, 700)

        # Model dosyasını bul
        self.model_path = Path(__file__).with_name("best.pt")
        if not self.model_path.exists():
            QMessageBox.critical(
                self,
                "Model Bulunamadı",
                f"'best.pt' dosyası bulunamadı.\n\nBeklenen yol:\n{self.model_path}",
            )
            sys.exit(1)

        # YOLO modelini yükle
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Model Yükleme Hatası",
                f"best.pt yüklenirken hata oluştu:\n{e}",
            )
            sys.exit(1)

        # Seçilen görüntü ve son tahmin
        self.current_image_path: Path | None = None
        self.last_result_image: np.ndarray | None = None

        self._build_ui()

    def _build_ui(self):
        # ---------- Sol: Original Image Panel ----------
        self.lbl_original = QLabel("Original Image")
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setFrameShape(QFrame.Box)
        self.lbl_original.setStyleSheet(
            "background-color: #222; color: white; font-size: 14px;"
        )

        # ---------- Sağ: Tagged Image Panel ----------
        self.lbl_tagged = QLabel("Tagged Image")
        self.lbl_tagged.setAlignment(Qt.AlignCenter)
        self.lbl_tagged.setFrameShape(QFrame.Box)
        self.lbl_tagged.setStyleSheet(
            "background-color: #222; color: white; font-size: 14px;"
        )

        # İki paneli yan yana koy
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.lbl_original, stretch=1)
        images_layout.addWidget(self.lbl_tagged, stretch=1)

        # ---------- Alt: Bilgi etiketi ----------
        self.lbl_info = QLabel("Hazır: Lütfen bir görüntü seçin.")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 13px; padding: 6px;")

        # ---------- Butonlar ----------
        btn_select = QPushButton("Select Image")
        btn_select.clicked.connect(self.on_select_image)

        btn_test = QPushButton("Test Image")
        btn_test.clicked.connect(self.on_test_image)

        btn_save = QPushButton("Save Image")
        btn_save.clicked.connect(self.on_save_image)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(btn_select)
        buttons_layout.addWidget(btn_test)
        buttons_layout.addWidget(btn_save)
        buttons_layout.addStretch(1)

        # ---------- Ana layout ----------
        main_layout = QVBoxLayout()
        main_layout.addLayout(images_layout, stretch=1)
        main_layout.addWidget(self.lbl_info)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    # -----------------------------
    # Slot fonksiyonları
    # -----------------------------
    def on_select_image(self):
        # Kullanıcıdan dosya seçmesini iste
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Bir görüntü seçin",
            str(Path.cwd()),
            "Görüntü Dosyaları (*.jpg *.jpeg *.png *.bmp)",
        )

        if not file_path:
            return

        self.current_image_path = Path(file_path)
        self.lbl_info.setText(f"Seçilen görüntü: {self.current_image_path.name}")

        # OpenCV ile oku ve sol panelde göster
        img = cv2.imread(str(self.current_image_path))
        if img is None:
            QMessageBox.warning(
                self,
                "Okuma Hatası",
                f"Görüntü okunamadı:\n{self.current_image_path}",
            )
            return

        pix = np_to_qpixmap(img)
        self._set_pixmap_scaled(self.lbl_original, pix)
        self.lbl_tagged.setText("Tagged Image")
        self.last_result_image = None

    def on_test_image(self):
        if self.current_image_path is None:
            QMessageBox.information(
                self,
                "Bilgi",
                "Önce 'Select Image' ile bir görüntü seçmelisiniz.",
            )
            return

        self.lbl_info.setText("Model çalışıyor, lütfen bekleyin...")

        # YOLO ile tahmin
        try:
            results = self.model(str(self.current_image_path))
            result = results[0]
        except Exception as e:
            QMessageBox.critical(
                self,
                "Tahmin Hatası",
                f"Model tahmin yaparken hata oluştu:\n{e}",
            )
            self.lbl_info.setText("Hata oluştu.")
            return

        # bounding box çizilmiş görüntü (BGR)
        im_bgr = result.plot()  # numpy array (BGR)
        self.last_result_image = im_bgr.copy()

        # sağ panele göster
        pix = np_to_qpixmap(im_bgr)
        self._set_pixmap_scaled(self.lbl_tagged, pix)

        # sınıf sayımlarını göster
        info_text = count_classes(result)
        self.lbl_info.setText(f"Tespit Sonucu: {info_text}")

    def on_save_image(self):
        if self.last_result_image is None:
            QMessageBox.information(
                self,
                "Bilgi",
                "Önce 'Test Image' ile bir tahmin çalıştırmalısınız.",
            )
            return

        default_name = "result_" + (
            self.current_image_path.name if self.current_image_path else "output.jpg"
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Sonucu Kaydet",
            str(Path.cwd() / default_name),
            "Görüntü Dosyaları (*.jpg *.png *.bmp)",
        )

        if not save_path:
            return

        # BGR görüntüyü kaydet
        try:
            cv2.imwrite(save_path, self.last_result_image)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Kaydetme Hatası",
                f"Görüntü kaydedilemedi:\n{e}",
            )
            return

        QMessageBox.information(
            self,
            "Kaydedildi",
            f"Görüntü başarıyla kaydedildi:\n{save_path}",
        )

    # -----------------------------
    # Yardımcı: QLabel içine resmi sığdır
    # -----------------------------
    def _set_pixmap_scaled(self, label: QLabel, pixmap: QPixmap):
        if pixmap.isNull():
            return
        target_size = label.size()
        if target_size.width() == 0 or target_size.height() == 0:
            # İlk açılışta label daha çizilmemiş olabilir, o yüzden sabit bir boyuta göre ölçekle
            target_size = label.maximumSize()
        scaled = pixmap.scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)


# -----------------------------
# Uygulama girişi
# -----------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
