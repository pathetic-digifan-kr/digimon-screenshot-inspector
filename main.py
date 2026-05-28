import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Vision OCR Inspector")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # 1. 이미지가 표시될 라벨
        self.image_label = QLabel("여기에 이미지가 표시됩니다.\n아래 버튼을 눌러 이미지를 불러오세요.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #aaa; background-color: #f9f9f9;")
        
        # ⭐ 중요: 라벨의 최소 크기를 설정하여 창이 줄어들 때 라벨이 뭉개지는 것을 방지
        self.image_label.setMinimumSize(400, 300) 
        layout.addWidget(self.image_label)
        
        # 2. 이미지 불러오기 버튼
        self.btn_load = QPushButton("이미지 불러오기")
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # 원본 OpenCV 이미지와 원본 PySide QPixmap을 보관할 변수
        self.cv_image = None
        self.orig_pixmap = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        
        if file_path:
            try:
                # 한글 경로 완벽 지원 방식으로 이미지 읽기
                file_bytes = np.fromfile(file_path, np.uint8)
                self.cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                self.image_label.setText(f"이미지 로딩 중 오류 발생: {e}")
                return
            
            if self.cv_image is not None:
                # BGR -> RGB 변환
                rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # ⭐ 나중에 창 크기가 바뀔 때 재사용하기 위해 원본 픽스맵을 변수에 저장해 둡니다.
                self.orig_pixmap = QPixmap.fromImage(q_img)
                
                # 이미지를 화면 크기에 맞게 업데이트하는 별도 함수 호출
                self.update_image_display()
            else:
                self.image_label.setText("이미지를 읽어오는데 실패했습니다.")

    def update_image_display(self):
        """현재 라벨 크기에 맞춰 이미지를 축소/확대하여 보여주는 함수"""
        if self.orig_pixmap is not None:
            # 여백을 약간 주기 위해 라벨 크기보다 20픽셀 작게 픽스맵 조절
            scaled_pixmap = self.orig_pixmap.scaled(
                self.image_label.width() - 20, 
                self.image_label.height() - 20, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation  # 안티앨리어싱 적용으로 부드럽게 확대/축소
            )
            self.image_label.setPixmap(scaled_pixmap)

    # ⭐ PySide6의 내장 이벤트 오버라이딩 (창 크기가 바뀔 때마다 자동 실행됨)
    def resizeEvent(self, event):
        # 창 크기가 바뀔 때 이미지가 로드된 상태라면 화면 표시를 실시간 업데이트
        if self.orig_pixmap is not None:
            self.update_image_display()
        # 원래 부모 클래스의 resizeEvent 처리를 그대로 이어받음 (필수)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())