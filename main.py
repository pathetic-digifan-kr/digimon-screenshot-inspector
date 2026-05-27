import sys
import cv2  # 이미지 처리를 위한 OpenCV
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Vision OCR Inspector")
        self.resize(800, 600)  # 이미지를 봐야 하므로 창 크기를 조금 키웠습니다.
        
        layout = QVBoxLayout()
        
        # 1. 이미지가 표시될 라벨 (처음에는 안내 문구)
        self.image_label = QLabel("여기에 이미지가 표시됩니다.\n아래 버튼을 눌러 이미지를 불러오세요.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 이미지 크기가 라벨보다 크면 자동으로 맞춰지도록 설정
        self.image_label.setStyleSheet("border: 1px dashed #aaa; background-color: #f9f9f9;")
        layout.addWidget(self.image_label)
        
        # 2. 이미지 불러오기 버튼
        self.btn_load = QPushButton("이미지 불러오기")
        # 버튼을 누르면 self.load_image 함수가 실행되도록 연결 (시그널-슬롯)
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # 원본 이미지를 보관할 변수
        self.cv_image = None

    def load_image(self):
        # 파일 탐색기 열기 (이미지 파일만 필터링)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        
        if file_path:
            import numpy as np  # 한글 경로 우회를 위해 넘파이 임포트
            
            try:
                # ❌ 기존 방식 (한글 경로에서 에러 발생):
                # self.cv_image = cv2.imread(file_path)
                
                # ⭕ 해결 방식 (한글 경로 완벽 지원):
                # 1. 파일 데이터를 바이트 배열로 읽어옴
                file_bytes = np.fromfile(file_path, np.uint8)
                # 2. 바이트 배열을 OpenCV 이미지 객체로 디코딩
                self.cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
            except Exception as e:
                self.image_label.setText(f"이미지 로딩 중 오류 발생: {e}")
                return
            
            if self.cv_image is not None:
                # 2. OpenCV(BGR 구조) 데이터를 PySide(RGB 구조)에 맞게 변환
                rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # QImage 객체 생성
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # 3. 화면에 띄우기 위해 QPixmap으로 변환 및 라벨에 세팅
                pixmap = QPixmap.fromImage(q_img)
                
                # 라벨 크기에 맞게 이미지 축소/확대 (비율 유지)
                scaled_pixmap = pixmap.scaled(
                    self.image_label.width() - 20, 
                    self.image_label.height() - 20, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("이미지를 읽어오는데 실패했습니다. (경로 확인 필요)")
            
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())