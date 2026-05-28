import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QPoint, QRect

class ResizableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent # 부모 윈도우 참조
        self.begin = QPoint()
        self.end = QPoint()
        self.is_drawing = False
        
        # ⭐ 이제 화면 좌표 대신 '원본 이미지 기준'의 사각형(QRect)을 저장합니다.
        self.orig_image_rect = QRect() 

    def get_image_render_rect(self):
        """현재 라벨 안에서 실제로 이미지가 그려져 있는 정확한 4각 영역(좌표와 크기)을 계산"""
        if not self.pixmap() or self.pixmap().isNull():
            return QRect()
            
        # 전체 라벨 크기 내에서 실제 이미지 픽스맵이 배치된 시작점 계산 (가운데 정렬 기준)
        pix_w = self.pixmap().width()
        pix_h = self.pixmap().height()
        start_x = (self.width() - pix_w) // 2
        start_y = (self.height() - pix_h) // 2
        
        return QRect(start_x, start_y, pix_w, pix_h)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.main_window.cv_image is not None:
            render_rect = self.get_image_render_rect()
            pos = event.position().toPoint()
            
            # 마우스 클릭 위치가 실제 이미지 영역 내부일 때만 드래그 시작
            if render_rect.contains(pos):
                self.begin = pos
                self.end = pos
                self.is_drawing = True
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            render_rect = self.get_image_render_rect()
            pos = event.position().toPoint()
            
            # 마우스가 이미지 영역을 벗어나지 않도록 강제 제한(클리핑)
            x = max(render_rect.left(), min(pos.x(), render_rect.right()))
            y = max(render_rect.top(), min(pos.y(), render_rect.bottom()))
            
            self.end = QPoint(x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            
            render_rect = self.get_image_render_rect()
            current_screen_rect = QRect(self.begin, self.end).normalized()
            
            if current_screen_rect.width() > 5 and current_screen_rect.height() > 5:
                # ⭐ [핵심] 화면 좌표를 원본 OpenCV 이미지 크기 기준의 좌표로 역산(변환)
                orig_h, orig_w = self.main_window.cv_image.shape[:2]
                
                # 이미지 내부에서의 상대적 마우스 위치 계산
                rel_x1 = current_screen_rect.x() - render_rect.x()
                rel_y1 = current_screen_rect.y() - render_rect.y()
                
                scale_x = orig_w / render_rect.width()
                scale_y = orig_h / render_rect.height()
                
                orig_x = int(rel_x1 * scale_x)
                orig_y = int(rel_y1 * scale_y)
                orig_w_size = int(current_screen_rect.width() * scale_x)
                orig_h_size = int(current_screen_rect.height() * scale_y)
                
                self.orig_image_rect = QRect(orig_x, orig_y, orig_w_size, orig_h_size)
                print(f"🎯 원본 이미지 기준 고정 좌표 저장 완료: X={orig_x}, Y={orig_y}, W={orig_w_size}, H={orig_h_size}")
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.main_window.cv_image is None:
            return
            
        render_rect = self.get_image_render_rect()
        painter = QPainter(self)
        pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        if self.is_drawing:
            # 드래그 중일 때는 현재 마우스 화면 좌표 기준으로 실시간 그림
            rect = QRect(self.begin, self.end).normalized()
            painter.drawRect(rect)
        elif not self.orig_image_rect.isEmpty():
            # ⭐ 드래그가 끝나서 고정되었거나 창 크기가 바뀔 때는, 
            # 저장해 둔 '원본 좌표'를 '현재 늘어난 화면 크기'에 맞춰 다시 변환해서 그림!
            orig_h, orig_w = self.main_window.cv_image.shape[:2]
            
            scale_x = render_rect.width() / orig_w
            scale_y = render_rect.height() / orig_h
            
            screen_x = render_rect.x() + int(self.orig_image_rect.x() * scale_x)
            screen_y = render_rect.y() + int(self.orig_image_rect.y() * scale_y)
            screen_w = int(self.orig_image_rect.width() * scale_x)
            screen_h = int(self.orig_image_rect.height() * scale_y)
            
            painter.drawRect(QRect(screen_x, screen_y, screen_w, screen_h))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision OCR Inspector")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # 커스텀 라벨에 부모(self)를 넘겨주어 OpenCV 이미지에 접근할 수 있게 합니다.
        self.image_label = ResizableImageLabel(self)
        self.image_label.setText("여기에 이미지가 표시됩니다.\n아래 버튼을 눌러 이미지를 불러오세요.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #aaa; background-color: #f9f9f9;")
        self.image_label.setMinimumSize(400, 300) 
        layout.addWidget(self.image_label)
        
        self.btn_load = QPushButton("이미지 불러오기")
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.cv_image = None
        self.orig_pixmap = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if file_path:
            import numpy as np
            try:
                file_bytes = np.fromfile(file_path, np.uint8)
                self.cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                self.image_label.setText(f"이미지 로딩 중 오류 발생: {e}")
                return
            
            if self.cv_image is not None:
                rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.orig_pixmap = QPixmap.fromImage(q_img)
                
                self.image_label.orig_image_rect = QRect() # 사각형 초기화
                self.update_image_display()

    def update_image_display(self):
        if self.orig_pixmap is not None:
            scaled_pixmap = self.orig_pixmap.scaled(
                self.image_label.width() - 20, 
                self.image_label.height() - 20, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        if self.orig_pixmap is not None:
            self.update_image_display()
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())