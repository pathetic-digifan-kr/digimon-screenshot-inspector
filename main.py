import sys
import os

# 💡 PaddleOCR 특유의 윈도우 OpenMP 충돌 에러 방지용 환경 변수 세팅 (필수)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 💡 [추가] 인텔 oneDNN CPU 가속 버그 우회를 위해 가속 기능을 강제로 끕니다.
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"  # 👈 최신 PaddleX 버그 브레이커 플래그

import cv2
import numpy as np
from paddleocr import PaddleOCR  # 👈 PaddleOCR 임포트

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QFileDialog)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QPoint, QRect


# 1. 마우스 드래그 및 반응형 사각형 렌더링이 가능한 커스텀 라벨 클래스
class ResizableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.begin = QPoint()
        self.end = QPoint()
        self.is_drawing = False
        self.orig_image_rect = QRect() 

    def get_image_render_rect(self):
        if not self.pixmap() or self.pixmap().isNull():
            return QRect()
        pix_w = self.pixmap().width()
        pix_h = self.pixmap().height()
        start_x = (self.width() - pix_w) // 2
        start_y = (self.height() - pix_h) // 2
        return QRect(start_x, start_y, pix_w, pix_h)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.main_window.cv_image is not None:
            render_rect = self.get_image_render_rect()
            pos = event.position().toPoint()
            if render_rect.contains(pos):
                self.begin = pos
                self.end = pos
                self.is_drawing = True
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            render_rect = self.get_image_render_rect()
            pos = event.position().toPoint()
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
                # 화면 좌표 -> 원본 이미지 기준 고정 좌표 역산
                orig_h, orig_w = self.main_window.cv_image.shape[:2]
                rel_x1 = current_screen_rect.x() - render_rect.x()
                rel_y1 = current_screen_rect.y() - render_rect.y()
                
                scale_x = orig_w / render_rect.width()
                scale_y = orig_h / render_rect.height()
                
                orig_x = int(rel_x1 * scale_x)
                orig_y = int(rel_y1 * scale_y)
                orig_w_size = int(current_screen_rect.width() * scale_x)
                orig_h_size = int(current_screen_rect.height() * scale_y)
                
                self.orig_image_rect = QRect(orig_x, orig_y, orig_w_size, orig_h_size)
                print(f"🎯 영역 선택 완료 (원본 기준): X={orig_x}, Y={orig_y}, W={orig_w_size}, H={orig_h_size}")
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
            rect = QRect(self.begin, self.end).normalized()
            painter.drawRect(rect)
        elif not self.orig_image_rect.isEmpty():
            orig_h, orig_w = self.main_window.cv_image.shape[:2]
            scale_x = render_rect.width() / orig_w
            scale_y = render_rect.height() / orig_h
            
            screen_x = render_rect.x() + int(self.orig_image_rect.x() * scale_x)
            screen_y = render_rect.y() + int(self.orig_image_rect.y() * scale_y)
            screen_w = int(self.orig_image_rect.width() * scale_x)
            screen_h = int(self.orig_image_rect.height() * scale_y)
            
            painter.drawRect(QRect(screen_x, screen_y, screen_w, screen_h))


# 2. 메인 윈도우 클래스
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision OCR Inspector (PaddleOCR)")
        self.resize(1000, 600)
        
        main_layout = QHBoxLayout()
        
        # ================= [왼쪽 레이아웃] =================
        left_layout = QVBoxLayout()
        
        self.image_label = ResizableImageLabel(self)
        self.image_label.setText("여기에 이미지가 표시됩니다.\n아래 버튼을 눌러 이미지를 불러오세요.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #aaa; background-color: #f9f9f9;")
        self.image_label.setMinimumSize(500, 400) 
        left_layout.addWidget(self.image_label)
        
        self.btn_load = QPushButton("이미지 불러오기")
        self.btn_load.clicked.connect(self.load_image)
        left_layout.addWidget(self.btn_load)
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, stretch=3)
        
        # ================= [오른쪽 레이아웃] =================
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        panel_title = QLabel("=== 검사 항목 설정 ===")
        panel_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(panel_title)
        
        mark_row_layout = QHBoxLayout()
        
        self.btn_register_mark = QPushButton("식별 마크 등록")
        self.btn_register_mark.clicked.connect(self.register_and_ocr_mark)
        mark_row_layout.addWidget(self.btn_register_mark)
        
        self.lbl_mark_result = QLabel("읽은 글자: (대기 중)")
        self.lbl_mark_result.setStyleSheet("color: #0055ff; font-weight: bold; padding-left: 10px;")
        mark_row_layout.addWidget(self.lbl_mark_result)
        
        mark_row_widget = QWidget()
        mark_row_widget.setLayout(mark_row_layout)
        right_layout.addWidget(mark_row_widget)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setStyleSheet("background-color: #f0f0f0; border-left: 1px solid #ccc;")
        main_layout.addWidget(right_widget, stretch=1)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # 이미지 데이터 변수
        self.cv_image = None
        self.orig_pixmap = None
        
        # 내부 저장용 변수
        self.saved_mark_rect = None
        self.saved_mark_text = ""
        
        # 👑 PaddleOCR 엔진 최초 초기화 (한글/영어 동시 지원 모드)
        print("🤖 PaddleOCR 엔진 초기화 중...")
        # lang='korean'으로 지정하면 기본적으로 영어도 내장해서 함께 읽어옵니다.
        self.ocr_engine = PaddleOCR(
            lang='korean', 
            enable_mkldnn=False,
            use_textline_orientation=False,  # 👈 회전 검사 원천 차단!
            use_doc_orientation_classify=False,
        )
        print("🤖 PaddleOCR 준비 완료!")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if file_path:
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
                
                self.image_label.orig_image_rect = QRect() 
                self.lbl_mark_result.setText("읽은 글자: (대기 중)")
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

    def register_and_ocr_mark(self):
        rect = self.image_label.orig_image_rect
        
        if self.cv_image is None or rect.isEmpty():
            self.lbl_mark_result.setText("읽은 글자: 이미지나 영역을 확인하세요.")
            return
            
        # 1. 안전 마진(Padding) 자동 부여
        orig_h, orig_w = self.cv_image.shape[:2]
        padding = 10
        
        x1 = max(0, rect.x() - padding)
        y1 = max(0, rect.y() - padding)
        x2 = min(orig_w, rect.x() + rect.width() + padding)
        y2 = min(orig_h, rect.y() + rect.height() + padding)
        
        crop_img = self.cv_image[y1:y2, x1:x2]
        
        if crop_img.size == 0:
            return
            
        # 2. 💡 [강력 전처리] 3배 확대 + 자모음 분리를 위한 침식 연산
        crop_img = cv2.resize(crop_img, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # ⭐ [핵심 해결책] 1채널 흑백 이미지를 PaddleOCR이 좋아하는 3채널(RGB) 형태로 복원 가공!
        crop_img = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
            
        # 디버깅용 파일 세이브
        cv2.imwrite("debug_crop.png", crop_img)
            
        self.lbl_mark_result.setText("읽은 글자: OCR 인식 중...")
        QApplication.processEvents() 
        
        text_outputs = []
        try:
            # 3. PaddleOCR 실행 (회전 방지 플래그 상시 유지)
            ocr_results = self.ocr_engine.predict(crop_img)
            
            # 4. 인덱스 에러를 방지하는 철통 방어 파싱
            if ocr_results:
                for res in ocr_results:
                    if isinstance(res, dict):
                        if 'rec_texts' in res and res['rec_texts']:
                            text_outputs.extend(res['rec_texts'])
                    elif hasattr(res, 'rec_texts'):
                        texts = getattr(res, 'rec_texts', [])
                        if texts:
                            text_outputs.extend(texts)
            
            final_text = " ".join(text_outputs).strip()
            
        except Exception as e:
            print(f"❌ OCR 에러: {e}")
            final_text = "(엔진 오류)"
        
        if not final_text:
            final_text = "(글자 인식 실패)"
            
        # 5. 내부 변수 보관 및 UI 반영
        self.saved_mark_rect = (rect.x(), rect.y(), rect.width(), rect.height())
        self.saved_mark_text = final_text
        self.lbl_mark_result.setText(f"읽은 글자: {final_text}")
        print(f"🔒 마크 등록 완료: '{final_text}'")

    def resizeEvent(self, event):
        if self.orig_pixmap is not None:
            self.update_image_display()
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())