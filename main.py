import sys
import os
import cv2
import difflib
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from paddleocr import PaddleOCR

# 💡 최신 PaddlePaddle / PaddleX oneDNN 가속 버그 완벽 차단 플래그
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

class ScalableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.orig_pixmap = None
        self.start_pos = None
        self.end_pos = None
        self.is_drawing = False
        self.orig_image_rect = QRect()

    def set_opencv_image(self, cv_img):
        # OpenCV BGR -> QImage RGB 변환 후 픽스맵 저장
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.orig_pixmap = QPixmap.fromImage(q_img)
        self.update_pixmap()
        
    def update_pixmap(self):
        if self.orig_pixmap is None:
            return

        scaled_pixmap = self.orig_pixmap.scaled(
            self.width() - 20, 
            self.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        if self.orig_pixmap is not None:
            self.update_pixmap()
        super().resizeEvent(event)

    def get_scale_factors(self):
        if not self.orig_pixmap or not self.pixmap():
            return 1.0, 1.0, 0, 0
        
        orig_w = self.orig_pixmap.width()
        orig_h = self.orig_pixmap.height()
        disp_w = self.pixmap().width()
        disp_h = self.pixmap().height()
        
        offset_x = (self.width() - disp_w) // 2
        offset_y = (self.height() - disp_h) // 2
        
        scale_x = orig_w / disp_w
        scale_y = orig_h / disp_h
        return scale_x, scale_y, offset_x, offset_y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.end_pos = event.pos()
            self.is_drawing = False
            self.calculate_orig_rect()
            self.update()

    def calculate_orig_rect(self):
        if not self.start_pos or not self.end_pos:
            return
        
        scale_x, scale_y, offset_x, offset_y = self.get_scale_factors()
        
        # 캔버스 래핑 오프셋 보정
        x1 = self.start_pos.x() - offset_x
        y1 = self.start_pos.y() - offset_y
        x2 = self.end_pos.x() - offset_x
        y2 = self.end_pos.y() - offset_y
        
        # 정규화된 상자 구하기
        rect = QRect(x1, y1, x2 - x1, y2 - y1).normalized()
        
        # 원본 해상도 좌표로 역산
        orig_x = int(rect.x() * scale_x)
        orig_y = int(rect.y() * scale_y)
        orig_w = int(rect.width() * scale_x)
        orig_h = int(rect.height() * scale_y)
        
        self.orig_image_rect = QRect(orig_x, orig_y, orig_w, orig_h)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_pos and self.end_pos:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_pos, self.end_pos).normalized()
            painter.drawRect(rect)


class DigimonInspectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("디지몬 스크린샷 멀티 인스펙터")
        self.setGeometry(100, 100, 1200, 700)

        # 데이터 보관용 멀티 리스트 (배열)
        self.ocr_zones = []  # 구조: [{'rect': (x,y,w,h), 'text': '성숙기'}]

        # AI 엔진 초기화 (회전 완벽 차단 옵션 고정)
        print("🤖 PaddleOCR 엔진 초기화 중...")
        self.ocr_engine = PaddleOCR(
            lang='korean', enable_mkldnn=False,
            use_angle_cls=False, use_doc_orientation_classify=False
        )
        print("🤖 PaddleOCR 준비 완료!")

        self.cv_image = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 왼쪽: 이미지 뷰어 레이아웃
        left_layout = QVBoxLayout()
        self.image_label = ScalableImageLabel()
        self.image_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label, stretch=4)

        # 컨트롤 버튼 구역
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("🖼️ 스크린샷 불러오기")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_add_zone = QPushButton("🎯 드래그 구역 리스트에 추가")
        self.btn_add_zone.clicked.connect(self.ocr_and_add_list)
        self.btn_clear = QPushButton("🗑️ 전체 비우기")
        self.btn_clear.clicked.connect(self.clear_all_data)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_add_zone)
        btn_layout.addWidget(self.btn_clear)
        left_layout.addLayout(btn_layout)
        
        main_layout.addLayout(left_layout, stretch=3)

        # 오른쪽: 그리드(표) 레이아웃
        right_layout = QVBoxLayout()
        
        lbl_table_title = QLabel("📋 OCR 구역별 매칭 리스트 (그리드)")
        lbl_table_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        right_layout.addWidget(lbl_table_title)

        # QTableWidget을 이용하여 그리드 표 구성
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["No.", "자른 좌표 (X, Y, W, H)", "OCR 판정 결과"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers) # 수정 금지
        
        right_layout.addWidget(self.table_widget)
        
        # 임시 Supabase 연동용 버튼 (자리 배치)
        self.btn_submit_db = QPushButton("🚀 이 그리드 리스트 전체를 Supabase DB로 전송")
        self.btn_submit_db.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 35px;")
        right_layout.addWidget(self.btn_submit_db)

        main_layout.addLayout(right_layout, stretch=2)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            file_bytes = np.fromfile(file_path, np.uint8)
            self.cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            self.image_label.set_opencv_image(self.cv_image)
            self.clear_all_data()

    def ocr_and_add_list(self):
        rect = self.image_label.orig_image_rect
        if self.cv_image is None or rect.isEmpty():
            return

        # 1. 안전 마진 적용 및 자르기
        orig_h, orig_w = self.cv_image.shape[:2]
        padding = 10
        x1 = max(0, rect.x() - padding)
        y1 = max(0, rect.y() - padding)
        x2 = min(orig_w, rect.x() + rect.width() + padding)
        y2 = min(orig_h, rect.y() + rect.height() + padding)
        
        crop_img = self.cv_image[y1:y2, x1:x2]
        if crop_img.size == 0: return

        # 2. 고화질 보정 전처리
        crop_img = cv2.resize(crop_img, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(binary, kernel, iterations=1)
        crop_img = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)

        text_outputs = []
        try:
            # 3. OCR 실행
            ocr_results = self.ocr_engine.predict(crop_img)
            if ocr_results:
                for res in ocr_results:
                    if isinstance(res, dict) and 'rec_texts' in res:
                        text_outputs.extend(res['rec_texts'])
                    elif hasattr(res, 'rec_texts'):
                        text_outputs.extend(getattr(res, 'rec_texts', []))
            
            raw_text = " ".join(text_outputs).strip()
            
            # 공백을 제거하고 순수 글자만 비교
            clean_text = raw_text.replace(" ", "")
            
            # 1단계 꼼수: '유년기'라는 단어가 포착되었다면 뒤에 붙은 잔상으로 강제 판정
            if "유년기" in clean_text:
                # '유년기' 텍스트 뒤에 붙은 글자들을 추출 (예: '유년기||' -> '||')
                suffix = clean_text.split("유년기")[-1]
                
                if len(suffix) >= 2:  # 뒤에 2글자 이상 붙어있으면 (II, 11, ll 등)
                    final_text = "유년기2"
                    print(f"🔮 유년기 꼼수 알고리즘 발동: '{raw_text}' ──> '유년기2'")
                else:  # 뒤에 1글자만 붙어있거나 없으면 (I, 1, l 등)
                    final_text = "유년기1"
                    print(f"🔮 유년기 꼼수 알고리즘 발동: '{raw_text}' ──> '유년기1'")
            else:
                # 4. 💡 진화단계 및 속성 타겟 마스터 사전 자동 보정
                DIGIMON_STAGE_DICT = ["스테이터스", "유년기1", "유년기2", "성장기", "성숙기", "완전체", "궁극체", "초궁극체", "백신", "데이터", "바이러스", "프리", "NO DATA", " 배리어블"]
                
                if raw_text in DIGIMON_STAGE_DICT:
                    final_text = raw_text
                else:
                    close_matches = difflib.get_close_matches(raw_text, DIGIMON_STAGE_DICT, n=1, cutoff=0.3)
                    final_text = close_matches[0] if close_matches else raw_text

        except Exception as e:
            print(f"❌ OCR 에러: {e}")
            final_text = "(인식 에러)"

        if not final_text: final_text = "(글자 없음)"

        # 5. 내부 리스트 배열에 누적 저장
        zone_info = {
            'rect': (rect.x(), rect.y(), rect.width(), rect.height()),
            'text': final_text
        }
        self.ocr_zones.append(zone_info)

        # 6. 그리드(표) UI 상에 행 추가 반영
        self.refresh_table_ui()

    def refresh_table_ui(self):
        self.table_widget.setRowCount(0) # 리셋 후 다시 그리기
        for idx, zone in enumerate(self.ocr_zones):
            row = self.table_widget.rowCount()
            self.table_widget.insertRow(row)
            
            # 컬럼 삽입
            self.table_widget.setItem(row, 0, QTableWidgetItem(str(idx + 1)))
            self.table_widget.setItem(row, 1, QTableWidgetItem(str(zone['rect'])))
            
            # 결과 아이템 강조 스타일 적용
            res_item = QTableWidgetItem(zone['text'])
            res_item.setTextAlignment(Qt.AlignCenter)
            res_item.setForeground(QColor(0, 102, 204)) # 파란색 글씨 강조
            self.table_widget.setItem(row, 2, res_item)

    def clear_all_data(self):
        self.ocr_zones.clear()
        self.table_widget.setRowCount(0)
        print("🗑️ 모든 등록 구역 데이터가 초기화되었습니다.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigimonInspectorWindow()
    window.show()
    sys.exit(app.exec_())