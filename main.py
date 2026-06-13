import sys
import os
import re
import cv2
import difflib
import numpy as np
from digimon_db import digimon_db
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QComboBox, QSizePolicy, QLineEdit,
                             QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QRect, QRectF, QPoint, QPointF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from paddleocr import PaddleOCR

import paddle

# 💡 최신 PaddlePaddle / PaddleX oneDNN 가속 버그 완벽 차단 플래그
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

print("CUDA 지원:", paddle.is_compiled_with_cuda())
print("장치:", paddle.device.get_device())

class ScalableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setMouseTracking(True)

        self.start_scale_x = 1
        self.start_scale_y = 1
        self.end_scale_x = 1
        self.end_scale_y = 1

        self.is_drawing = False
        self.scale_rect = QRectF()
        self.org_image = None
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def set_opencv_image(self, cv_img:cv2.Mat):
        # OpenCV BGR -> QImage RGB 변환 후 픽스맵 저장
        self.org_image = cv_img
        self.update_image()
    
    def update_image(self):
        if self.org_image is None:
            return
        rect = self.contentsRect()
        resize_image = cv2.resize(self.org_image, (rect.width(), rect.height()), interpolation=cv2.INTER_LANCZOS4)
        h, w, ch = resize_image.shape
        byte_per_line = ch * w

        image = QImage(
            resize_image.data,
            w,
            h,
            byte_per_line,
            QImage.Format_BGR888
        )
        self.setPixmap(QPixmap.fromImage(image))

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.position()
            self.start_scale_x = event.position().x() / self.width()
            self.start_scale_y = event.position().y() / self.height()

            self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_pos = event.position()
            self.end_scale_x = event.position().x() / self.width()
            self.end_scale_y = event.position().y() / self.height()

            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.end_pos = event.position()
            self.end_scale_x = event.position().x() / self.width()
            self.end_scale_y = event.position().y() / self.height()

            self.is_drawing = False

            leftTop = QPointF(self.start_scale_x, self.start_scale_y)
            rightBottom = QPointF(self.end_scale_x, self.end_scale_y)

            self.scale_rect = QRectF(leftTop, rightBottom)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_scale_x != self.end_scale_x and self.start_scale_y != self.end_scale_y:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)

            left = self.start_scale_x * self.width()
            top = self.start_scale_y * self.height()
            right = self.end_scale_x * self.width()
            bottom = self.end_scale_y * self.height()
            
            leftTop = QPoint(left, top)
            rightBottom = QPoint(right, bottom)

            rect = QRect(leftTop, rightBottom)
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
        self.identifier_roi = QRectF(0,0 ,0 ,0)
        self.lbl_identifier = QLabel()
        self.edit_directory = QLineEdit()

        self.digimon_db = digimon_db()
        self.digimon_db.connect()

        self.load_generations()
        self.load_attributes()

        self.init_ui()

    def load_generations(self):
        locale = "ko"
        self.generations = self.digimon_db.load_generations(locale)
        self.generations = {gen['name']: gen['id'] for gen in self.generations}
        print(self.generations)
    
    def load_attributes(self):
        locale = "ko"
        self.attributes = self.digimon_db.load_attributes(locale)
        self.attributes = {attr['name']: attr['id'] for attr in self.attributes}
        print(self.attributes)

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

        # 오른쪽: 설정 레이아웃
        right_layout = QVBoxLayout()

        # 우측 상단 : 식별 레이아웃
        idenfier_layout = QHBoxLayout()
        lbl_identifier = QLabel("식별 영역")
        lbl_identifier.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        btn_identifier = QPushButton("등록")
        btn_identifier.clicked.connect(self.ocr_and_set_identifier)
        idenfier_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        idenfier_layout.addWidget(lbl_identifier)
        idenfier_layout.addWidget(btn_identifier)
        idenfier_layout.addWidget(self.lbl_identifier)
        right_layout.addLayout(idenfier_layout)
        
        # 우측 중단 : 테이블 레이아웃
        lbl_table_title = QLabel("📋 OCR 구역별 매칭 리스트 (그리드)")
        lbl_table_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        right_layout.addWidget(lbl_table_title)

        # QTableWidget을 이용하여 그리드 표 구성
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["No.", "DB 필드명", "자른 좌표 (X, Y, W, H)", "OCR 판정 결과"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        right_layout.addWidget(self.table_widget)

        # 탐색 디렉토리 레이아웃
        directory_layout = QHBoxLayout()
        btn_directory = QPushButton("디렉토리 선택")
        btn_directory.clicked.connect(self.select_directory)
        directory_layout.addWidget(self.edit_directory, stretch=4)
        directory_layout.addWidget(btn_directory, stretch=1)
        right_layout.addLayout(directory_layout)


        # 임시 Supabase 연동용 버튼 (자리 배치)
        self.btn_submit_db = QPushButton("🚀 이 그리드 리스트 전체를 Supabase DB로 전송")
        self.btn_submit_db.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 35px;")
        self.btn_submit_db.clicked.connect(self.debug_final_payload) 
        right_layout.addWidget(self.btn_submit_db)

        main_layout.addLayout(right_layout, stretch=2)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            file_bytes = np.fromfile(file_path, np.uint8)
            self.cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            self.image_label.set_opencv_image(self.cv_image)
            self.clear_all_data()

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "디렉토리 선택")
        if directory:
            self.edit_directory.setText(directory)
    
    def searchDirectory(self):
        dir = Path(self.edit_directory.text())
        pattern = re.compile(r"^(\d+)\.\s*(.+)")
        for digimon_dir in dir.iterdir():

            # 디렉토리 이름 형식 #.디지몬이름
            m = pattern.match(digimon_dir.name)
            if m:
                print(f'{m.group(1)}번 : {m.group(2)}')
                jpg_files = [
                    f for f in digimon_dir.iterdir()
                    if f.is_file() and f.suffix.lower() == '.jpg'
                ]

                # jpg 파일 목록 탐색
                for jpg in jpg_files:
                    if self.is_valid_image(jpg):
                        #self.save_webp(jpg, f"image/{m.group(1)}.webp")
                        ret = self.ocr_regions(jpg)
                        print(f"✅ {jpg.name} - OCR 결과: {ret}")

                        print(f"🔍 OCR로 추출된 진화단계: '{ret['generation']}' | 속성: '{ret['attribute']}'")

                        # 도감 이미지 저장
                        webp_name = f"{m.group(1)}.webp"
                        webp_full_name =f"image/{m.group(1)}.webp"
                        self.save_webp(jpg, webp_full_name)
                        
                        # # 도감 이미지 업로드
                        file_url  = self.digimon_db.upload_digimon_profile_image(webp_full_name, webp_name)
                        self.digimon_db.update_digimon_profile_image(int(m.group(1)), file_url)
                        self.digimon_db.insert_digimon_name_translation(int(m.group(1)), "ko", m.group(2))

                        self.digimon_db.update_digimon_profile_generation(int(m.group(1)), self.generations.get(ret['generation'], None))
                        self.digimon_db.update_digimon_profile_attribute(int(m.group(1)), self.attributes.get(ret['attribute'], None))
                        break
    
    def is_valid_image(self, image_file):
        file_bytes = np.fromfile(image_file, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        identifier = self.ocr(image, self.identifier_roi)

        return self.lbl_identifier.text() == identifier
    
    def save_webp(self, image_file, target_path):
        file_bytes = np.fromfile(image_file, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite(target_path, image, [cv2.IMWRITE_WEBP_QUALITY, 100])
        return target_path
    
    def ocr_regions(self, image_file):
        file_bytes = np.fromfile(image_file, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = {
            'generation': None,
            'attribute': None,
        }
        for idx, zone in enumerate(self.ocr_zones):
            field_name = zone['field']
            text = self.ocr(image, zone['rect'])
            result[field_name] = text
        return result

    def ocr(self, image:cv2.Mat, rect:QRectF):
        if image is None or rect.isEmpty():
            return
        # 1. 안전 마진 적용 및 자르기
        orig_h, orig_w = image.shape[:2]
        x1 = int(max(0, rect.x() * orig_w))
        y1 = int(max(0, rect.y() * orig_h ))
        x2 = int(min(orig_w, (rect.x() + rect.width()) * orig_w))
        y2 = int(min(orig_h, (rect.y() + rect.height()) * orig_h))
        
        crop_img = image[y1:y2, x1:x2]
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
                DIGIMON_STAGE_DICT = ["스테이터스", "유년기1", "유년기2", "성장기", "성숙기", "완전체", "궁극체", "초궁극체", "아머체", "하이브리드체", "백신", "데이터", "바이러스", "프리", "NO DATA", "배리어블", "불명"]
                
                if raw_text in DIGIMON_STAGE_DICT:
                    final_text = raw_text
                else:
                    close_matches = difflib.get_close_matches(raw_text, DIGIMON_STAGE_DICT, n=1, cutoff=0.3)
                    final_text = close_matches[0] if close_matches else raw_text

        except Exception as e:
            print(f"❌ OCR 에러: {e}")
            final_text = "(인식 에러)"

        if not final_text: final_text = "(글자 없음)"

        return final_text
    
    def ocr_and_set_identifier(self):
        rect = self.image_label.scale_rect
        if self.cv_image is None or rect.isEmpty():
            return
        
        text = self.ocr(self.cv_image, rect)

        self.identifier_roi = rect
        self.lbl_identifier.setText(text)

    def ocr_and_add_list(self):
        rect = self.image_label.scale_rect
        if self.cv_image is None or rect.isEmpty():
            return

        final_text = self.ocr(self.cv_image, rect)

        # 5. 내부 리스트 배열에 누적 저장
        zone_info = {
            'rect': rect,
            'text': final_text,
            'field': 'generation',
        }
        self.ocr_zones.append(zone_info)

        # 6. 그리드(표) UI 상에 행 추가 반영
        self.refresh_table_ui()

    def update_field_data(self, row, index):
        if 0 <= row < len(self.ocr_zones):
            field_mapping = {0: 'generation', 1: 'attribute', 2: 'etc_mark'}
            self.ocr_zones[row]['field'] = field_mapping.get(index, 'generation')
            print(f"🔄 Row {row+1} DB 필드명이 '{self.ocr_zones[row]['field']}'(으)로 변경되었습니다.")

    def refresh_table_ui(self):
        self.table_widget.setRowCount(0) # 리셋 후 다시 그리기
        for idx, zone in enumerate(self.ocr_zones):
            row = self.table_widget.rowCount()
            self.table_widget.insertRow(row)
            
            # 컬럼 삽입
            self.table_widget.setItem(row, 0, QTableWidgetItem(str(idx + 1)))

            # DB 필드명 선택용 콤보 박스
            combo = QComboBox()
            combo.addItems(["generation (진화단계)", "attribute (속성)", "etc_mark (기타마크)"])
            
            # 기존에 선택되어 있던 값이 있다면 인덱스 복원 맞춤
            if zone['field'] == 'generation': combo.setCurrentIndex(0)
            elif zone['field'] == 'attribute': combo.setCurrentIndex(1)
            elif zone['field'] == 'etc_mark': combo.setCurrentIndex(2)

            # 유저가 드롭다운을 바꿀 때마다 메모리(self.ocr_zones) 값도 실시간 싱크 동기화
            combo.currentIndexChanged.connect(lambda index, r=row: self.update_field_data(r, index))
            self.table_widget.setCellWidget(row, 1, combo)

            # 2번 열: 좌표
            self.table_widget.setItem(row, 2, QTableWidgetItem(str(zone['rect'])))
            
            # 결과 아이템 강조 스타일 적용
            res_item = QTableWidgetItem(zone['text'])
            res_item.setTextAlignment(Qt.AlignCenter)
            res_item.setForeground(QColor(0, 102, 204)) # 파란색 글씨 강조
            self.table_widget.setItem(row, 3, res_item)
    
    # 🔍 전송 전, 최종 페이로드 데이터가 완벽한 구조인지 터미널에 뿌려보는 디버깅 함수
    def debug_final_payload(self):
        self.searchDirectory()
        print("\n📦 === [READY TO SEND DB] 최종 적재 데이터 목록 ===")
        for idx, zone in enumerate(self.ocr_zones):
            print(f" [{idx+1}] 필드명: {zone['field']} | 데이터 값: {zone['text']} | 좌표: {zone['rect']}")
        print("==================================================\n")
        print("💡 위 데이터 구조가 그대로 Supabase에 한 줄로 깔끔하게 꽂히게 됩니다!")

    def clear_all_data(self):
        self.ocr_zones.clear()
        self.table_widget.setRowCount(0)
        print("🗑️ 모든 등록 구역 데이터가 초기화되었습니다.")

    def closeEvent(self, event):
        print('애플리케이션 종료')
        self.digimon_db.disconnect()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigimonInspectorWindow()
    window.show()
    sys.exit(app.exec_())