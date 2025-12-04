import customtkinter as ctk
# import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os  # 파일 존재 확인을 위해 os 모듈 추가

# =========================================================
# I. 상수 및 헬퍼 함수 정의 (변경 없음)
# =========================================================

MODEL_PATH = 'best.pt'
TEST_IMAGE_PATH = r'C:\Users\chawo\PycharmProjects\test_photo3(가장 잘나옴).jpg'


# ... (기존 상수 및 헬퍼 함수 생략 - get_representative_color, kmeans_clustering, get_center_coords) ...

# (편의상 기존 함수 복사)
def get_representative_color(img, box_coords):
    """선수의 바운딩 박스 중앙 1/3 영역의 픽셀을 추출합니다."""
    x1, y1, x2, y2 = map(int, box_coords)
    y_start = y1 + (y2 - y1) // 3
    y_end = y1 + 2 * (y2 - y1) // 3
    roi = img[y_start:y_end, x1:x2]
    if roi.size == 0:
        return None
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = np.float32(roi_rgb.reshape((-1, 3)))
    return pixels


def kmeans_clustering(all_pixels, k):
    """K-means 클러스터링을 실행합니다."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    if all_pixels.size == 0 or len(all_pixels) < k:
        return np.array([]), np.array([])
    ret, labels, centers = cv2.kmeans(all_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels, np.uint8(centers)


def get_center_coords(box):
    """바운딩 박스의 중심 좌표를 계산합니다."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


# K-means 클러스터 인덱스
GRASS_CLUSTER_INDEX = 0
TEAM_OURS_INDEX = 1
TEAM_OPPONENT_INDEX = 2

# 시각화 색상 (BGR 포맷)
COLOR_OURS = (0, 255, 0)  # Green
COLOR_OPPONENT = (0, 0, 255)  # Red
COLOR_OTHER = (100, 100, 100)  # Gray
COLOR_BALL = (0, 255, 255)  # Yellow
BALL_CLASS_ID = 1

# (수정) 1번 방식: 수동 목록 매칭을 위한 선수 이름
OURS_NAMES = ["[GK]Marc-André Ter Stegen", "[DF]Alejandro Balde", "CB_R", "LB", "RB", "CM_L",
              "CM_R", "LW", "RW", "ST_L", "ST_R"]  # 예시 포지션으로 수정
OPPONENT_NAMES = ["Opponent 1", "Opponent 2", "Opponent 3", "Opponent 4", "Opponent 5", "Opponent 6", "Opponent 7",
                  "Opponent 8", "Opponent 9", "Opponent 10", "Opponent 11"]


# ... (기존 process_detection_and_classification 함수 생략 - 내용 동일) ...
def process_detection_and_classification(model, image_path):
    """
    YOLO를 실행하여 선수와 공을 탐지하고, K-means를 사용하여 선수에게 팀을 분류한 후,
    X-좌표 순으로 이름을 할당합니다. (기존 로직과 동일)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, [], []

    try:
        results = model.predict(source=image_path, conf=0.01, iou=0.1, save=False, verbose=False)
    except Exception as e:
        print(f"YOLO 모델 예측 중 오류 발생: {e}")
        return img, [], []

    if not results or not results[0].boxes:
        return img, [], []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    all_player_pixels = []
    player_boxes_list = []
    ball_boxes = []

    # 1. 선수 픽셀 수집 및 공 박스 저장
    for box, cls in zip(boxes, classes):
        box_list = box.tolist()
        if int(cls) == 0:  # Player
            pixels = get_representative_color(img, box_list)
            if pixels is not None:
                all_player_pixels.append(pixels)
                player_boxes_list.append(box_list)
        elif int(cls) == BALL_CLASS_ID:  # Ball
            ball_boxes.append(box_list)

    if not all_player_pixels:
        print("탐지된 선수 객체가 없어 K-means를 실행할 수 없습니다.")
        return img, [], ball_boxes

    all_pixels_combined = np.vstack(all_player_pixels)

    # 2. K-means 클러스터링
    labels, cluster_centers = kmeans_clustering(all_pixels_combined, 3)

    if labels.size == 0:
        print("K-means 클러스터링에 문제가 발생했습니다.")
        return img, [], ball_boxes

    # 3. 각 선수에 팀 레이블만 임시 할당
    current_pixel_index = 0
    temp_player_data = []

    for box in player_boxes_list:
        x1, y1, x2, y2 = map(int, box)

        # ROI 픽셀 수 계산
        roi_height = ((2 * (y2 - y1) // 3) - ((y2 - y1) // 3))
        pixel_count = (x2 - x1) * roi_height

        current_labels = labels[current_pixel_index: current_pixel_index + pixel_count]
        (unique, counts) = np.unique(current_labels, return_counts=True)

        valid_counts = {}
        for idx, cluster_label in enumerate(unique):
            if cluster_label != GRASS_CLUSTER_INDEX:
                valid_counts[cluster_label] = counts[idx]

        team_id = "OTHER"

        if valid_counts:
            player_cluster = max(valid_counts, key=valid_counts.get)

            if player_cluster == TEAM_OURS_INDEX:
                team_id = "TEAM_OURS"
            elif player_cluster == TEAM_OPPONENT_INDEX:
                team_id = "TEAM_OPPONENT"

        temp_player_data.append({
            'team': team_id,
            'box': box,
            'center': get_center_coords(box)
        })

        current_pixel_index += pixel_count

    # 🚨 4. 팀별로 분리 및 X-좌표(가로 위치) 기준으로 정렬
    ours_list = [p for p in temp_player_data if p['team'] == 'TEAM_OURS']
    opponent_list = [p for p in temp_player_data if p['team'] == 'TEAM_OPPONENT']
    other_list = [p for p in temp_player_data if p['team'] == 'OTHER']

    # X-좌표(center[0])가 작은 순서대로 정렬 (왼쪽 -> 오른쪽)
    ours_list.sort(key=lambda p: p['center'][0])
    opponent_list.sort(key=lambda p: p['center'][0])

    # 5. 정렬된 순서대로 이름 할당 및 최종 리스트 병합
    final_player_data = []

    # Our Team Naming
    for i, player in enumerate(ours_list):
        name = OURS_NAMES[i] if i < len(OURS_NAMES) else f"Ours Player {i + 1} (Sorted X)"
        player['name'] = name
        # 고유 식별자 추가
        player['id'] = f"OURS_{i}"
        final_player_data.append(player)

    # Opponent Team Naming
    for i, player in enumerate(opponent_list):
        name = OPPONENT_NAMES[i] if i < len(OPPONENT_NAMES) else f"Opponent Player {i + 1} (Sorted X)"
        player['name'] = name
        # 고유 식별자 추가
        player['id'] = f"OPPONENT_{i}"
        final_player_data.append(player)

    # Other Team Naming
    for i, player in enumerate(other_list):
        player['name'] = f"Other {i + 1}"
        player['id'] = f"OTHER_{i}"
        final_player_data.append(player)

    return img, final_player_data, ball_boxes


# ... (기존 visualize_results 함수 생략 - 내용 동일) ...
def visualize_results(original_img, player_data, ball_boxes, filter_type):
    """
    필터 타입에 따라 선수와 공을 시각화한 이미지를 반환합니다. (기존 로직과 동일)
    """
    if original_img is None:
        return None, []

    # 'NONE' 필터일 경우 원본 이미지를 그대로 반환하고 선수 목록은 비워둡니다.
    if filter_type == 'NONE':
        return original_img.copy(), []

    img_classified = original_img.copy()
    players_to_draw = []
    nearest_player = None

    # 1. 필터링 로직
    if filter_type == 'OURS':
        players_to_draw = [p for p in player_data if p['team'] == 'TEAM_OURS']
    elif filter_type == 'OPPONENT':
        players_to_draw = [p for p in player_data if p['team'] == 'TEAM_OPPONENT']
    elif filter_type == 'ALL':
        players_to_draw = player_data
    elif filter_type == 'NEAREST_BALL':
        min_distance_sq = float('inf')

        if ball_boxes:
            ball_centers = [get_center_coords(box) for box in ball_boxes]

            for player in player_data:
                p_center_x, p_center_y = player['center']

                # 모든 공과의 최소 거리 찾기
                for b_center_x, b_center_y in ball_centers:
                    dist_sq = (p_center_x - b_center_x) ** 2 + (p_center_y - b_center_y) ** 2

                    if dist_sq < min_distance_sq:
                        min_distance_sq = dist_sq
                        nearest_player = player

            if nearest_player:
                players_to_draw = [nearest_player]

    # 2. 선수 시각화
    for player in players_to_draw:
        x1, y1, x2, y2 = map(int, player['box'])
        team = player['team']
        name = player['name']

        color = COLOR_OTHER
        if team == "TEAM_OURS":
            color = COLOR_OURS
        elif team == "TEAM_OPPONENT":
            color = COLOR_OPPONENT

        # 일반 바운딩 박스
        cv2.rectangle(img_classified, (x1, y1), (x2, y2), color, 2)

        text_label = name

        # 'NEAREST_BALL'인 경우, 특별히 표시
        if filter_type == 'NEAREST_BALL' and player is nearest_player:
            # 가장 가까운 선수에게 노란색 테두리 추가
            cv2.rectangle(img_classified, (x1, y1), (x2, y2), (0, 255, 255), 4)

            # 이름 텍스트 출력
        cv2.putText(img_classified, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. 공 시각화
    for box in ball_boxes:
        x1, y1, x2, y2 = map(int, box)
        text_label = f"BALL"

        cv2.rectangle(img_classified, (x1, y1), (x2, y2), COLOR_BALL, 3)
        (text_w, text_h), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_classified, (x1, y1 - text_h - baseline), (x1 + text_w, y1), COLOR_BALL, -1)
        cv2.putText(img_classified, text_label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_classified, players_to_draw


# =========================================================
# IV. CustomTkinter GUI 클래스 (수정됨)
# =========================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 창 설정
        self.title("축구선수 팀 분류 및 시각화 앱 (Zoom 지원)")
        self.geometry("1280x850")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.model = YOLO(MODEL_PATH)
        self.player_data = []
        self.ball_boxes = []
        self.original_img = None
        self.current_tk_image = None
        self.current_filter_type = 'NONE'

        # 이미지 크기 설정
        self.FRAME_WIDTH = 900
        self.zoom_factor = 1.0  # 줌 배율

        # 🚨 추가: 선수 프로필 데이터 딕셔너리
        # '선수 ID'를 키로 사용합니다. 선수 ID는 process_detection_and_classification에서 할당됩니다.
        self.player_profiles = {
            'OURS_0': {'photo': r'C:\Users\chawo\PycharmProjects\PlayersProfile\Team1\gk1_Ter_Stegen.webp',
                       'profile': '포지션: 골키퍼\n생년월일: 30/04/1992\n키: 187cm\n몸무게: 88kg'},
            'OURS_1': {'photo': r'C:\Users\chawo\PycharmProjects\PlayersProfile\Team1\def1_Balde.webp',
                       'profile': '포지션: 수비수\n생년월일: 18/10/2003\n키: 177cm\n몸무게: 76kg'},
            'OURS_2': {'photo': 'player_def.jpg', 'profile': '포지션: 센터백 (오른쪽)\n나이: 30세\n키: 188cm\n특징: 강한 피지컬과 리더십.'},
            'OPPONENT_0': {'photo': 'opponent_player.jpg', 'profile': '포지션: 공격수\n나이: 22세\n키: 178cm\n특징: 빠른 스피드와 드리블.'},
            # 여기에 더 많은 선수 프로필을 추가할 수 있습니다.
            # 'photo' 경로는 실제 사진 파일 경로로 대체해야 합니다.
        }
        # 대체 이미지 경로 (사진 파일이 없을 경우 사용)
        self.default_photo_path = 'default_player.jpg'

        # 2. 메인 콘텐츠 (왼쪽) - 이미지 프레임
        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)

        # 3. 이미지 표시를 위한 캔버스 설정 (Zoom 기능 구현)
        self.image_canvas = ctk.CTkCanvas(self.main_content_frame,
                                          bg=self._apply_appearance_mode(
                                              ctk.ThemeManager.theme["CTkFrame"]["fg_color"]),
                                          highlightthickness=0)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        self.image_item = self.image_canvas.create_image(0, 0, anchor="nw", image=None)

        # 줌 이벤트 바인딩 (생략 없이 유지)
        self.image_canvas.bind("<MouseWheel>", self.zoom_handler)
        self.image_canvas.bind("<Button-4>", lambda event: self.zoom_handler(event, 1))
        self.image_canvas.bind("<Button-5>", lambda event: self.zoom_handler(event, -1))
        self.image_canvas.bind("<ButtonPress-1>", self.move_start)
        self.image_canvas.bind("<B1-Motion>", self.move_move)
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

        # 초기 로딩 텍스트
        self.loading_text = self.image_canvas.create_text(
            self.FRAME_WIDTH // 2,
            self.FRAME_WIDTH // 2,
            text="이미지 로딩 및 분석 중...",
            fill="white",
            font=("Arial", 20)
        )

        # 🚨 1. 데이터 전처리 및 분류 실행 (딱 한 번)
        print("시작: YOLO 탐지 및 K-means 팀 분류...")
        self.original_img, self.player_data, self.ball_boxes = process_detection_and_classification(self.model,
                                                                                                    TEST_IMAGE_PATH)
        print("완료: 데이터 처리 완료.")

        if self.original_img is None:
            self.display_error_ui(f"❌ 오류: 이미지 로드 또는 모델 처리 실패.\n경로 확인: '{TEST_IMAGE_PATH}' 또는 '{MODEL_PATH}'")
            return

        # 4. 목록 패널 설정 (변경 없음)
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=10, fg_color="#444444")
        self.sidebar_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.sidebar_frame.grid_columnconfigure(0, weight=1)
        for i in range(4): self.sidebar_frame.grid_rowconfigure(i, weight=0)
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        self.sidebar_title = ctk.CTkLabel(self.sidebar_frame,
                                          text="선수 보기 옵션 및 현재 목록",
                                          font=ctk.CTkFont(size=18, weight="bold"))
        self.sidebar_title.grid(row=0, column=0, padx=10, pady=(15, 5), sticky="n")

        # 5. 필터 체크박스 섹션 (변경 없음)
        self.checkbox_scroll_frame = ctk.CTkFrame(self.sidebar_frame, height=120)
        self.checkbox_scroll_frame.grid(row=1, column=0, padx=10, pady=(1, 1), sticky="nsew")

        checkbox_items = [
            ("공과 가장 가까운 선수", "NEAREST_BALL"),
            ("우리 팀만 보기 (OURS)", "OURS"),
            ("상대 팀만 보기 (OPPONENT)", "OPPONENT"),
            ("전체 선수 보기 (ALL)", "ALL")
        ]

        self.checkboxes = {}
        for i, (text, filter_key) in enumerate(checkbox_items):
            checkbox = ctk.CTkCheckBox(self.checkbox_scroll_frame,
                                       text=text,
                                       font=("Arial", 15),
                                       height=18,
                                       checkbox_height=18,
                                       checkbox_width=18,
                                       command=lambda fk=filter_key: self.toggle_checkboxes(fk))

            self.checkboxes[filter_key] = checkbox
            checkbox.pack(pady=(5, 3), padx=10, anchor="w")

        # 6. 초기화 버튼 (변경 없음)
        self.reset_button = ctk.CTkButton(self.sidebar_frame,
                                          text="필터 초기화 (원본 이미지)",
                                          command=self.reset_filters,
                                          fg_color="#a03030",
                                          hover_color="#c04040")

        self.reset_button.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")

        # 7. 선수 목록 프레임 (변경 없음)
        self.player_list_frame = ctk.CTkScrollableFrame(self.sidebar_frame,
                                                        label_text="현재 표시 선수 목록 (0명)",
                                                        fg_color="transparent")

        self.player_list_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="nsew")

        self.player_list_items = []  # 라벨과 버튼 프레임을 저장할 리스트

        # 8. 초기 상태 설정
        self.reset_filters()

    # ... (기존 move_start, move_move, zoom_handler, on_canvas_resize, update_image_display 함수 생략 - 내용 동일) ...
    def move_start(self, event):
        """이미지 이동 시작 시점 (마우스 버튼 클릭)"""
        self.image_canvas.scan_mark(event.x, event.y)

    def move_move(self, event):
        """이미지 이동 중 (마우스 드래그)"""
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom_handler(self, event, direction=None):
        """마우스 휠 이벤트 발생 시 줌 인/아웃 처리"""
        scale_factor = 1.1
        if direction is not None:
            if direction == 1:
                delta = 1
            else:
                delta = -1
        else:
            delta = event.delta / 120

        if delta > 0:
            self.zoom_factor *= scale_factor
        elif delta < 0:
            self.zoom_factor /= scale_factor

        self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))
        self.update_image_display(self.current_filter_type, zoom_update=True)

    def on_canvas_resize(self, event):
        """캔버스 크기가 변경될 때 이미지를 중앙에 다시 배치합니다."""
        self.update_image_display(self.current_filter_type, zoom_update=True)

    def update_image_display(self, filter_type, zoom_update=False):
        """
        필터 타입에 따라 이미지를 업데이트하고 GUI에 표시합니다.
        """
        if self.original_img is None:
            return

        self.current_filter_type = filter_type

        # 1. 시각화 및 선수 목록 필터링
        classified_img_bgr, players_to_draw = visualize_results(self.original_img, self.player_data, self.ball_boxes,
                                                                filter_type)

        # 2. BGR -> RGB 변환 및 PIL Image로 변환
        img_rgb = cv2.cvtColor(classified_img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # 3. GUI 크기 및 줌 배율에 맞게 이미지 리사이즈
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width = self.FRAME_WIDTH
            canvas_height = self.FRAME_WIDTH * pil_image.height // pil_image.width

        aspect_ratio = pil_image.width / pil_image.height

        if canvas_width / aspect_ratio <= canvas_height:
            display_width = canvas_width
            display_height = int(display_width / aspect_ratio)
        else:
            display_height = canvas_height
            display_width = int(display_height * aspect_ratio)

        display_width = int(display_width * self.zoom_factor)
        display_height = int(display_height * self.zoom_factor)

        resized_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # 4. CustomTkinter 이미지 객체 생성 및 캔버스 업데이트
        self.current_tk_image = ImageTk.PhotoImage(image=resized_image)
        center_x = canvas_width // 2
        center_y = canvas_height // 2

        self.image_canvas.delete(self.loading_text)
        self.image_canvas.itemconfig(self.image_item, image=self.current_tk_image)
        self.image_canvas.coords(self.image_item, center_x - display_width // 2, center_y - display_height // 2)

        # 5. 선수 목록 업데이트 호출
        self.update_player_list(players_to_draw)

    # =========================================================
    # VI. 기능 함수 정의 (수정/추가)
    # =========================================================

    def show_player_profile(self, player_id, name):
        """
        선수 프로필 팝업 창을 띄웁니다.
        """
        # 팝업 윈도우 생성
        profile_window = ctk.CTkToplevel(self)
        profile_window.title(f"{name} 선수 프로필")
        profile_window.geometry("400x500")
        profile_window.resizable(False, False)
        profile_window.attributes('-topmost', True)  # 항상 위에 표시

        # 중앙 프레임
        main_frame = ctk.CTkFrame(profile_window, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        main_frame.grid_columnconfigure(0, weight=1)

        # 선수 데이터 가져오기 (없으면 기본값 사용)
        profile_data = self.player_profiles.get(player_id,
                                                {'photo': self.default_photo_path,
                                                 'profile': '프로필 정보 없음.'})
        photo_path = profile_data['photo']
        profile_text = profile_data['profile']

        # 1. 선수 사진 표시
        photo_size = (400, 500)
        try:
            # 사진 파일 확인 및 로드
            if not os.path.exists(photo_path):
                print(f"경고: 선수 사진 파일 '{photo_path}'이(가) 존재하지 않습니다. 기본 이미지를 사용합니다.")
                photo_path = self.default_photo_path

            img = Image.open(photo_path).resize(photo_size, Image.Resampling.LANCZOS)
            photo_tk = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"사진 로드 중 오류 발생 ({photo_path}): {e}")
            # 오류 발생 시 빈 이미지 사용
            img = Image.new('RGB', photo_size, color='gray')
            photo_tk = ImageTk.PhotoImage(img)

        photo_label = ctk.CTkLabel(main_frame, text="", image=photo_tk)
        photo_label.image = photo_tk  # 가비지 컬렉션 방지
        photo_label.grid(row=0, column=0, pady=(0, 10), sticky="n")

        # 2. 선수 이름 (제목)
        name_label = ctk.CTkLabel(main_frame, text=name, font=ctk.CTkFont(size=20, weight="bold"))
        name_label.grid(row=1, column=0, pady=(5, 10), sticky="n")

        # 3. 프로필 정보
        profile_label = ctk.CTkLabel(main_frame,
                                     text=profile_text,
                                     font=ctk.CTkFont(size=14),
                                     justify="left")
        profile_label.grid(row=2, column=0, pady=(10, 20), sticky="ew")

        # 4. 닫기 버튼
        close_button = ctk.CTkButton(main_frame, text="닫기", command=profile_window.destroy)
        close_button.grid(row=3, column=0, pady=(10, 0), sticky="s")

    def update_player_list(self, players_to_draw):
        """
        현재 화면에 표시된 선수들만 목록에 업데이트하고 '조회' 버튼을 추가합니다.
        """
        # 기존 항목 모두 삭제
        for item in self.player_list_items:
            item.destroy()
        self.player_list_items.clear()

        # 새 항목(프레임 + 라벨 + 버튼) 생성
        for player in players_to_draw:
            player_info = f"[{player['team'].replace('TEAM_', '')}] {player['name']}"
            player_id = player.get('id', 'UNKNOWN')  # 고유 ID를 가져옴

            # 1. 항목 전체를 담을 프레임 생성 (선수 정보와 버튼을 가로로 배치)
            item_frame = ctk.CTkFrame(self.player_list_frame, fg_color="transparent")
            item_frame.pack(fill="x", padx=5, pady=2)
            item_frame.grid_columnconfigure(0, weight=1)  # 라벨이 공간을 차지하도록 설정

            # 2. 선수 정보 라벨 (왼쪽)
            player_label = ctk.CTkLabel(item_frame,
                                        text=player_info,
                                        anchor="w",
                                        padx=5,
                                        pady=2)
            player_label.grid(row=0, column=0, sticky="ew")

            # 3. 조회 버튼 (오른쪽)
            view_button = ctk.CTkButton(item_frame,
                                        text="조회",
                                        width=50,
                                        height=20,
                                        font=ctk.CTkFont(size=12, weight="bold"),
                                        command=lambda pid=player_id, pname=player['name']: self.show_player_profile(
                                            pid, pname))
            view_button.grid(row=0, column=1, padx=(5, 0), sticky="e")

            # 항목 저장
            self.player_list_items.append(item_frame)

        # 라벨 업데이트
        self.player_list_frame.configure(label_text=f"현재 표시 선수 목록 ({len(players_to_draw)}명)")

    def reset_filters(self):
        """
        모든 체크박스를 해제하고 'NONE' (원본 이미지) 필터를 기본으로 설정합니다.
        """
        for checkbox in self.checkboxes.values():
            checkbox.deselect()

        self.zoom_factor = 1.0  # 줌 배율 초기화
        # 캔버스 스크롤 위치 초기화
        self.image_canvas.xview_moveto(0)
        self.image_canvas.yview_moveto(0)

        self.update_image_display('NONE')  # 원본 이미지를 표시
        print("모든 필터가 초기화되었습니다. (원본 이미지 보기)")

    def toggle_checkboxes(self, selected_filter_key):
        """
        단일 선택을 구현하고, 선택된 필터에 따라 이미지를 업데이트합니다.
        """
        selected_checkbox = self.checkboxes[selected_filter_key]

        # 1. 체크박스가 선택된 경우: 다른 모든 체크박스 해제 및 이미지 업데이트
        if selected_checkbox.get() == 1:
            for filter_key, checkbox in self.checkboxes.items():
                if filter_key != selected_filter_key:
                    checkbox.deselect()

            # 선택된 필터로 이미지 업데이트
            self.update_image_display(selected_filter_key)

        # 2. 체크박스가 해제된 경우: 모든 체크박스가 해제되었다면 원본 이미지 상태로 전환
        else:
            is_any_checked = any(cb.get() == 1 for cb in self.checkboxes.values())

            if not is_any_checked:
                self.update_image_display('NONE')  # 원본 이미지 상태로 전환


# =========================================================
# VII. 앱 실행
# =========================================================

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"애플리케이션 실행 중 치명적인 오류 발생: {e}")
