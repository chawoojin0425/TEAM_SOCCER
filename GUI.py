import customtkinter as ctk
import tkinter as tk
from PIL import Image

# 1. 외관 및 테마 설정
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 창 설정
        self.title("축구선수 정보 목록")
        self.geometry("1280x850")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # 2. 메인 콘텐츠 (왼쪽) - 이미지 추가
        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)

        # --- 이미지 추가 부분 (예시) ---
        # 🚨 이 경로는 실제 이미지 파일 경로로 변경해야 합니다!
        image_path = r'C:\Users\chawo\PycharmProjects\test_photo.jpg'
        try:
            original_image = Image.open(image_path)
            frame_width = 1000
            aspect_ratio = original_image.width / original_image.height
            resized_image = original_image.resize((frame_width, int(frame_width / aspect_ratio)))
            my_image = ctk.CTkImage(light_image=resized_image, dark_image=resized_image,
                                    size=(resized_image.width, resized_image.height))

            self.image_label = ctk.CTkLabel(self.main_content_frame, image=my_image, text="")
            self.image_label.grid(row=0, column=0, sticky="nsew")
        except FileNotFoundError:
            self.image_label = ctk.CTkLabel(self.main_content_frame, text=f"이미지 파일을 찾을 수 없습니다.\n경로 확인: {image_path}",
                                            text_color="red")
            self.image_label.grid(row=0, column=0, sticky="nsew")
        # --- 이미지 추가 끝 ---

        # 3. 선수 정보 목록 패널 (오른쪽) 생성 (회색 배경: #444444)
        self.sidebar_frame = ctk.CTkFrame(self,
                                          width=250,
                                          corner_radius=10,
                                          fg_color="#444444")
        self.sidebar_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        self.sidebar_frame.grid_columnconfigure(0, weight=1)
        self.sidebar_frame.grid_rowconfigure(0, weight=0)
        self.sidebar_frame.grid_rowconfigure(1, weight=0)
        self.sidebar_frame.grid_rowconfigure(2, weight=0)
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        # 4. 목록 패널 안에 제목 추가
        self.sidebar_title = ctk.CTkLabel(self.sidebar_frame,
                                          text="선수 보기 옵션 및 전체 목록",
                                          font=ctk.CTkFont(size=18, weight="bold"))
        self.sidebar_title.grid(row=0, column=0, padx=10, pady=(15, 5), sticky="n")

        # --- 5. 필터 체크박스 섹션 ---
        self.checkbox_scroll_frame = ctk.CTkFrame(self.sidebar_frame,  height=60)
        self.checkbox_scroll_frame.grid(row=1, column=0, padx=10, pady=(1, 1), sticky="nsew")

        checkbox_items = [
            "공과 가장 가까운 선수",
            "Team1 전체",
            "Team2 전체",
            "전체 선수"
        ]

        self.checkboxes = []
        for i, text in enumerate(checkbox_items):
            checkbox = ctk.CTkCheckBox(self.checkbox_scroll_frame,
                                       text=text,
                                       font=("Arial", 15),  # 폰트도 함께 줄여야 자연스럽습니다.
                                       height=18,  # ✅ 위젯 전체 높이
                                       checkbox_height=18,  # ✅ 체크 사각형 높이
                                       checkbox_width=18)  # ✅ 체크 사각형 너비

            # ✅ 수정된 부분: lambda에 checkbox 객체를 기본값으로 바인딩하여 UnboundLocalError 해결
            checkbox.configure(command=lambda cb=checkbox: self.toggle_checkboxes(cb))

            self.checkboxes.append(checkbox)
            checkbox.pack(pady=(5, 3), padx=10, anchor="w")

            # 기본적으로 '전체 선수' 선택
            if i == len(checkbox_items) - 1:
                checkbox.select()

        # --- 6. 초기화 버튼 (체크박스 바로 아래) ---
        self.reset_button = ctk.CTkButton(self.sidebar_frame,
                                          text="필터 초기화",
                                          command=self.reset_filters,
                                          fg_color="#a03030",
                                          hover_color="#c04040")

        self.reset_button.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")

        # --- 7. 22명 선수 전체 목록 섹션 ---
        self.player_list_frame = ctk.CTkScrollableFrame(self.sidebar_frame,
                                                        label_text="전체 선수 목록 (22명)",
                                                        fg_color="transparent")

        self.player_list_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="nsew")

        self.player_labels = []
        for i in range(22):
            # 팀 구분
            team = "팀1" if i < 11 else "팀2"
            player_name = f"{team} 선수 {i + 1}"

            player_label = ctk.CTkLabel(self.player_list_frame,
                                        text=player_name,
                                        anchor="w",
                                        padx=5,
                                        pady=2)
            self.player_labels.append(player_label)
            player_label.pack(fill="x", padx=5, pady=2)

    # =========================================================
    # 8. 기능 함수 정의
    # =========================================================

    def reset_filters(self):
        """
        모든 체크박스의 선택 상태를 해제합니다.
        """
        for checkbox in self.checkboxes:
            checkbox.deselect()
        print("모든 필터가 초기화되었습니다.")

    def toggle_checkboxes(self, selected_checkbox):
        """
        선택된 체크박스를 제외한 모든 체크박스를 해제하여 단일 선택을 구현합니다.
        """
        # 체크박스가 선택된 경우에만 다른 체크박스를 해제합니다.
        if selected_checkbox.get() == 1:
            for checkbox in self.checkboxes:
                # 현재 선택된 체크박스가 아닌 경우에만 해제
                if checkbox is not selected_checkbox:
                    checkbox.deselect()


# 9. 앱 실행
if __name__ == "__main__":
    app = App()
    app.mainloop()
