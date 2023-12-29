import tkinter as tk
from tkinter.messagebox import showinfo
import random
import numpy
from PIL import Image, ImageTk
from DFS import dfs
from BFS import bfs

class PacMan():
    def __init__(self):
        """ 游戏参数设置 """
        global fps
        fps = 150     # pacman的移动速度（单位毫秒）
        self.row_cells = 31    # 列
        self.col_cells = 26    # 行
        self.canvas_bg = 'black'
        self.cell_size = 27
        self.cell_gap = 0
        self.frame_x = 0
        self.frame_y = 0
        self.color_dict = {0: 'black', 1: 'blue', 2: 'black', 3: 'black', 8: 'black'}
        self.step = []
        self.pac_man = []

        self.run_game()

    def window_center(self, window, w_size, h_size):
        """ 窗口居中 """
        screenwidth = window.winfo_screenwidth()
        screenheight = window.winfo_screenheight()
        left = (screenwidth - w_size) // 2
        top = (screenheight - h_size) // 2
        window.geometry("%dx%d+%d+%d" % (w_size, h_size, left, top))
        window.resizable(0, 0)

    def create_map(self):
        """ 创建地图列表 """
        global game_map
        # game_map = numpy.zeros((self.col_cells, self.row_cells))
        game_map1 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                     [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                     [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                     [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                     [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                     [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                     [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                     [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                     [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                     [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                     [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                     [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1],
                     [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                     [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                     [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ]
        game_map = numpy.array(game_map1)

    def create_canvas(self):
        """ 创建画布 """
        global canvas
        canvas_h = self.cell_size * self.col_cells + self.frame_y * 2
        canvas_w = self.cell_size * self.row_cells + self.frame_x * 2
        canvas = tk.Canvas(window, bg=self.canvas_bg,
                           height=canvas_h,
                           width=canvas_w,
                           highlightthickness=0)
        canvas.place(x=0, y=0)

    def create_cells(self):
        """ 创建单元格 """
        for y in range(0, self.col_cells):
            for x in range(0, self.row_cells):
                a = self.frame_x + self.cell_size * x
                b = self.frame_y + self.cell_size * y
                c = self.frame_x + self.cell_size * (x + 1)
                d = self.frame_y + self.cell_size * (y + 1)
                e = self.canvas_bg
                f = self.cell_gap
                g = self.color_dict[game_map[y][x]]
                canvas.itemconfig(canvas.create_rectangle(a, b, c, d, outline=e, width=f, fill=g), fill=g)

    def creat_food(self):
        """ 创建食物 """
        global food_xy, photo1, mm1
        food_xy = [0, 0]
        food_xy[1] = random.randint(1, self.row_cells - 2)
        food_xy[0] = random.randint(1, self.col_cells - 2)
        while game_map[food_xy[0]][food_xy[1]] != 0:
            food_xy[1] = random.randint(1, self.row_cells - 2)
            food_xy[0] = random.randint(1, self.col_cells - 2)
        # game_map[food_xy[0]][food_xy[1]] = 2
        mm1 = tk.Label(window, height=20, width=20, image=photo1, bg='black')
        mm1.place(relx=float((food_xy[1] + 0.476) / 31), rely=float((food_xy[0] + 0.47) / 26), anchor="center")

    def creat_pacman(self):
        """ 创建pacman """
        self.pac_man = [self.col_cells // 2, self.row_cells // 2]
        game_map[self.pac_man[0], self.pac_man[1]] = 2

    def pacman_xy(self):
        """ 获取pacman坐标 """
        global head_x, head_y
        # print(game_map)
        b = numpy.where(game_map == 3)
        head_x = b[0][0]
        head_y = b[1][0]

    def compute_move(self):
        self.step = dfs(game_map, self.pac_man, food_xy)

    def eat_food(self):
        global mm1
        if [head_x, head_y] == food_xy:
            print("吃到啦！")
            mm1.place_forget()
            self.pac_man = [head_x, head_y]
            self.creat_food()
            self.compute_move()

    # def auto_move(self):
    #     """ 自动前进 """
    #     def move(d, x, y):
    #         if dd[0] == d:  # 根据方向值来决定走向
    #             game_map[head_x + x][head_y + y] = 3
    #             game_map[head_x + 0][head_y + 0] = 0
    #
    #     move(1, -1, 0)
    #     move(2, 1, 0)
    #     move(3, 0, -1)
    #     move(4, 0, 1)


    def game_loop(self):
        """ 游戏循环刷新 """
        global head_x, head_y
        print("move")
        global loop_id
        self.eat_food()
        # self.auto_move()
        # self.pacman_xy()
        head_x, head_y = self.step.pop(0)
        mm.place_forget()
        self.papa_png()
        canvas.delete('all')
        self.create_cells()
        # self.game_over()
        if loop == 1:
            loop_id = window.after(fps, self.game_loop)

    def papa_png(self):
        global mm
        global photo
        mm = tk.Label(window, height=20, width=20, image=photo, bg='black')
        mm.place(relx=float((head_y + 0.476) / 31), rely=float((head_x + 0.47) / 26), anchor="center")
        # mm.place_forget()

    def game_start(self):
        """  """
        global window, backup_map, dd, loop
        global head_x, head_y
        loop = 1  # 暂停标记，1为开启，0为暂停
        dd = [0]  # 记录按键方向
        self.create_map()
        self.creat_pacman()
        self.creat_food()
        self.compute_move()
        # self.pacman_xy()
        head_x, head_y = self.step.pop(0)
        self.papa_png()
        self.game_loop()

        def close_w():
            global loop
            loop = 0
            window.after_cancel(loop_id)
            window.destroy()

        window.protocol('WM_DELETE_WINDOW', close_w)
        window.mainloop()

    def run_game(self):
        """ 开启游戏 """
        global window, Pac, Food
        window = tk.Tk()
        window.focus_force()  # 主窗口焦点
        window.title('PacMan')
        Pac = Image.open('000.png')
        Pac = ImageTk.PhotoImage(Pac)
        Food = Image.open('111.png')
        Food = ImageTk.PhotoImage(Food)
        win_w_size = self.row_cells * self.cell_size + self.frame_x * 2
        win_h_size = self.col_cells * self.cell_size + self.frame_y * 2
        self.window_center(window, win_w_size, win_h_size)
        self.create_canvas()
        self.game_start()

if __name__ == '__main__':

    PacMan()
