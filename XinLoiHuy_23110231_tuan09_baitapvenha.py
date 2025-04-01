import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pygame
import time
import random
import sys
from collections import deque
import heapq

WIDTH, HEIGHT = 300, 400
GRID_SIZE = WIDTH // 3
FONT_SIZE = 40
BACKGROUND_COLOR = "white"
TILE_COLOR = "green"
TEXT_COLOR = "white"
EMPTY_TILE_COLOR = "gray"

# Các hướng di chuyển: xuống, lên, phải, trái
move = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def find_zero(table):
    """Tìm vị trí của số 0 trong bảng."""
    for i in range(3):
        for j in range(3):
            if table[i][j] == 0:
                return i, j

def swap(state, row, col, new_row, new_col):
    """Hoán đổi vị trí của số 0 với một ô lân cận."""
    state_list = [list(row) for row in state]
    state_list[row][col], state_list[new_row][new_col] = state_list[new_row][new_col], state_list[row][col]
    return tuple(tuple(row) for row in state_list)

def new_states(table):
    """Tạo ra các trạng thái mới từ trạng thái hiện tại."""
    row, col = find_zero(table)
    states = []
    for i, j in move:
        new_row, new_col = row + i, col + j
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_table = swap(table, row, col, new_row, new_col)
            states.append(new_table)
    return states

def bfs(table, final_table):
    queue = deque([(table, [table])])
    visited = set([table])
    iterations = 0

    while queue:
        table, path = queue.popleft()
        iterations += 1
        if table == final_table:
            return path, iterations

        for state in new_states(table):
            if state not in visited:
                visited.add(state)
                queue.append((state, path + [state]))

    return None, None

def dfs(table, final_table):
    stack = [(table, [table])]
    visited = set([table])
    iterations = 0

    while stack:
        table, path = stack.pop()
        iterations += 1
        
        if table == final_table:
            return path, iterations

        for next_state in new_states(table):
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [next_state]))

    return None, None

def dls(start, final_state, depth_max=39):
    stack = [(start, [start], 0)]
    visited = set([start])
    iterations = 0
    
    while stack:
        state, path, depth = stack.pop()
        iterations += 1

        if state == final_state:
            return path, iterations
        
        if depth < depth_max:
            for next_state in new_states(state):
                if next_state not in visited:
                    visited.add(next_state)
                    stack.append((next_state, path + [next_state], depth + 1))
                    
    return None,iterations

def iddfs(start,final_state):
    depth = 0
    all_iterations = 0
    while True:
        path, iterations = dls(start, final_state, depth)
        all_iterations += iterations
        if path:
            return path, all_iterations
        depth += 1 

def manhattan_distance(state, goal):
    distance = 0
    # Tạo một từ điển lưu vị trí của mỗi số trong trạng thái đích
    goal_positions = {}
    for i in range(3):
        for j in range(3):
            if goal[i][j] != 0:
                goal_positions[goal[i][j]] = (i, j)
    
    # Tính tổng khoảng cách Manhattan
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                goal_i, goal_j = goal_positions[state[i][j]]
                distance += abs(i - goal_i) + abs(j - goal_j)
    
    return distance

def misplaced_tiles(state, goal):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                count += 1
    return count

def ucs(start, goal):

    pq = [(0, 0, start, [start])]
    visited = set([start])
    iterations = 0
    counter = 1  # counter giúp tránh lỗi khi so sánh các tuple cùng cost
    
    while pq:
        cost, _, state, path = heapq.heappop(pq)
        iterations += 1
        
        if state == goal:
            return path, iterations
        
        for next_state in new_states(state):
            if next_state not in visited:
                visited.add(next_state)
                # Chi phí là 1 cho mỗi bước
                new_cost = len(path) + 1
                heapq.heappush(pq, (new_cost, counter, next_state, path + [next_state]))
                counter += 1
    
    return None, iterations

def greedy_best_first_search(start, goal, type):
    if type == "Misplaced Tiles":
        heuristic = misplaced_tiles
    else:
        heuristic = manhattan_distance
    pq = [(heuristic(start, goal), 0, start, [start])]
    visited = set([start])
    iterations = 0
    counter = 1
    
    while pq:
        _, _, state, path = heapq.heappop(pq)
        iterations += 1
        
        if state == goal:
            return path, iterations
        
        for next_state in new_states(state):
            if next_state not in visited:
                visited.add(next_state)
                heapq.heappush(pq, (heuristic(next_state, goal), counter, next_state, path + [next_state]))
                counter += 1
    
    return None, iterations

def A_Star(start, goal, type, alpha=None):
    if type == "Misplaced Tiles":
        heuristic = misplaced_tiles
    else:
        heuristic = manhattan_distance

    pq = [(heuristic(start, goal), 0, start, [start])]
    visited = set([start])
    iterations = 0
    counter = 1

    

    if alpha is not None and heuristic(start, goal) > alpha:
        return None, iterations
    
    
    while pq:
        cost, _, state, path = heapq.heappop(pq)
        iterations += 1
        
        if state == goal:
            return path, iterations
        
        
        
        for next_state in new_states(state):
            if next_state not in visited:
                visited.add(next_state)
                # Chi phí mới = chi phí di chuyển tới trạng thái hiện tại + khoảng cách heuristic
                # Chi phí là 1 cho mỗi bước
                new_cost = len(path) + heuristic(next_state, goal)
                if alpha and new_cost <= alpha:
                    heapq.heappush(pq, (new_cost, counter, next_state, path + [next_state]))
                    counter += 1
                elif alpha is None:
                    heapq.heappush(pq, (new_cost, counter, next_state, path + [next_state]))
                    counter += 1
    
    return None, iterations

def IDA_Star(start, goal, type, alpha):
    all_iterations = 0
    while True:
        path, iterations = A_Star(start, goal, type, alpha=alpha)
        all_iterations += iterations
        if path:
            return path, all_iterations
        alpha += 1
        

def SHC(start, goal, type):
    iterations = 0
    if type == "Misplaced Tiles":
        heuristic = misplaced_tiles
    else:
        heuristic = manhattan_distance

    current_state = start
    current_cost = heuristic(current_state,goal)
    path = [current_state]

    while True:
        iterations += 1
        best_cost = current_cost
        best_state = None

        if current_state == goal:
            return path, iterations
        
        next_states = new_states(current_state)
        random.shuffle(next_states)

        for next_state in next_states:
            new_cost = heuristic(next_state, goal)
            if new_cost < best_cost:
                best_state = next_state
                best_cost = new_cost
                break

        if best_state is None:
            return None, iterations 
        
        current_state = best_state
        path.append(current_state)
    
def SAHC(start, goal, type):
    
    if type == "Misplaced":
        heuristic = misplaced_tiles
    else:
        heuristic = manhattan_distance
    
    current_state = start
    path = [current_state]
    iterations = 0
    
    while True:
        iterations += 1

        if current_state == goal:
            return path, iterations
        
        next_states = new_states(current_state)
        
        neighbor_costs = []
        for next_state in next_states:
            cost = heuristic(next_state, goal)
            neighbor_costs.append((cost, next_state))
        
        # Sắp xếp các trạng thái neighbor theo chi phí tăng dần
        neighbor_costs.sort(key=lambda x: x[0])
        
        best_neighbor_cost, best_neighbor = neighbor_costs[0]
        
        current_cost = heuristic(current_state, goal)
        

        if best_neighbor_cost >= current_cost:
            return None, iterations
        
        current_state = best_neighbor
        path.append(current_state)



class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("23110231 - Xín Lợi Huy")
        self.root.geometry("600x400")

        self.initial_table = None
        self.final_table = (
                (1, 2, 3),
                (4, 5, 6),
                (7, 8, 0)
            )
        self.path = None
        self.iterations = 0

        self.current_state = None
        self.step = 0

        self.RUN = False

        self.create_widgets()
        
    def create_widgets(self):
        input_frame = tk.Frame(self.root)
        input_frame.grid(row=0, column=0, pady=10)
        
        self.label_algorithm = tk.Label(input_frame, text="Chọn thuật toán:")
        self.label_algorithm.grid(row=0, column=0, padx=5)
        
        self.algorithms = ["BFS", "UCS", "DFS", "DLS", "IDDFS", "Greedy Search", "A*", "IDA*","SHC","SAHC"]
        self.combobox = ttk.Combobox(input_frame, values=self.algorithms)
        self.combobox.grid(row=0, column=1, padx=5)
        self.combobox.current(0)
        
        self.label_input = tk.Label(input_frame, text="Nhập input:")
        self.label_input.grid(row=0, column=2, padx=5)
        self.entry = tk.Entry(input_frame, width=20)
        self.entry.grid(row=0, column=3, padx=5)

        self.input_depth_limit_frame = tk.Frame(self.root)
        self.input_depth_limit_frame.grid(row=1, column=0, pady=5)
        self.input_depth_limit_label = tk.Label(self.input_depth_limit_frame, text="Depth Limit:")
        self.input_depth_limit = tk.Entry(self.input_depth_limit_frame, width=20)
        self.input_depth_limit_label.grid(row=0, column=0, padx=5)
        self.input_depth_limit.grid(row=0, column=1, padx=5)
        self.input_depth_limit_label.grid_remove()
        self.input_depth_limit.grid_remove() 

        self.heuristics = ["Manhattan Distance", "Misplaced Tiles"]
        self.combobox_heuristics_frame = tk.Frame(self.root)
        self.combobox_heuristics_frame.grid(row=2, column=0, pady=5)
        self.label_heuristic = tk.Label(self.combobox_heuristics_frame, text="Chọn heuristic:")
        self.label_heuristic.grid(row=0, column=0, padx=5)
        self.combobox_heuristics = ttk.Combobox(self.combobox_heuristics_frame, values=self.heuristics)
        self.combobox_heuristics.grid(row=0, column=1, padx=5)
        self.combobox_heuristics.current(0)
        self.combobox_heuristics.grid_remove()
        self.label_heuristic.grid_remove()

        self.combobox.bind("<<ComboboxSelected>>", self.on_algorithm_change)
        
        self.btn_run = tk.Button(input_frame, text="Run", command=self.run, bg="lightgreen")
        self.btn_run.grid(row=0, column=4)
        
        self.btn_random_input = tk.Button(input_frame, text="Random Input", 
                                        command=lambda: [self.randomize(), self.change_random_color()], 
                                        bg="#FAF884")
        self.btn_random_input.grid(row=0, column=5, padx=5)

        self.btn_visual = tk.Button(input_frame, text="Visual", command=self.start_visualization, bg="yellow")
        self.btn_visual.grid(row=1, column=5, padx=4,pady=4,sticky="se")
        
        result_frame = tk.Frame(self.root)
        result_frame.grid(row=3, column=0, pady=10, columnspan=2)

        self.result_text = tk.Text(result_frame, height=13, width=60, background="lightblue", wrap="none")
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar = tk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=v_scrollbar.set)

        self.result_text.config(font=("Arial", 12, "bold"), fg="black")
        
    def change_random_color(self):
        rainbow_colors = ["#FFDDC1", "#FFABAB", "#FFC3A0", "#D5AAFF", "#85E3FF", "#B9FBC0", "#FAF884"]
        color = random.choice(rainbow_colors)
        self.btn_random_input.config(bg=color)
    
    def on_algorithm_change(self, event=None):
        selected_algorithm = self.combobox.get()
        
        if selected_algorithm == "DLS":
            self.input_depth_limit_label.grid(row=0, column=0, padx=5)
            self.input_depth_limit.grid(row=0, column=1, padx=5)
        else:
            self.input_depth_limit_label.grid_remove()
            self.input_depth_limit.grid_remove()

        if selected_algorithm in ["Greedy Search", "A*", "IDA*","SHC","SAHC"]:
            self.combobox_heuristics_frame.grid(row=2, column=0, pady=10)
            self.label_heuristic.grid(row=0, column=0, padx=5)
            self.combobox_heuristics.grid(row=0, column=1, padx=5)
        else:
            self.combobox_heuristics_frame.grid_remove()
            self.label_heuristic.grid_remove()
            self.combobox_heuristics.grid_remove()


    def count_inversions(self, puzzle):
        """Đếm số lượng nghịch đảo trong chuỗi puzzle."""
        inversions = 0
        for i in range(len(puzzle)):
            for j in range(i + 1, len(puzzle)):
                if puzzle[i] != '0' and puzzle[j] != '0' and puzzle[i] > puzzle[j]:
                    inversions += 1
        return inversions

    def randomize(self):
        """Tạo một chuỗi ngẫu nhiên hợp lệ cho 8-puzzle."""
        while True:
            puzzle = list("123456780")
            random.shuffle(puzzle)
            puzzle = ''.join(puzzle)
            
            if self.count_inversions(puzzle) % 2 == 0:
                self.entry.delete(0, tk.END)
                self.entry.insert(0, puzzle)
                return
    
    def run(self):
        self.RUN = True
        selected_algorithm = self.combobox.get()
        input_data = self.entry.get()
        input_depth = self.input_depth_limit.get()
        
        if not selected_algorithm:
            messagebox.showerror("Lỗi", "Vui lòng chọn thuật toán!")
            return
        
        if not input_data:
            messagebox.showerror("Lỗi", "Vui lòng nhập dữ liệu đầu vào!")
            return
        
        if not input_depth and selected_algorithm == "DLS":
            messagebox.showerror("Lỗi", "Vui lòng nhập depth limit!")
            return
        
        if self.count_inversions(input_data) % 2 != 0:
            messagebox.showerror("Lỗi", "Không thể giải bài toán với trạng thái hiện tại!")
            return
        
        self.run_algorithm(selected_algorithm)

        if self.path is None:
            return
        
    
    def run_algorithm(self, algorithm):
        input_data = self.entry.get()
        chunks = [input_data[i:i+3] for i in range(0, len(input_data), 3)]
        tuples = [tuple(int(char) for char in chunk) for chunk in chunks]
        self.initial_table = tuple(tuples)
        start_time = time.time()
        
        if algorithm == "BFS":  #Breadth First Search
            self.path, self.iterations = bfs(self.initial_table, self.final_table)
        elif algorithm == "DFS":    #Depth First Search
            self.path, self.iterations = dfs(self.initial_table, self.final_table)
        elif algorithm == "DLS":    #Depth Limited Search
            input_depth = int(self.input_depth_limit.get())
            self.path, self.iterations = dls(self.initial_table, self.final_table,depth_max=input_depth)
        elif algorithm == "UCS":    #Uniform Cost Search
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = ucs(self.initial_table, self.final_table)
        elif algorithm == "IDDFS":  #Iterative Deepening DFS
            self.path, self.iterations = iddfs(self.initial_table, self.final_table)
        elif algorithm == "Greedy Search":  #Best First Search
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = greedy_best_first_search(self.initial_table, self.final_table, type=heuristic_type)
        elif algorithm == "A*":     #A Star
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = A_Star(self.initial_table, self.final_table, type=heuristic_type)
        elif algorithm == "IDA*":   #Iterative A*
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = IDA_Star(self.initial_table, self.final_table, type=heuristic_type, alpha=22)
        elif algorithm == "SHC":    #Simple Hill Climbing
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = SHC(self.initial_table, self.final_table, type=heuristic_type)
        elif algorithm == "SAHC":   #Steepest-Ascent Hill Climbing
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = SAHC(self.initial_table, self.final_table, type=heuristic_type)
        else:
            messagebox.showerror("Lỗi", "Thuật toán chưa được triển khai")
            return
        
        if self.path is None:
            result = "Không giải được!"
        else:
            result = "Giải thành công!"
            depth = len(self.path) - 1
        
        execution_time = time.time() - start_time

        self.result_text.insert(tk.END, f"Kết quả cho thuật toán {self.combobox.get()}: {result}\n")
        if result == "Giải thành công!":
            self.result_text.insert(tk.END, f"Số bước giải: {len(self.path) - 1}\n")
            self.result_text.insert(tk.END, f"Depth: {depth}\n")
            self.result_text.insert(tk.END, f"Iterations: {self.iterations}\n")
            self.result_text.insert(tk.END, f"Độ phức tạp: O({self.iterations})\n")
            self.result_text.insert(tk.END, f"Thời gian thực thi: {execution_time:.10f}s\n")
            step = 0
            for x in self.path:
                self.result_text.insert(tk.END,f"Step {step}:\n")
                step += 1
                for y in x:
                    for z in y:
                        self.result_text.insert(tk.END, str(z) + " ")
                    self.result_text.insert(tk.END,"\n")
                self.result_text.insert(tk.END,"\n")

            self.result_text.insert(tk.END, "-"*80 + "\n")
        
        return
    
    def start_visualization(self):
        if self.RUN == False:
            messagebox.showerror("Lỗi","Vui lòng Run trước khi Visual")
            return
        thread = threading.Thread(target=self.run_pygame_visualization)
        thread.start()
    
    def run_pygame_visualization(self):
        pygame.init()
        self.step = 0
        all_step = len(self.path)
        screen = pygame.display.set_mode((300, 500))
        pygame.display.set_caption("23110231 - Xín Lợi Huy")
        clock = pygame.time.Clock()
        running = True
        visualization_paused = True

        button_pause = pygame.Rect(10, 430, 80, 40)
        button_prev = pygame.Rect(110, 430, 80, 40)
        button_next = pygame.Rect(210, 430, 80, 40)

        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        visualization_paused = not visualization_paused
                    if event.key == pygame.K_RIGHT and visualization_paused:
                        self.step = (self.step + 1) % all_step
                    if event.key == pygame.K_LEFT and visualization_paused:
                        self.step = (self.step - 1) % all_step

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x,y = event.pos
                    if button_pause.collidepoint(x, y):
                        visualization_paused = not visualization_paused
                    if button_next.collidepoint(x, y) and visualization_paused:
                        self.step = (self.step + 1) % all_step
                    if button_prev.collidepoint(x, y) and visualization_paused:
                        self.step = (self.step - 1) % all_step

            if not visualization_paused and self.step < all_step:
                state = self.path[self.step]
                self.draw_puzzle(screen, state, self.step, button_pause, button_prev, button_next, visualization_paused)
                self.step += 1
                pygame.time.delay(100)
            elif visualization_paused and self.step < all_step:
                state = self.path[self.step]
                self.draw_puzzle(screen, state, self.step,button_pause, button_prev, button_next, visualization_paused)
            if self.step == all_step - 1:
                visualization_paused = True
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
    
    def draw_puzzle(self, screen, table, step, button_pause, button_prev, button_next,visualization_paused):
        screen.fill(BACKGROUND_COLOR)
        dialogue_font = pygame.font.SysFont('arial', 32)
        text = dialogue_font.render(str(f"{self.combobox.get()}\nStep {step}:"), True, "black")
        screen.blit(text, (20, 20))

        pygame.draw.rect(screen, (200, 200, 200), button_pause)
        pygame.draw.rect(screen, (200, 200, 200), button_prev)
        pygame.draw.rect(screen, (200, 200, 200), button_next)

        font = pygame.font.Font(None, 24)

        pause_text = font.render("Run/Stop", True, (0, 0, 0))
        prev_text = font.render("Previous", True, (0, 0, 0))
        next_text = font.render("Next", True, (0, 0, 0))

        screen.blit(pause_text, pause_text.get_rect(center=button_pause.center))
        screen.blit(prev_text, prev_text.get_rect(center=button_prev.center))
        screen.blit(next_text, next_text.get_rect(center=button_next.center))

        for i in range(3):
            for j in range(3):
                value = table[i][j]
                rect = pygame.Rect(j * GRID_SIZE, i * GRID_SIZE + 100, GRID_SIZE, GRID_SIZE)

                if value == 0:
                    pygame.draw.rect(screen, EMPTY_TILE_COLOR, rect)
                else:
                    pygame.draw.rect(screen, TILE_COLOR, rect)

                pygame.draw.rect(screen, (0, 0, 0), rect, 3)

                if value != 0:
                    font = pygame.font.Font(None, FONT_SIZE)
                    text = font.render(str(value), True, TEXT_COLOR)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                

        pygame.display.flip()

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()

