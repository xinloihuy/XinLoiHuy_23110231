import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pygame
import time
import random
import sys
from collections import deque
import heapq
import math
import copy

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
            if new_cost >= best_cost:
                break
            else:
                best_state = next_state
                best_cost = new_cost
                break

        if best_state is None:
            return None, iterations 
        
        current_state = best_state
        current_cost = best_cost
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

def StoHC(start, goal, type):
    iterations = 0
    if type == "Misplaced Tiles":
        heuristic = misplaced_tiles
    else:
        heuristic = manhattan_distance

    current_state = start
    current_cost = heuristic(current_state, goal)
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
        current_cost = best_cost
        path.append(current_state)

def SA(start,final,type):
    if type == "Misplaced Tiles":
        heuristic = misplaced_tiles
    else:
        heuristic = manhattan_distance

    T = 1e6
    alpha = 0.995
    iterations = 0
    max_iters = 100000

    current_state = start
    current_cost = heuristic(current_state,final)

    while T > 1e-5 and iterations < max_iters:
        iterations += 1

        if current_cost == 0:
            return current_state, iterations

        neighbors = new_states(current_state)
        next_state = random.choice(neighbors)
        next_cost = heuristic(next_state,final)

        if next_cost < current_cost or random.random() < math.exp((current_cost - next_cost) / T):
            current_state = next_state
            current_cost = next_cost

        T *= alpha

    return None, iterations


def BS(start, goal, type, beam_width=2):
    if type == "Misplaced":
        heuristic = misplaced_tiles
    else:
        heuristic = manhattan_distance
    
    beam = [(heuristic(start, goal), start, [start])]
    visited = set([start])
    iterations = 0
    
    while beam:
        iterations += 1
        candidates = []
        
        for _, state, path in beam:
            if state == goal:
                return path, iterations
        
            for next_state in new_states(state):
                if next_state not in visited:
                    visited.add(next_state)
                    cost = heuristic(next_state, goal)
                    candidates.append((cost, next_state, path + [next_state]))
        
        if not candidates:
            return None, iterations
        
        candidates.sort(key=lambda x: x[0])
        
        # Chọn beam_width trạng thái tốt nhất cho beam tiếp theo
        beam = candidates[:beam_width]
    
    return None, iterations

class AndOrSearch:
    def __init__(self, problem, depth_limit=50):
        self.problem = problem
        self.path = None
        self.iterations = 0
        self.depth_limit = depth_limit

    def search(self):
        self.path = self.or_search(self.problem.initial_table, set())
        return (self.path, self.iterations) if self.path is not None else (None, self.iterations)

    def or_search(self, state, path):
        self.iterations += 1
        if self.problem.goal_test(state):
            return []  # Trả về kế hoạch rỗng nếu đạt trạng thái mục tiêu
        if state in path or len(path) > self.depth_limit:
            return None  # Tránh vòng lặp vô hạn hoặc đệ quy quá sâu

        path.add(state)
        for action in self.problem.actions(state):
            result_states = self.problem.result(state, action)
            plan = self.and_search(result_states, path.copy())
            if plan is not None:
                return [action] + plan  # Chuyển đổi dict thành danh sách
        return None  # Nếu không tìm thấy kế hoạch

    def and_search(self, states, path):
        plan = []  # Chuyển sang danh sách thay vì dict
        for state in states:
            sub_plan = self.or_search(state, path.copy())
            if sub_plan is None:
                return None  # Nếu một trạng thái không có kế hoạch, thất bại
            plan.extend(sub_plan)  # Thêm kế hoạch vào danh sách
        return plan  # Trả về danh sách thay vì dict

class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("23110231 - Xín Lợi Huy")
        self.root.geometry("1200x600")

        self.create_widgets()
        
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
        self.searcher = AndOrSearch(self)

    def create_widgets(self):
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Algorithm
        self.label_algorithm = tk.Label(control_frame, text="Algorithm:")
        self.label_algorithm.grid(row=0, column=0, padx=5)
        
        self.algorithms = ["BFS", "UCS", "DFS", "DLS", "IDDFS", "Greedy Search", "A*", "IDA*", "SHC", "SAHC", "STOHC", "SA", "BS","And_or_graph_search"]
        self.combobox = ttk.Combobox(control_frame, values=self.algorithms)
        self.combobox.grid(row=0, column=1, padx=5)
        self.combobox.current(0)
        
        # Depth limit (hidden)
        self.input_depth_limit_frame = tk.Frame(control_frame)
        self.input_depth_limit_frame.grid(row=0, column=2, padx=5)
        self.input_depth_limit_label = tk.Label(self.input_depth_limit_frame, text="Depth Limit:")
        self.input_depth_limit = tk.Entry(self.input_depth_limit_frame, width=10)
        self.input_depth_limit_label.pack(side=tk.LEFT)
        self.input_depth_limit.pack(side=tk.LEFT)
        self.input_depth_limit_frame.grid_remove()

        # Heuristics (hidden)
        self.combobox_heuristics_frame = tk.Frame(control_frame)
        self.combobox_heuristics_frame.grid(row=0, column=3, padx=5)
        self.label_heuristic = tk.Label(self.combobox_heuristics_frame, text="Heuristic:")
        self.combobox_heuristics = ttk.Combobox(self.combobox_heuristics_frame, 
                                              values=["Manhattan Distance", "Misplaced Tiles"],
                                              width=25)
        self.label_heuristic.pack(side=tk.LEFT)
        self.combobox_heuristics.pack(side=tk.LEFT)
        self.combobox_heuristics_frame.grid_remove()
        self.combobox_heuristics.current(0)

        # Buttons
        self.btn_run = tk.Button(control_frame, text="Run", command=self.run, bg="lightgreen")
        self.btn_run.grid(row=0, column=4, padx=5)
        
        self.btn_random_input = tk.Button(control_frame, text="Random Input", 
                                       command=lambda: [self.randomize(), self.change_random_color()], 
                                       bg="#FAF884")
        self.btn_random_input.grid(row=0, column=5, padx=5)

        self.btn_visual = tk.Button(control_frame, text="Visual", command=self.start_visualization, bg="yellow")
        self.btn_visual.grid(row=0, column=6, padx=5)

        # Puzzle frames
        puzzle_frame = tk.Frame(self.root)
        puzzle_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Initial state
        initial_frame = tk.LabelFrame(puzzle_frame, text="Initial State")
        initial_frame.grid(row=0, column=0, padx=20, pady=10)
        self.initial_entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = tk.Entry(initial_frame, width=4, font=('Arial', 24), justify='center')
                entry.grid(row=i, column=j, ipady=10, padx=2, pady=2)
                row_entries.append(entry)
            self.initial_entries.append(row_entries)

        # Final state
        final_frame = tk.LabelFrame(puzzle_frame, text="Goal State")
        final_frame.grid(row=0, column=1, padx=20, pady=10)
        self.final_entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = tk.Entry(final_frame, width=4, font=('Arial', 24), justify='center')
                entry.grid(row=i, column=j, ipady=10, padx=2, pady=2)
                row_entries.append(entry)
            self.final_entries.append(row_entries)
        
        # Set default final state
        self.set_default_final_state()

        # Result frame
        result_frame = tk.Frame(self.root)
        result_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="nsew")

        self.result_text = tk.Text(result_frame, height=15, width=100, background="lightblue", wrap="none")
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar = tk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=v_scrollbar.set)
        self.result_text.config(font=("Arial", 12, "bold"), fg="black")

        # Configure grid weights
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.combobox.bind("<<ComboboxSelected>>", self.on_algorithm_change)

    def set_default_final_state(self):
        default_final = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
        for i in range(3):
            for j in range(3):
                self.final_entries[i][j].delete(0, tk.END)
                self.final_entries[i][j].insert(0, str(default_final[i][j]))

        
    def change_random_color(self):
        rainbow_colors = ["#FFDDC1", "#FFABAB", "#FFC3A0", "#D5AAFF", "#85E3FF", "#B9FBC0", "#FAF884"]
        color = random.choice(rainbow_colors)
        self.btn_random_input.config(bg=color)
    
    def on_algorithm_change(self, event=None):
        selected_algorithm = self.combobox.get()
        
        if selected_algorithm == "DLS":
            self.input_depth_limit_frame.grid()
        else:
            self.input_depth_limit_frame.grid_remove()

        if selected_algorithm in ["Greedy Search", "A*", "IDA*", "SHC", "SAHC", "STOHC", "SA", "BS","And_or_graph_search"]:
            self.combobox_heuristics_frame.grid()
        else:
            self.combobox_heuristics_frame.grid_remove()


    def count_inversions(self, puzzle):
        inversions = 0
        for i in range(len(puzzle)):
            if puzzle[i] == 0:
                continue
            for j in range(i + 1, len(puzzle)):
                if puzzle[j] != 0 and puzzle[i] > puzzle[j]:
                    inversions += 1
        return inversions

    def is_solvable(self, puzzle):
        flat_puzzle = [num for num in puzzle if num != 0]
        return self.count_inversions(flat_puzzle) % 2 == 0

    def randomize(self):
        self.RUN = False
        while True:
            numbers = list(range(9))
            random.shuffle(numbers)
            
            if self.is_solvable(numbers):
                puzzle = [numbers[i*3:(i+1)*3] for i in range(3)]
                for i in range(3):
                    for j in range(3):
                        self.initial_entries[i][j].delete(0, tk.END)
                        self.initial_entries[i][j].insert(0, str(puzzle[i][j]))
                return
    
    def get_init_state(self):
        state = []
        for i in range(3):
            row = []
            for j in range(3):
                value = self.initial_entries[i][j].get()
                row.append(int(value) if value.isdigit() else 0)
            state.append(tuple(row))
        return tuple(state)

    def run(self):
        self.RUN = True
        selected_algorithm = self.combobox.get()
        input_data = self.get_init_state()
        input_depth = self.input_depth_limit.get()
        final_state = []
        for i in range(3):
            row = []
            for j in range(3):
                value = self.final_entries[i][j].get()
                row.append(int(value) if value.isdigit() else 0)
            final_state.append(tuple(row))
        
        self.final_table = tuple(final_state)
        
        if not selected_algorithm:
            messagebox.showerror("Lỗi", "Vui lòng chọn thuật toán!")
            return
        
        if not input_data:
            messagebox.showerror("Lỗi", "Vui lòng nhập dữ liệu đầu vào!")
            return
        
        if not input_depth and selected_algorithm == "DLS":
            messagebox.showerror("Lỗi", "Vui lòng nhập depth limit!")
            return

        self.run_algorithm(selected_algorithm)

        if self.path is None:
            return
        
    
    def run_algorithm(self, algorithm):
        self.initial_table = self.get_init_state()
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
        elif algorithm == "STOHC":  #Stochastic Hill Climbing
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = StoHC(self.initial_table, self.final_table, type=heuristic_type)
        elif algorithm == "SA":     #Simulated Annealing
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = SA(self.initial_table, self.final_table, type=heuristic_type)
        elif algorithm == "BS":     #Beam Search
            heuristic_type = self.combobox_heuristics.get()
            self.path, self.iterations = BS(self.initial_table, self.final_table, type=heuristic_type)
        elif algorithm == "And_or_graph_search":
            self.path, self.iterations = self.searcher.search()
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
            self.result_text.insert(tk.END, f"Thời gian thực thi: {execution_time:.30f}s\n")
            step = 0
            for x in self.path:
                self.result_text.insert(tk.END,f"Step {step}:\n")
                step += 1
                for y in x:
                    for z in y:
                        self.result_text.insert(tk.END, str(z) + " ")
                    self.result_text.insert(tk.END,"\n")
                self.result_text.insert(tk.END,"\n")

            self.result_text.insert(tk.END, "="*300 + "\n")
        
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
    
    def goal_test(self, state):
        return state == self.final_table
    
    def actions(self, state):
        x, y = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0][0]
        moves = []
        if x > 0: moves.append("Up")
        if x < 2: moves.append("Down")
        if y > 0: moves.append("Left")
        if y < 2: moves.append("Right")
        return moves
    
    def result(self, state, action):
        x, y = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0][0]
        new_state = [list(row) for row in state]  # Chuyển tuple thành list để có thể thay đổi
        if action == "Up": new_state[x][y], new_state[x-1][y] = new_state[x-1][y], new_state[x][y]
        if action == "Down": new_state[x][y], new_state[x+1][y] = new_state[x+1][y], new_state[x][y]
        if action == "Left": new_state[x][y], new_state[x][y-1] = new_state[x][y-1], new_state[x][y]
        if action == "Right": new_state[x][y], new_state[x][y+1] = new_state[x][y+1], new_state[x][y]
        return [tuple(tuple(row) for row in new_state)]  # Chuyển lại thành tuple để đảm bảo bất biến
if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()

