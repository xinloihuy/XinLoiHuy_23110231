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
from copy import deepcopy
import itertools

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

# Ánh xạ hướng di chuyển từ chuỗi sang tuple tọa độ
direction_map = {
    "down": (1, 0),
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1)
}

def make_move(state, direction):
    """
    Di chuyển số 0 trong bảng theo hướng cho trước ('up', 'down', 'left', 'right').
    
    :param state: Trạng thái hiện tại (3x3 tuple)
    :param direction: Một trong các chuỗi: 'up', 'down', 'left', 'right'
    :return: Trạng thái mới nếu hợp lệ, hoặc None nếu không hợp lệ
    """
    if direction not in direction_map:
        return None  # Hướng không hợp lệ

    di, dj = direction_map[direction]
    row, col = find_zero(state)
    new_row, new_col = row + di, col + dj

    if 0 <= new_row < 3 and 0 <= new_col < 3:
        return swap(state, row, col, new_row, new_col)
    return None

class GenericSearch:
    def __init__(self, initial_state, goal_state, heuristic_type="Manhattan"):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.population_size = 200
        self.max_generations = 5000
        self.mutation_rate = 0.1
        self.elitism_count = 5
        self.max_moves = 30     # max genome length
        self.possible_moves = ['up', 'down', 'left', 'right']

        if heuristic_type.lower() == "misplaced":
            self.heuristic = misplaced_tiles
        else:
            self.heuristic = manhattan_distance

    def initialize_population(self):
        self.population = [self._create_individual() for _ in range(self.population_size)]

    def _create_individual(self):
        length = random.randint(1, self.max_moves)
        moves = [random.choice(self.possible_moves) for _ in range(length)]
        state, path = self.apply_moves(moves)
        return {"moves": moves, "state": state, "path": path}

    def apply_moves(self, moves):
        state = deepcopy(self.initial_state)
        path = [state]
        for mv in moves:
            new_state = make_move(state, mv)
            if new_state is None:
                # invalid move -> skip
                continue
            state = new_state
            path.append(state)
            if state == self.goal_state:
                break
        return state, path

    def fitness(self, individual):
        # combine heuristic and path length penalty
        h = self.heuristic(individual['state'], self.goal_state)
        l = len(individual['moves'])
        return 1 / (1 + h + 0.1 * l)

    def selection(self):
        # tournament selection
        selected = []
        for _ in range(self.population_size):
            contestants = random.sample(self.population, 3)
            winner = max(contestants, key=self.fitness)
            selected.append(winner)
        return selected

    def crossover(self, p1, p2):
        m1, m2 = p1['moves'], p2['moves']
        if len(m1) < 2 or len(m2) < 2:
            return deepcopy(random.choice([p1, p2]))
        cp1 = random.randint(1, len(m1)-1)
        cp2 = random.randint(1, len(m2)-1)
        child_moves = m1[:cp1] + m2[cp2:]
        state, path = self.apply_moves(child_moves)
        return {"moves": child_moves, "state": state, "path": path}

    def mutate(self, individual):
        moves = individual['moves'][:]
        if random.random() < 0.5 and len(moves) < self.max_moves:
            # insert a random move
            idx = random.randint(0, len(moves))
            moves.insert(idx, random.choice(self.possible_moves))
        else:
            # modify or delete
            idx = random.randrange(len(moves))
            if random.random() < 0.5:
                moves[idx] = random.choice(self.possible_moves)
            else:
                moves.pop(idx)
        state, path = self.apply_moves(moves)
        return {"moves": moves, "state": state, "path": path}

    def evolve(self, result_text: tk.Text):
        self.initialize_population()
        for gen in range(self.max_generations):
            # evaluate
            scored = [(ind, self.fitness(ind)) for ind in self.population]
            scored.sort(key=lambda x: x[1], reverse=True)
            best, best_fit = scored[0]
            result_text.insert(tk.END, f"Gen {gen}: best fit={best_fit:.4f}, moves={len(best['moves'])}\n")
            result_text.see(tk.END)
            result_text.update()

            if best['state'] == self.goal_state:
                return best['path'], gen

            # selection
            selected = self.selection()
            # generate children
            children = []
            for i in range(0, self.population_size - self.elitism_count, 2):
                c1 = self.crossover(selected[i], selected[i+1])
                c2 = self.crossover(selected[i+1], selected[i])
                children.extend([c1, c2])
            # mutate
            children = [self.mutate(c) if random.random() < self.mutation_rate else c for c in children]
            # elitism
            elites = [ind for ind, _ in scored[:self.elitism_count]]
            self.population = children + elites

        return None, self.max_generations

class NoisyGridProblem:
    def __init__(self, initial_table, goal):
        self.initial_table = initial_table
        self.goal = goal

    def goal_test(self, state):
        return state == self.goal

    def actions(self, state):
        # Find position of empty tile (0)
        x, y = None, None
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
                    break
            if x is not None:
                break

        possible_moves = []
        # Check each possible move
        if x > 0: possible_moves.append("Up")
        if x < 2: possible_moves.append("Down") 
        if y > 0: possible_moves.append("Left")
        if y < 2: possible_moves.append("Right")

        return possible_moves

    def result(self, state, action):
        # Convert state to list for modification
        state_list = [list(row) for row in state]
        
        # Find empty tile
        x, y = None, None
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
                    break
            if x is not None:
                break

        # Make the move
        if action == "Up":
            state_list[x][y], state_list[x-1][y] = state_list[x-1][y], state_list[x][y]
        elif action == "Down":
            state_list[x][y], state_list[x+1][y] = state_list[x+1][y], state_list[x][y]
        elif action == "Left":
            state_list[x][y], state_list[x][y-1] = state_list[x][y-1], state_list[x][y]
        elif action == "Right":
            state_list[x][y], state_list[x][y+1] = state_list[x][y+1], state_list[x][y]

        # Convert back to tuple and return
        return [tuple(tuple(row) for row in state_list)]

class AndOrSearch:
    def __init__(self, problem, depth_limit=30):
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
            return [state]

        if state in path or len(path) > self.depth_limit:
            return None

        path.add(state)
        print(state)
        for action in self.problem.actions(state):
            
            result_states = self.problem.result(state, action)
            plan = self.and_search(result_states, path.copy())
            if plan is not None:
                return [state] + plan
        return None

    def and_search(self, states, path):
        plan = []
        for state in states:
            sub_plan = self.or_search(state, path.copy())
            if sub_plan is None:
                return None
            plan.extend(sub_plan)
        return plan


class BacktrackingSolver:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solutions = []
        self.trace_steps = []
        self.solution_indices = []

    def solve_all(self):
        self.solutions.clear()
        self.trace_steps.clear()
        self.solution_indices.clear()
        self._recursive_backtrack({})
        return self.solutions

    def _recursive_backtrack(self, assignment):
        self.trace_steps.append(assignment.copy())
        if len(assignment) == len(self.variables):
            self.solutions.append(assignment.copy())
            self.solution_indices.append(len(self.trace_steps) - 1)
            return
        for var in self.variables:
            if var not in assignment:
                break
        for value in self.domains[var]:
            if self.constraints(var, value, assignment):
                assignment[var] = value
                self._recursive_backtrack(assignment)
                del assignment[var]

class AC3:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def revise(self, Xi, Xj):
        revised = False
        for x in self.domains[Xi][:]:
            if not any(self.constraints(Xi, x, {Xj: y}) for y in self.domains[Xj]):
                self.domains[Xi].remove(x)
                revised = True
        return revised

    def run(self):
        queue = deque((Xi, Xj) for Xi in self.variables for Xj in self.variables if Xi != Xj)
        while queue:
            Xi, Xj = queue.popleft()
            if self.revise(Xi, Xj):
                if not self.domains[Xi]:
                    return False
                for Xk in self.variables:
                    if Xk != Xi and Xk != Xj:
                        queue.append((Xk, Xi))
        return True

def default_constraint(var, value, assignment):
    if value in assignment.values():
        return False
    if var in [0, 1, 2] and value in [2, 3, 7]:
        return False
    return True

class BacktrackingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Backtracking Visualizer")

        self.variables = list(range(9))
        self.default_domains = {var: list(range(9)) for var in self.variables}
        self.domains = {var: list(range(9)) for var in self.variables}
        self.constraint_func = default_constraint

        self.solver = BacktrackingSolver(self.variables, self.domains, self.constraint_func)
        self.solver.solve_all()

        self.current_index = 0
        self.running = False

        self.create_widgets()
        self.update_grid(self.solver.trace_steps[0])
        self.update_solutions()

    def create_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        grid_frame = tk.Frame(main_frame)
        grid_frame.pack(side=tk.LEFT, padx=10)

        self.grid_labels = []
        for row in range(3):
            for col in range(3):
                label = tk.Label(grid_frame, text="", width=12, height=6,
                                 font=("Arial", 12), borderwidth=2, relief="groove", bg="white")
                label.grid(row=row, column=col, padx=5, pady=5)
                self.grid_labels.append(label)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10)

        tk.Label(right_frame, text="Thuật toán:").pack()
        self.algo_var = tk.StringVar(value="Backtracking")
        algo_options = ["Backtracking", "Generate and Test", "AC3 Only", "AC3 + Backtracking"] 
        self.algorithm_combobox = ttk.Combobox(right_frame, 
                             textvariable=self.algo_var,
                             values=algo_options,
                             width=20)
        self.algorithm_combobox.pack(pady=5)
        self.algorithm_combobox.current(0)

        tk.Label(right_frame, text="Các nghiệm đã tìm được:", font=("Arial", 12, "bold")).pack()
        self.solution_listbox = tk.Listbox(right_frame, width=30, height=10)
        self.solution_listbox.pack()

        tk.Label(right_frame, text="Ràng buộc (Python):", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.constraint_text = tk.Text(right_frame, height=6, width=40, font=("Courier", 10))
        self.constraint_text.pack()

        default_code = (
            "def custom_constraint(var, value, assignment):\n"
            "    if value in assignment.values():\n"
            "        return False\n"
            "    if var in [0, 1, 2] and value in [2, 3, 7]:\n"
            "        return False\n"
            "    return True\n"
        )
        self.constraint_text.insert(tk.END, default_code)

        tk.Button(right_frame, text="🔁 Chạy lại với ràng buộc", command=self.run_with_custom_constraint).pack(pady=5)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="◀ Lùi", command=self.step_back).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Tiến ▶", command=self.step_forward).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="⏸ Dừng", command=self.pause).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="▶ Tiếp tục", command=self.play).pack(side=tk.LEFT, padx=5)

        self.status = tk.Label(self.root, text="", font=("Arial", 12))
        self.status.pack(pady=5)

    def update_grid(self, assignment):
        for i in range(9):
            val = assignment.get(i, "")
            if isinstance(val, list):
                val = ",".join(map(str, val))
            self.grid_labels[i].config(text=str(val) if val != "" else "")
        self.status.config(text=f"Bước {self.current_index + 1} / {len(self.solver.trace_steps)}")

    def update_solutions(self):
        self.solution_listbox.delete(0, tk.END)
        for idx, step_index in enumerate(self.solver.solution_indices):
            if step_index <= self.current_index:
                solution = self.solver.trace_steps[step_index]
                row = [solution.get(i, "") for i in range(9)]
                self.solution_listbox.insert(tk.END, f"Nghiệm {idx + 1}: {row}")
                self.solution_listbox.see(tk.END)

    def step_forward(self):
        if self.current_index < len(self.solver.trace_steps) - 1:
            self.current_index += 1
            self.update_grid(self.solver.trace_steps[self.current_index])
            self.update_solutions()

    def step_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_grid(self.solver.trace_steps[self.current_index])
            self.update_solutions()

    def pause(self):
        self.running = False

    def play(self):
        self.running = True
        self.auto_step()

    def auto_step(self):
        if self.running and self.current_index < len(self.solver.trace_steps) - 1:
            self.current_index += 1
            self.update_grid(self.solver.trace_steps[self.current_index])
            self.update_solutions()
            self.root.after(300, self.auto_step)
        else:
            self.running = False

    def run_with_custom_constraint(self):
        code = self.constraint_text.get("1.0", tk.END)
        try:
            local_scope = {}
            exec(code, {}, local_scope)
            custom_fn = local_scope.get("custom_constraint")
            if not custom_fn:
                raise ValueError("Không tìm thấy hàm 'custom_constraint'.")
            self.constraint_func = custom_fn
            algo = self.algorithm_combobox.get()

            self.domains = {var: list(range(9)) for var in self.variables}

            start_time = time.time()

            if algo == "Backtracking":
                self.solver = BacktrackingSolver(self.variables, self.domains, self.constraint_func)
                self.solver.solve_all()
                print(f"Backtracking time: {time.time() - start_time:.4f} seconds")

            elif algo == "Generate and Test":
                self.solver = BacktrackingSolver(self.variables, self.domains, self.constraint_func) 
                self.solver.trace_steps = []
                self.solver.solution_indices = []
                self.solver.solutions = []
                for p in itertools.permutations(range(9)):
                    assignment = {i: p[i] for i in range(9)}
                    self.solver.trace_steps.append(assignment.copy())
                    if all(self.constraint_func(i, assignment[i], {k: assignment[k] for k in assignment if k != i}) for i in range(9)):
                        self.solver.solutions.append(assignment.copy())
                        self.solver.solution_indices.append(len(self.solver.trace_steps) - 1)
                print(f"Generate and Test time: {time.time() - start_time:.4f} seconds")

            elif algo == "AC3 Only":
                ac3 = AC3(self.variables, self.domains, self.constraint_func)
                success = ac3.run()
                if not success:
                    messagebox.showinfo("Kết quả", "AC3 phát hiện không có miền khả thi.")
                    return
                self.solver = BacktrackingSolver(self.variables, self.domains, self.constraint_func)
                self.solver.trace_steps = [dict((var, self.domains[var]) for var in self.variables)]
                self.solver.solution_indices = []
                print(f"AC3 Only time: {time.time() - start_time:.4f} seconds")

            elif algo == "AC3 + Backtracking":
                ac3 = AC3(self.variables, self.domains, self.constraint_func)
                success = ac3.run()
                if not success:
                    messagebox.showinfo("Kết quả", "AC3 phát hiện không có miền khả thi.")
                    return
                self.solver = BacktrackingSolver(self.variables, ac3.domains, self.constraint_func)
                self.solver.solve_all()
                print(f"AC3 + Backtracking time: {time.time() - start_time:.4f} seconds")

            self.current_index = 0
            self.update_grid(self.solver.trace_steps[0])
            self.update_solutions()
        except Exception as e:
            messagebox.showerror("Lỗi ràng buộc", f"Có lỗi trong đoạn mã ràng buộc:\n{e}")


# --- Cụ thể cho 9 ô ---
variables = list(range(9))
domains = {var: list(range(0, 9)) for var in variables}


class NondeterministicEightPuzzle:
    def __init__(self, num_initial_states, num_goal_states, num_visible_tiles=3):
        self.num_initial_states = num_initial_states
        self.num_goal_states = num_goal_states
        self.num_visible_tiles = num_visible_tiles
        self.initial_states = [self.generate_random_state() for _ in range(num_initial_states)]
        self.goal_states = [self.generate_random_state() for _ in range(num_goal_states)]
        self.current_states = self.initial_states.copy()
        self.visited_states = [set() for _ in range(num_initial_states)]  # Theo dõi riêng từng init state
    def generate_random_state(self):
        import random
        tiles = list(range(9))
        random.shuffle(tiles)
        return [tiles[i:i+3] for i in range(0, 9, 3)]

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def get_actions_for_states(self, state, state_idx):
        """Lấy các hành động khả dụng cho một trạng thái cụ thể"""
        row, col = self.find_blank(state)
        possible_actions = []
        
        if row > 0: possible_actions.append("up")
        if row < 2: possible_actions.append("down")
        if col > 0: possible_actions.append("left")
        if col < 2: possible_actions.append("right")
        
        # Lọc các hành động dẫn đến trạng thái chưa thăm
        filtered_actions = []
        for action in possible_actions:
            new_state = self.apply_action(copy.deepcopy(state), action)
            if new_state and tuple(tuple(row) for row in new_state) not in self.visited_states[state_idx]:
                filtered_actions.append(action)
        
        # Use filtered actions if available, otherwise use all possible actions
        actions = filtered_actions if filtered_actions else possible_actions
        # Randomly shuffle the actions
        random.shuffle(actions)
        return actions

    def apply_action(self, state, action):
        """Áp dụng hành động lên một trạng thái và trả về trạng thái mới"""
        row, col = self.find_blank(state)
        
        if action == "up" and row > 0:
            state[row][col], state[row-1][col] = state[row-1][col], state[row][col]
        elif action == "down" and row < 2:
            state[row][col], state[row+1][col] = state[row+1][col], state[row][col]
        elif action == "left" and col > 0:
            state[row][col], state[row][col-1] = state[row][col-1], state[row][col]
        elif action == "right" and col < 2:
            state[row][col], state[row][col+1] = state[row][col+1], state[row][col]
        else:
            return None
        
        return state

    def transition_model(self, action, states):
        result_states = []
        for state in states:
            row, col = self.find_blank(state)
            new_state = copy.deepcopy(state)

            if action == "up" and row > 0:
                new_state[row][col], new_state[row-1][col] = new_state[row-1][col], new_state[row][col]
            elif action == "down" and row < 2:
                new_state[row][col], new_state[row+1][col] = new_state[row+1][col], new_state[row][col]
            elif action == "left" and col > 0:
                new_state[row][col], new_state[row][col-1] = new_state[row][col-1], new_state[row][col]
            elif action == "right" and col < 2:
                new_state[row][col], new_state[row][col+1] = new_state[row][col+1], new_state[row][col]
            else:
                continue

            result_states.append(new_state)

        return result_states

    def percept(self, state):
        return [tile for row in state for tile in row if tile != 0][:self.num_visible_tiles]

    def goal_test(self):
        goal_reached = []
        non_goal = []
        for state in self.current_states:
            for goal in self.goal_states:
                if self.percept(goal) == self.percept(state):
                    goal_reached.append(state)
                    break
            else:
                non_goal.append(state)
        return goal_reached, non_goal


class NondeterministicEightPuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nondeterministic 8-Puzzle Solver")

        self.puzzle = None
        self.running = False
        self.speed = 1.0
        self.current_display_index = 0

        self.build_widgets()

    def build_widgets(self):
        # Frame chứa các control
        frm_input = tk.Frame(self.root)
        frm_input.pack(pady=10)

        # Các control nhập liệu
        tk.Label(frm_input, text="Số trạng thái khởi đầu:").grid(row=0, column=0)
        self.init_entry = tk.Entry(frm_input, width=5)
        self.init_entry.insert(0, "2")
        self.init_entry.grid(row=0, column=1)

        tk.Label(frm_input, text="Số trạng thái đích:").grid(row=0, column=2)
        self.goal_entry = tk.Entry(frm_input, width=5)
        self.goal_entry.insert(0, "3")
        self.goal_entry.grid(row=0, column=3)

        tk.Label(frm_input, text="Số ô percept:").grid(row=0, column=4)
        self.percept_entry = tk.Entry(frm_input, width=5)
        self.percept_entry.insert(0, "3")
        self.percept_entry.grid(row=0, column=5)

        tk.Label(frm_input, text="Tốc độ:").grid(row=0, column=6)
        self.speed_cb = ttk.Combobox(frm_input, values=["Chậm", "Vừa", "Nhanh"], width=7)
        self.speed_cb.set("Vừa")
        self.speed_cb.grid(row=0, column=7)

        self.run_button = tk.Button(frm_input, text="Chạy", command=self.start_solver)
        self.run_button.grid(row=0, column=8, padx=10)

        self.status_lbl = tk.Label(self.root, text="", fg="green")
        self.status_lbl.pack(pady=5)

        # Frame hiển thị trạng thái đầu và đích
        frm_states = tk.Frame(self.root)
        frm_states.pack(pady=10)

        self.init_text = tk.Text(frm_states, height=12, width=35)
        self.init_text.pack(side=tk.LEFT, padx=10)
        self.goal_text = tk.Text(frm_states, height=12, width=35)
        self.goal_text.pack(side=tk.LEFT, padx=10)

        # Frame chứa các canvas hiển thị trạng thái hiện tại
        self.frm_canvases = tk.Frame(self.root)
        self.frm_canvases.pack(pady=10)
        self.canvases = []
        
        # Kết quả quá trình giải
        self.result_text = tk.Text(self.root, height=13, width=75)
        self.result_text.pack(pady=10)

    def create_canvases(self, num_states):
        # Xóa canvas cũ nếu có
        for widget in self.frm_canvases.winfo_children():
            widget.destroy()
        self.canvases = []
        
        # Tạo canvas mới
        self.canvas_size = 120
        self.cell_size = self.canvas_size // 3
        
        for i in range(num_states):
            frame = tk.Frame(self.frm_canvases, bd=2, relief=tk.RIDGE)
            frame.grid(row=0, column=i, padx=5, pady=5)
            
            label = tk.Label(frame, text=f"State {i+1}")
            label.pack()
            
            canvas = tk.Canvas(frame, width=self.canvas_size, height=self.canvas_size)
            canvas.pack()
            self.canvases.append(canvas)

    def start_solver(self):
        try:
            init_count = int(self.init_entry.get())
            goal_count = int(self.goal_entry.get())
            percept_tiles = int(self.percept_entry.get())
        except ValueError:
            self.status_lbl.config(text="Giá trị nhập không hợp lệ!", fg="red")
            return

        selected_speed = self.speed_cb.get()
        self.speed = {"Chậm": 1.0, "Vừa": 0.3, "Nhanh": 0.1}.get(selected_speed, 0.3)

        self.status_lbl.config(text="Đang chạy thuật toán...", fg="blue")
        self.result_text.delete(1.0, tk.END)

        self.puzzle = NondeterministicEightPuzzle(init_count, goal_count, percept_tiles)
        self.display_states()
        
        # Tạo canvas tương ứng với số trạng thái khởi đầu
        self.create_canvases(init_count)

        self.running = True
        threading.Thread(target=self.solve_thread, daemon=True).start()

    def display_states(self):
        self.init_text.delete(1.0, tk.END)
        self.goal_text.delete(1.0, tk.END)

        self.init_text.insert(tk.END, "⚪ Trạng thái khởi đầu:\n\n")
        for i, state in enumerate(self.puzzle.initial_states):
            self.init_text.insert(tk.END, f"Init {i+1}:\n")
            for row in state:
                self.init_text.insert(tk.END, " ".join(str(x) for x in row) + "\n")
            self.init_text.insert(tk.END, "\n")

        self.goal_text.insert(tk.END, "🔴 Trạng thái đích:\n\n")
        for i, state in enumerate(self.puzzle.goal_states):
            self.goal_text.insert(tk.END, f"Goal {i+1}:\n")
            for row in state:
                self.goal_text.insert(tk.END, " ".join(str(x) for x in row) + "\n")
            self.goal_text.insert(tk.END, "\n")

    def update_canvases(self, states):
        for i, canvas in enumerate(self.canvases):
            canvas.delete("all")
            if i < len(states):
                state = states[i]
                for row in range(3):
                    for col in range(3):
                        x1 = col * self.cell_size
                        y1 = row * self.cell_size
                        x2 = x1 + self.cell_size
                        y2 = y1 + self.cell_size
                        color = "lightblue" if state[row][col] == 0 else "white"
                        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                        value = state[row][col]
                        if value != 0:
                            canvas.create_text(
                                x1 + self.cell_size//2,
                                y1 + self.cell_size//2,
                                text=str(value),
                                font=("Arial", 16),
                                fill="blue"
                            )

    def solve_thread(self):
        iteration = 0
        max_iterations = 30000
        solutions_found = [False] * len(self.puzzle.initial_states)
        
        # Khởi tạo với tất cả trạng thái ban đầu
        self.puzzle.current_states = self.puzzle.initial_states.copy()
        self.update_canvases(self.puzzle.current_states)
        
        while self.running and iteration < max_iterations and not all(solutions_found):
            self.result_text.see(tk.END)
            
            # Tìm hành động chung từ các trạng thái chưa hoàn thành
            common_action = None
            for i, state in enumerate(self.puzzle.current_states):
                if not solutions_found[i]:
                    actions = self.puzzle.get_actions_for_states(state, i)
                    if actions:
                        common_action = actions[0]  # Lấy hành động đầu tiên của state đầu tiên chưa hoàn thành
                        break
            
            # Xử lý từng trạng thái với hành động chung
            new_states = []
            for i, state in enumerate(self.puzzle.current_states):
                if solutions_found[i]:
                    new_states.append(state)  # Giữ nguyên nếu đã giải xong
                    continue
                    
                # Kiểm tra goal cho từng state
                goal_reached = False
                for goal in self.puzzle.goal_states:
                    if state == goal:
                        solutions_found[i] = True
                        goal_reached = True
                        self.result_text.insert(tk.END, f"✅ State {i + 1} đạt mục tiêu:\n")
                        for row in state:
                            self.result_text.insert(tk.END, " ".join(str(x) for x in row) + "\n")
                        self.result_text.insert(tk.END, "\n")
                        new_states.append(state)
                        break
                
                if not goal_reached:
                    if common_action:  # Áp dụng hành động chung nếu có
                        new_state = self.puzzle.apply_action(copy.deepcopy(state), common_action)
                        
                        if new_state and new_state != state:
                            new_states.append(new_state)
                            # Thêm vào visited_states
                            state_tuple = tuple(tuple(row) for row in new_state)
                            self.puzzle.visited_states[i].add(state_tuple)
                        else:
                            new_states.append(state)
                    else:
                        new_states.append(state)
                        self.result_text.insert(tk.END, f"⛔ State {i+1}: Không còn hành động khả dụng\n")
            
            self.puzzle.current_states = new_states
            self.update_canvases(new_states)
            time.sleep(self.speed)
            iteration += 1

            if all(solutions_found):
                break

        if iteration >= max_iterations:
            self.status_lbl.config(text="Đạt giới hạn vòng lặp.", fg="orange")
        else:
            self.status_lbl.config(text="Hoàn thành!", fg="green")
        self.running = False



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
        
        

    def create_widgets(self):
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Algorithm
        self.label_algorithm = tk.Label(control_frame, text="Algorithm:")
        self.label_algorithm.grid(row=0, column=0, padx=5)
        
        self.algorithms = ["BFS", "UCS", "DFS", "DLS", "IDDFS", "Greedy Search", "A*", "IDA*", "SHC", "SAHC", "STOHC", "SA", "BS", "GA", "And_or_graph_search","BackTracking","Nondeterministic 8-Puzzle"]
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

        if selected_algorithm in ["Greedy Search", "A*", "IDA*", "SHC", "SAHC", "STOHC", "SA", "BS", "GA"]:
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
            heuristic_type = self.combobox_heuristics.get()
            problem = NoisyGridProblem(self.initial_table, self.final_table)
            AND_OR_Searcher = AndOrSearch(problem)
            self.path, self.iterations = AND_OR_Searcher.search()
        elif algorithm == "GA":
            heuristic_type = self.combobox_heuristics.get()
            GENERIC_Searcher = GenericSearch(self.initial_table,self.final_table, heuristic_type)
            self.path, self.iterations = GENERIC_Searcher.evolve(self.result_text)
        elif algorithm == "BackTracking":
            window = tk.Tk()
            app = BacktrackingGUI(window)
            window.mainloop()
        elif algorithm == "Nondeterministic 8-Puzzle":
            window2 = tk.Tk()
            app = NondeterministicEightPuzzleGUI(window2)
            window2.mainloop()
        else:
            messagebox.showerror("Lỗi", "Thuật toán chưa được triển khai")
            return
        
        if self.path is None:
            result = "Không giải được!"
            self.RUN = False
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
            # self.result_text.see(tk.END)
        
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

