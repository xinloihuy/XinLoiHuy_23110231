# BÁO CÁO CÁ NHÂN DỰ ÁN 8-PUZZLE AI
Tên: Xín Lợi Huy
MSSV: 23110231
## 1. Mục tiêu

Mục tiêu của project là áp dụng các nhóm thuật toán AI khác nhau để giải bài toán 8-puzzle và so sánh hiệu suất giữa chúng. Qua đó, nó giúp em hiểu rõ hơn về cách các thuật toán tìm kiếm hoạt động cũng như hiệu quả của chúng trong việc giải quyết một bài toán cụ thể.

## 2. Nội dung

### 2.1. Các thuật toán Tìm kiếm không có thông tin

**Thuật toán sử dụng:**
- BFS (Breadth-First Search)
- DFS (Depth-First Search)
- UCS (Uniform Cost Search)
- IDDFS (Iterative Deepening Depth-First Search)

**Minh họa:**
#### BFS (Breadth-First Search)
![BFS](gif_files/BFS.gif)

#### DFS (Depth-First Search)
![DFS](gif_files/DFS.gif)

#### UCS (Uniform-Cost Search)
![UCS](gif_files/UCS.gif)

#### IDDFS (Iterative Deepening DFS)
![IDDFS](gif_files/IDDFS.gif)

- (Biểu đồ so sánh số nút đã duyệt, độ sâu, thời gian thực thi.)

**Các thành phần chính của bài toán tìm kiếm:**
- **State:** vị trí hiện tại của các ô số.
- **Initial state:** trạng thái ban đầu do người dùng nhập hoặc tạo ngẫu nhiên.
- **Goal state:** trạng thái đích chuẩn `[1,2,3,4,5,6,7,8,0]`.
- **Actions:** di chuyển ô trống lên/xuống/trái/phải.
- **Transition model:** thực hiện hành động sẽ tạo ra trạng thái mới.
- **Path cost:** tính bằng số bước di chuyển (với UCS thì trọng số có thể bằng 1 cho mỗi bước).
- **Giải pháp (solution):** là chuỗi các hành động dẫn từ trạng thái bắt đầu đến trạng thái mục tiêu.

**Nhận xét:**
- BFS luôn tìm ra đường đi ngắn nhất nhưng tốn bộ nhớ.
- DFS có thể bị lặp vô hạn và không tối ưu.
- UCS hiệu quả nếu chi phí hành động khác nhau (ở đây do chi phí bằng nhau nên giống BFS).
- IDDFS tận dụng được ưu điểm của DFS và BFS nhưng mất thời gian do duyệt lại nhiều lần.

---

### 2.2. Các thuật toán Tìm kiếm có thông tin

**Thuật toán sử dụng:**
- A* Search
- IDA* (Iterative Deepening A*)
- Greedy Best-First Search

**Minh họa:**
#### Greedy Best-First Search
![GreedySearch](gif_files/GreedySearch.gif)

#### A* Search
![AStar](gif_files/AStar.gif)

#### IDA* (Iterative Deepening A*)
![IDAStar](gif_files/IDAStar.gif)

- (Biểu đồ hiệu suất, số nút duyệt, thời gian)

**Các thành phần chính của bài toán tìm kiếm:**
- **State:** Vị trí hiện tại của các ô số trên bảng.
- **Initial state:** Trạng thái ban đầu do người dùng nhập hoặc tạo ngẫu nhiên.
- **Goal state:** Trạng thái đích chuẩn `[1,2,3,4,5,6,7,8,0]`.
- **Actions:** Các thao tác di chuyển ô trống lên, xuống, trái, phải.
- **Transition model:** Thực hiện một hành động sẽ sinh ra trạng thái mới.
- **Path cost:** Tổng số bước di chuyển hoặc chi phí cho mỗi hành động.
- **Solution:** Chuỗi các hành động dẫn từ trạng thái bắt đầu đến trạng thái mục tiêu.

**Heuristic dùng:**
- Số ô sai vị trí (Misplaced Tiles)
- Tổng khoảng cách Manhattan

**Nhận xét:**
- A* rất mạnh và tối ưu, tìm được đường đi nhanh chóng.
- Greedy tìm rất nhanh nhưng không tối ưu.
- IDA* tiết kiệm bộ nhớ hơn A* nhưng chậm hơn.

---

### 2.3. Tìm kiếm cục bộ (Local Search)

**Thuật toán sử dụng:**
- Simple Hill Climbing
- Steepest Ascent Hill Climbing
- Stochastic Hill Climbing
- Simulated Annealing
- Beam Search
- Genetic Algorithm

**Minh họa:**
#### Simple Hill Climbing (SHC)
![SHC](gif_files/SHC.gif)

#### Steepest-Ascent Hill Climbing (SAHC)
![SAHC](gif_files/SAHC.gif)

#### Stochastic Hill Climbing
![StoHC](gif_files/STOHC.gif)

#### Simulated Annealing
![SA](gif_files/SA.gif)

#### Beam Search 
![BS](gif_files/BS.gif)

#### Genetic Search 
![GA](gif_files/GA.gif)

**Các thành phần chính của bài toán tìm kiếm cục bộ:**

- **State:** Trạng thái hiện tại của các ô trên bảng.
- **Initial state:** Một trạng thái ngẫu nhiên được chọn làm điểm bắt đầu.
- **Goal state:** Trạng thái đích `[1,2,3,4,5,6,7,8,0]` hoặc tùy chọn.
- **Actions:** Thay đổi trạng thái bằng cách di chuyển ô trống để cải thiện giá trị heuristic.
- **Solution:** Dãy các trạng thái liên tiếp mà mỗi trạng thái sau đều tốt hơn trạng thái trước (theo heuristic), cho đến khi không thể cải thiện thêm nữa.

**Nhận xét:**
- Hill climbing dễ bị kẹt do local maximum.
- Simulated Annealing đôi khi thoát được nhưng không đảm bảo tìm được đường.
- Genetic Algorithm đa dạng trong giải pháp tìm đường nhưng việc điều chỉnh các tham số và thiết kế hàm fitness đòi hỏi nhiều thử nghiệm để đạt hiệu quả tốt.
- Beam Search cho kết quả khá tốt nếu lựa chọn số chùm phù hợp, tuy nhiên không đảm bảo lời giải tối ưu nếu chùm quá nhỏ.

---

### 2.4. Tìm kiếm trong môi trường phức tạp

**Thuật toán sử dụng:**
- And-Or Search
- Belief State
- Search with Partial Observation

**Minh họa:**
#### And Or Graph Search 
![AND_OR](gif_files/AND_OR.gif)

**Các thành phần chính của bài toán tìm kiếm trong môi trường phức tạp:**
- **State:** Trạng thái có thể là một tập hợp các trạng thái có thể xảy ra (Belief state) do không quan sát đầy đủ.
- **Initial state:** Trạng thái ban đầu, có thể không xác định chính xác.
- **Goal state:** Trạng thái mục tiêu cần đạt được.
- **Actions:** Các hành động được lựa chọn dựa trên thông tin không đầy đủ.
- **Solution:** Chuỗi chiến lược giúp đạt được trạng thái đích trong điều kiện thiếu thông tin hoặc không xác định.

**Nhận xét:**
- Không áp dụng mạnh và không tối ưu cho 8-puzzle vì đây là bài toán xác định và quan sát đầy đủ.
- Tuy nhiên, cần triển khai để hiểu cách thuật toán này hoạt động.

---

### 2.5. Tìm kiếm trong môi trường có ràng buộc

**Thuật toán sử dụng:**
- Backtracking
- Generate and Test
- AC-3

**Minh họa:**
#### Generate And Test
![GAT](gif_files/Generate_And_Test.gif)

#### Backtracking
![BACKTRACKING](gif_files/Backtracking.gif)

#### AC3 and AC3+Backtracking
![AC3_BACKTRACKING](gif_files/AC3.gif)

**Các thành phần chính của bài toán tìm kiếm có ràng buộc:**
- **State:** Trạng thái hiện tại của bài toán, trong đó các biến phải tuân theo một số ràng buộc nhất định.
- **Initial state:** Trạng thái khởi tạo ban đầu của các biến.
- **Goal state:** Trạng thái mà tất cả các ràng buộc đều được thỏa mãn.
- **Constraints:** Các điều kiện ràng buộc giữa các biến (trạng thái) cần được đảm bảo trong suốt quá trình tìm kiếm.
- **Solution:** Trạng thái đáp ứng đầy đủ các ràng buộc đã đặt ra.

**Nhận xét:**
- Không phổ biến cho 8-puzzle nhưng có thể dùng để thử nghiệm xem mỗi trạng thái có hợp lệ hay không theo ràng buộc.

---

### 2.6. Học củng cố (Reinforcement Learning)

**Thuật toán sử dụng:**
- Q-learning
- Reinforcement Learning (Value iteration, Policy iteration)

**Nhận xét:**
- Cần nhiều thời gian để train.
- Q-learning học được chính sách để giải puzzle, tuy chậm nhưng đáng thử.

---

## 3. Kết luận

Qua project này, em đã:
- Hiểu rõ hơn các loại thuật toán AI và dần quen với cách áp dụng chúng vào các bài toán khác.
- Thấy rõ được điểm mạnh và điểm yếu của từng nhóm thuật toán.
- Cải thiện kỹ năng lập trình và có cái nhìn mới về AI.

---

## TÀI LIỆU THAM KHẢO
[1] GeeksforGeeks, "8 puzzle problem using branch and bound", GeeksforGeeks, 23/02/2025. [Online]. Trích dẫn: https://www.geeksforgeeks.org/8-puzzle-problem-using-branch-and-bound/

[2] GeeksforGeeks, "Genetic Algorithms", GeeksforGeeks, 08/03/2024. [Online]. Trích dẫn: https://www.geeksforgeeks.org/genetic-algorithms/

[3] GeeksforGeeks, "What is Simulated Annealing", GeeksforGeeks, 12/09/2024. [Online]. Trích dẫn: https://www.geeksforgeeks.org/what-is-simulated-annealing/

[4] GeeksforGeeks, "A* Search Algorithm", GeeksforGeeks, 30/07/2024. [Online]. Trích dẫn: https://www.geeksforgeeks.org/a-search-algorithm/

[5] GeeksforGeeks, "Uniform-Cost Search (Dijkstra for large Graphs)", GeeksforGeeks, 20/04/2023. [Online]. Trích dẫn: https://www.geeksforgeeks.org/uniform-cost-search-dijkstra-for-large-graphs/

[6] GeeksforGeeks, "Hill Climbing in Artificial Intelligence", GeeksforGeeks, 10/10/2024. [Online]. Trích dẫn: https://www.geeksforgeeks.org/introduction-hill-climbing-artificial-intelligence/

[7] GeeksforGeeks, "Constraint Satisfaction Problems (CSP) in Artificial Intelligence", GeeksforGeeks, 05/05/2025. [Online]. Trích dẫn: https://www.geeksforgeeks.org/constraint-satisfaction-problems-csp-in-artificial-intelligence/
