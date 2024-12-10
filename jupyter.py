def ucs(graph, start, goal):
    priority_queue = [(start, [start])]
    visited = []

    while priority_queue:
        # Get the next node to process
        current, path = priority_queue.pop(0)

        # Skip if the node is already visited
        if current in visited:
            continue

        visited.append(current)

        # Goal check
        if current == goal:
            return path

        # Explore neighbors
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                priority_queue.append((neighbor, path + [neighbor]))
    return []

---------------------------------------------------

# Uniform Cost Search (UCS) Algorithm Implementation
def ucs(graph, start, goal):
    """
    Perform Uniform Cost Search to find the shortest path from start to goal.

    Parameters:
        graph (dict): Graph as an adjacency list with edge weights.
        start (str): Starting node.
        goal (str): Goal node.

    Returns:
        tuple: (path, cost) where path is a list of nodes and cost is the total cost.
    """
    # Priority queue implemented using a list
    priority_queue = [(0, start, [start])]
    visited = []

    while priority_queue:
        # Find the node with the lowest cost
        priority_queue.sort(key=lambda item: item[0])
        lowest_cost_item = priority_queue.pop(0)
        cost, current, path = lowest_cost_item

        # Skip if the node is already visited
        if current in visited:
            continue

        visited.append(current)

        # Goal check
        if current == goal:
            return path, cost

        # Explore neighbors
        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in visited:
                priority_queue.append((cost + weight, neighbor, path + [neighbor]))

    # If the goal is not reachable
    return None, float('inf')

# Example Usage
if __name__ == "__main__":
    # First Graph
    graph1 = {
        'A': {'B': 10, 'C': 6, 'D': 3},
        'B': {'E': 5, 'F': 3},
        'C': {'G': 3, 'H': 4},
        'D': {'H': 5},
        'E': {},
        'F': {},
        'G': {},
        'H': {}
    }

    path, cost = ucs(graph1, 'A', 'G')
    print("Path from A to G:", path, "with cost:", cost)

    # Second Graph
    graph2 = {
        'A': {'B': 20, 'D': 30},
        'B': {'A': 20, 'C': 25, 'D': 70},
        'D': {'A': 30, 'B': 70, 'F': 15, 'G': 20, 'E': 35},
        'C': {'B': 25, 'E': 40},
        'E': {'D': 35, 'C': 40, 'G': 50, 'H': 70},
        'F': {'D': 15, 'G': 10},
        'G': {'F': 10, 'D': 20, 'E': 50, 'H': 60},
        'H': {'E': 70, 'G': 60}
    }

    path, cost = ucs(graph2, 'A', 'H')
    print("Path from A to H:", path, "with cost:", cost)

--------------------------------------------------
def creategraph(graph):
    n=int(input("Plase enter number of nodes: "))
    for i in range(n):
        node=input("Plase enter the node like A:B,C,D: ")
        l=node.split(":")
        graph[l[0]]=l[1].split(",")
    return graph
def DFS(graph, start,goal):
    result=['Posible',[]]
    viseted=[]
    stack=[start]
    while stack:
        curent_node=stack.pop()
        if curent_node not in viseted:
            viseted.append(curent_node)
            if curent_node==goal:
                result[0]='Posible'
                break
            if curent_node in graph:
                for n in reversed(graph[curent_node]):
                    if n not in viseted:
                        stack.append(n)
    result[1]=viseted
    return result
graph={}
graph=creategraph(graph)
start=input("start: ")
goal=input("goal: ")
print(DFS(graph,start,goal))



--------------------------------------------------------

def CreateGraph(graph):
    n = int(input("please enter number of nodes:"))
    for i in range(n):
        node = input("please enter the node and its childs in the form of node:child1, child2,...")
        l = node.split(':')
        graph[l[0]] = l[1].split(',')
    return graph
def BFS(graph, start, goal):
    result = ['impossible',list()]
    reached = list()
    Frontier = list()
    Frontier.append(start)
    reached.append(start)
    while Frontier:
        currentNode = Frontier.pop(0)
        if currentNode not in graph.keys():
            continue
  
        for node in graph[currentNode]:
            if node==goal:
                result[0] = 'Possible'
                reached.append(node)
                break 
            if node not in reached:
                reached.append(node)
                Frontier.append(node)
        if result[0]=='Possible':
            break


    result[1]=reached
    return result
graph = dict()
graph = CreateGraph(graph)
start = input('please enter the start node')
goal  = input('please enter the goal node')
result = BFS(graph, start,goal)
print(result[0])
print(result[1])



-----------------------------------------------------


# A* Algorithm Implementation

def a_star(graph, start, goal, heuristic):
    open_set = {start: heuristic[start]}  # Priority queue with f(n) = g(n) + h(n)
    g_costs = {start: 0}  # Cost from start to each node
    parents = {}  # To reconstruct the path

    while open_set:
        # Get the node with the smallest f(n) value
        current = min(open_set, key=open_set.get)
        current_f_cost = open_set.pop(current)

        # If the goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parents.get(current)
            return path[::-1], g_costs[goal]

        # Explore neighbors
        for neighbor, cost in graph.get(current, {}).items():
            tentative_g_cost = g_costs[current] + cost

            # If a better g_cost is found, update the neighbor's data
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic[neighbor]
                open_set[neighbor] = f_cost
                parents[neighbor] = current

    # If the goal is not reachable
    return None, float('inf')

# Second Graph and Heuristic
graph2 = {
    'A': {'B': 20, 'D': 30},
    'B': {'A': 20, 'C': 25, 'D': 70},
    'D': {'A': 30, 'B': 70, 'F': 15, 'G': 20, 'E': 35},
    'C': {'B': 25, 'E': 40},
    'E': {'D': 35, 'C': 40, 'G': 50, 'H': 70},
    'F': {'D': 15, 'G': 10},
    'G': {'F': 10, 'D': 20, 'E': 50, 'H': 60},
    'H': {'E': 70, 'G': 60}
}
heuristic2 = {'A': 90, 'B': 95, 'C': 88, 'D': 77, 'E': 50, 'F': 55, 'G': 50, 'H': 0}

path, cost = a_star(graph2, 'A', 'H', heuristic2)
print("Path from A to H:", path, "with cost:", cost)




------------------------------------------------------



# Graph Definition
graph = {
    'A': {'B': 10, 'C': 6, 'D': 3},
    'B': {'E': 5, 'F': 3},
    'C': {'G': 3, 'H': 4},
    'D': {'H': 5},
    'E': {},
    'F': {},
    'G': {},
    'H': {}
}

# Heuristic Definition
Heuristic = {'A': 8, 'B': 8, 'C': 2, 'D': 7, 'E': 3, 'F': 6, 'G': 0, 'H': 3}

# A* Algorithm
def A_star(graph, start, goal, h):
    parent = dict()
    frontier = {start: h[start]}
    visited = {start: h[start]}
    while frontier:
        sorted_queue = sorted(frontier, key=frontier.get)  # sorted_queue is a list
        
        node = sorted_queue[0]
        g = frontier.pop(node) - h[node]
        if node == goal:
            return parent
        for n in graph.get(node):
            if n not in visited:
                F = h[n] + graph[node][n] + g
                visited[n] = F
                frontier[n] = F
                parent[n] = node
            else:
                F = h[n] + graph[node][n] + g
                if visited[n] >= F:
                    visited[n] = F
                    frontier[n] = F
                    parent[n] = node

# Traceback Function
def traceback(parent, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

# First Example Execution
parent = A_star(graph, 'A', 'G', Heuristic)
print("Path from A to G:", traceback(parent, 'A', 'G'))  # Output: ['A', 'C', 'G']

# Second Graph Definition
graph2 = {
    'A': {'B': 20, 'D': 30},
    'B': {'A': 20, 'C': 25, 'D': 70},
    'D': {'A': 30, 'B': 70, 'F': 15, 'G': 20, 'E': 35},
    'C': {'B': 25, 'E': 40},
    'E': {'D': 35, 'C': 40, 'G': 50, 'H': 70},
    'F': {'D': 15, 'G': 10},
    'G': {'F': 10, 'D': 20, 'E': 50, 'H': 60},
    'H': {'E': 70, 'G': 60}
}

# Second Heuristic Definition
Heuristic2 = {'A': 90, 'B': 95, 'C': 88, 'D': 77, 'E': 50, 'F': 55, 'G': 50, 'H': 0}

# Second Example Execution
parent = A_star(graph2, 'A', 'H', Heuristic2)
print("Path from A to H:", traceback(parent, 'A', 'H'))  # Output: ['A', 'D', 'G', 'H']




----------------------------------------


# Graph Definition
graph = {
    'A': {'B': 10, 'C': 6, 'D': 3},
    'B': {'E': 5, 'F': 3},
    'C': {'G': 3, 'H': 4},
    'D': {'H': 5},
    'E': {},
    'F': {},
    'G': {},
    'H': {}
}

# Heuristic Definition
Heuristic = {'A': 8, 'B': 8, 'C': 2, 'D': 7, 'E': 3, 'F': 6, 'G': 0, 'H': 3}

# Greedy Best-First Search (GBFS)
def greedy_bfs(graph, start, goal, h):
    parent = dict()
    frontier = [start]  # Nodes to explore
    visited = []  # Explored nodes

    while frontier:
        # Sort frontier by heuristic value
        frontier.sort(key=lambda node: h[node])
        node = frontier.pop(0)  # Pick the node with the smallest h(n)

        if node == goal:
            return parent

        visited.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited and neighbor not in frontier:
                frontier.append(neighbor)
                parent[neighbor] = node

    return None  # If no path is found

# Traceback Function
def traceback(parent, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

# Example Execution
parent = greedy_bfs(graph, 'A', 'G', Heuristic)
if parent:
    print("Path from A to G:", traceback(parent, 'A', 'G'))  # Expected Output: ['A', 'C', 'G']
else:
    print("No path found from A to G")



---------------------------------------------





