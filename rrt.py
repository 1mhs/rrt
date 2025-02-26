import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacles, map_size, step_size=0.03, max_iter=2000):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacles = obstacles
        self.map_size = map_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.path = None
        self.fig, self.ax = plt.subplots()
        self.ani = None

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    
    def nearest_node(self, rnd_node):
        return min(self.nodes, key=lambda node: self.distance(node, rnd_node))

    def steer(self, from_node, to_node):
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * np.cos(theta)
        new_y = from_node.y + self.step_size * np.sin(theta)
        return Node(new_x, new_y)

    def is_collision(self, node):
        for (ox, oy, w, h) in self.obstacles:
            if ox <= node.x <= ox + w and oy <= node.y <= oy + h:
                return True
        return False

    def plan(self):
        for _ in range(self.max_iter):
            rand_x, rand_y = random.uniform(0, 1), random.uniform(0, 1)
            rand_node = Node(rand_x, rand_y)
            nearest = self.nearest_node(rand_node)
            new_node = self.steer(nearest, rand_node)
            
            if not self.is_collision(new_node):
                new_node.parent = nearest
                self.nodes.append(new_node)
                
                if self.distance(new_node, self.goal) < self.step_size:
                    self.goal.parent = new_node
                    self.nodes.append(self.goal)
                    self.path = self.extract_path()
                    return self.path
        return None
    
    def extract_path(self):
        path = []
        node = self.goal
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    
    def update_plot(self, frame):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        for node in self.nodes[:frame]:
            if node.parent:
                self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], '-k', linewidth=0.5)
        
        for (ox, oy, w, h) in self.obstacles:
            rect = plt.Rectangle((ox, oy), w, h, color='r', fill=False, linewidth=2)
            self.ax.add_patch(rect)
        
        self.ax.scatter(self.start.x, self.start.y, color='blue', s=100)
        self.ax.scatter(self.goal.x, self.goal.y, color='blue', s=100)
        
        if self.path and frame >= len(self.nodes) - 1:
            px, py = zip(*self.path)
            self.ax.plot(px, py, '-b', linewidth=2)

    def animate(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=len(self.nodes), interval=10, repeat=False)
        plt.show()

def is_overlapping(new_rect, existing_rects):
    nx, ny, nw, nh = new_rect
    for ox, oy, ow, oh in existing_rects:
        if not (nx + nw < ox or ox + ow < nx or ny + nh < oy or oy + oh < ny):
            return True 
    return False

def generate_random_obstacles(map_size, num_obstacles=5):
    obstacles = []
    max_attempts = 100
    for _ in range(num_obstacles):
        for _ in range(max_attempts): 
            ox = random.uniform(0.1, map_size[0] - 0.3)
            oy = random.uniform(0.1, map_size[1] - 0.3)
            w = random.uniform(0.1, 0.2)
            h = random.uniform(0.1, 0.2)
            new_obstacle = (ox, oy, w, h)
            if not is_overlapping(new_obstacle, obstacles):
                obstacles.append(new_obstacle)
                break
    return obstacles

def is_inside_obstacles(point, obstacles):
    x, y = point
    for (ox, oy, w, h) in obstacles:
        if ox <= x <= ox + w and oy <= y <= oy + h:
            return True
    return False

def generate_random_points(obstacles):
    while True:
        start = (random.uniform(0, 1), random.uniform(0, 1))
        goal = (random.uniform(0, 1), random.uniform(0, 1))
        if (not is_inside_obstacles(start, obstacles) and 
            not is_inside_obstacles(goal, obstacles) and 
            np.linalg.norm(np.array(start) - np.array(goal)) >= 0.2):
            return start, goal

map_size = (1.0, 1.0)
obstacles = generate_random_obstacles(map_size, num_obstacles=5)
start, goal = generate_random_points(obstacles)

rrt = RRT(start, goal, obstacles, map_size)
path = rrt.plan()
rrt.animate()
