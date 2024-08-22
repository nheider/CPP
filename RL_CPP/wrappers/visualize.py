import gymnasium as gym 
from gymnasium import Wrapper
import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

class VisualizeWrapper(Wrapper): 
    def __init__(self, env):
        super().__init__(env)

        self.fig = plt.figure(figsize=(8, 8))  # Larger figure size
        self.ax = self.fig.gca()
        plt.ion()  # Turn on interactive mode
        
    def reset(self):
        obs = self.env.reset()
        #self._update_display()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info  = self.env.step(action)
        self._update_display()
        return obs, reward, terminated, truncated, info
    
    def _update_display(self):
        field_polygon = Polygon(self.env.unwrapped.playground.polygon, facecolor='lightgreen', edgecolor='green', alpha=0.5)
        self.ax.add_patch(field_polygon)
        print("TEST")
        # Plot the path if provided
        if self.env.unwrapped.playground.path:
            path_x, path_y = zip(*self.env.unwrapped.playground.path)
            self.ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
            
        # Plot the path polygon if provided
        if self.env.unwrapped.playground.path_polygon:
            path_poly_patch = Polygon(self.env.unwrapped.playground.path_polygon, facecolor='red', edgecolor='red', alpha=0.3)
            self.ax.add_patch(path_poly_patch)
            
        # Set axis limits
        self.ax.set_xlim(self.env.unwrapped.playground.bounding_box[0], self.env.unwrapped.playground.bounding_box[1])
        self.ax.set_ylim(self.env.unwrapped.playground.bounding_box[2], self.env.unwrapped.playground.bounding_box[3])
        self.ax.set_aspect('equal', 'box')
            
        # Remove spines for a cleaner look
        for spine in self.ax.spines.values():
            spine.set_visible(False)
            
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

        
    

