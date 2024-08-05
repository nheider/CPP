import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import math 
import pyclipper
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

class Env: # Contains all the logic of the CPP Environment  
    def __init__(self, max_size=1000, num_points=8, vehicle_width=10, sub_steps=10):

        # Field Variables 
        self.max_size = max_size           # Largest possible x and y-coords of the field
        self.num_points = num_points       # Number of random points for the convex hull field generator, higher number = higher mean number of fiel vertices 
        self.bounding_box = []             # Bounding Box of the generated field 
        self.cover_polygon = []            # A polygon that records the intersection between the Vehicle Path and the Field 
        self.path_matrix = None            # A Matrix that is 1 where the vehicle covered the field and 0 where it didnt pass. 

        self.polygon = self.create_field(max_size, num_points)
        self.matrix = self.create_field_matrix(self.polygon)
        self.start_point, self.heading = self.random_point_on_polygon_perimeter(self.polygon)

        self.visit_matrix = np.zeros_like(self.matrix, dtype=np.int32) # Records the number of times each cell gets visited, used for visualization 

        # Path Variables 
        self.width = vehicle_width      # The width of the vehicles covered path 
        self.sub_steps = sub_steps      # Number of picewise linear spline points of the path segments, higher := better resolution 
        self.path = [self.start_point]  
        self.left_edge = []
        self.right_edge = []

    def create_field(self, num_points=8):
        points = np.random.randint(0, self.max_size, size=(self.num_points, 2))
        hull = ConvexHull(points)
        field_points = points[hull.vertices].tolist()
        field_points.append(field_points[0]) # Ensures Polygon is closed

        x_coordinates, y_coordinates = zip(*field_points)

        self.bounding_box = [min(x_coordinates), max(x_coordinates), min(y_coordinates), max(y_coordinates)]
        self.polygon = field_points

    def create_field_matrix(self):
        field_vertices = np.array([self.polygon], dtype=np.int32)
        matrix = np.full((self.max_size, self.max_size), 99, dtype=np.uint8) # Generate a Matrix the size of the biggest possible field 
        field_matrix = cv2.fillPoly(matrix, field_vertices, 0)               # Fill cells that are inside the field polygon with 0 
        self.matrix = field_matrix                                           # Matrix with 99 := Outside, 0 := Field 

    def size(self):
        mask = self.matrix != 99
        count_of_elements = np.sum(mask)
        return count_of_elements

    def cover_polygon(self):
        pc = pyclipper.Pyclipper()
        pc.AddPath(self.path_polygon, pyclipper.PT_SUBJECT, True)
        pc.AddPath(self.polygon, pyclipper.PT_CLIP, True)
        intersection = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        self.cover_polygon = intersection       # A Polygon that represents the intersection of the vehicle path and the field 

    def update_matrix(self):
        self.matrix[(self.matrix != 99) & (self.path_matrix == 1)] = 1

    def update_visit_counts(self):
        self.visit_matrix[(self.matrix != 99) & (self.path_matrix == 1)] += 1  

    def get_stats(self):
        field_mask = self.matrix != 99
        visits = self.visit_matrix[field_mask]
        return {
            'max_visits': np.max(visits),
            'min_visits': np.min(visits),
            'avg_visits': np.mean(visits),
            'total_visits': np.sum(visits),
            'total_unique_visits': np.count_nonzero(visits),
            'unvisited_cells': np.sum(visits == 0)
        }
   
    def random_point_on_polygon_perimeter(self): # To determin the start point and heading of the agent 
        perimeter = 0
        for i in range(len(self.polygon)):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % len(self.polygon)]
            perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
 
        random_distance = np.random.uniform(0, perimeter)
        cumulative_distance = 0

        for i in range(len(self.polygon)-1):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % len(self.polygon)]
            segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
            if cumulative_distance + segment_length >= random_distance:
                segment_fraction = (random_distance - cumulative_distance) / segment_length
                px = x1 + segment_fraction * (x2 - x1)
                py = y1 + segment_fraction * (y2 - y1)

                # Calculate the heading perpendicular to the edge
                edge_angle_radians = math.atan2(y2 - y1, x2 - x1)
                perpendicular_heading_radians = edge_angle_radians + math.radians(90)
                perpendicular_heading_degrees = math.degrees(perpendicular_heading_radians)

                return (px, py), perpendicular_heading_degrees % 360
            cumulative_distance += segment_length


    def next_point_in_path(self, path, spline_len, heading_degrees, spline_angle_degrees, width, start = False):
        x1, y1 = path[-1]
        total_angle_radians = math.radians(heading_degrees + spline_angle_degrees)

        # Calculate the endpoint
        x2 = x1 + spline_len * math.cos(total_angle_radians)
        y2 = y1 + spline_len * math.sin(total_angle_radians)

        # Calculate offsets for width
        offset_top_dx, offset_top_dy = self.calculate_offsets(total_angle_radians, width)
        offset_bottom_dx, offset_bottom_dy = self.calculate_offsets(total_angle_radians, -width)

        # Top and bottom endpoints
        x2_top = x2 + offset_top_dx
        y2_top = y2 + offset_top_dy
        x2_bot = x2 + offset_bottom_dx
        y2_bot = y2 + offset_bottom_dy

        # Edge case width for the first ever segment, that gets placed randomly # To do: rewrite function so this is not needed 
        if start: 
            initial_angle_radians = math.radians(heading_degrees)
            initial_offset_top_dx, initial_offset_top_dy = self.calculate_offsets(initial_angle_radians, width)
            initial_offset_bottom_dx, initial_offset_bottom_dy = self.calculate_offsets(initial_angle_radians, -width)
            
            x1_top = x1 + initial_offset_top_dx
            y1_top = y1 + initial_offset_top_dy
            x1_bot = x1 + initial_offset_bottom_dx
            y1_bot = y1 + initial_offset_bottom_dy

        else: 
            x1_top = y1_top = x1_bot = y1_bot = None 

        return [x2, y2], [x2_top, y2_top], [x2_bot, y2_bot], [x1_top, y1_top], [x1_bot, y1_bot]

    def calculate_offsets(self, angle_radians, width):
        offset_angle = angle_radians - math.radians(90)
        offset_dx = 0.5 * width * math.cos(offset_angle)
        offset_dy = 0.5 * width * math.sin(offset_angle)
        return offset_dx, offset_dy

    def steering_to_curve(self, path, distance, steering_angle, heading, width, sub_steps):
        if steering_angle < -90 or steering_angle > 90:
            raise ValueError("only works with steering angles up to +/- 90 degrees")


        spline_len = distance / sub_steps
        spline_angle = steering_angle / sub_steps

        new_path = []
        new_left_edge = []
        new_right_edge = []

        for i in range(sub_steps):
            if i == 0:
                mid, top, bot, initial_top, initial_bot = self.next_point_in_path(path, spline_len, heading, spline_angle, width, start=True)
                new_left_edge.append(initial_top)
                new_right_edge.append(initial_bot)
            else:
                mid, top, bot, _, _ = self.next_point_in_path(new_path, spline_len, heading, spline_angle, width)
            heading += spline_angle

            new_path.append(mid)
            new_left_edge.append(top)
            new_right_edge.append(bot)

        path.extend(new_path)
        self.left_edge.extend(new_left_edge)
        self.right_edge.extend(new_right_edge)

        # Create the path polygon
        self.path_polygon = self.left_edge + list(reversed(self.right_edge))
        if self.path_polygon[0] != self.path_polygon[-1]:
            self.path_polygon.append(self.path_polygon[0])

        self.heading = heading
        self.path = path

        return path, new_path, heading, self.path_polygon

    def extend_path(self, path, distance, steering_angle, heading):
        self.path, self.new_path, self.heading, self.path_polygon = self.steering_to_curve(path, distance, steering_angle, heading, self.width, self.sub_steps)
    def position(self):
        self.position = self.path[-1]

    def visualize(self, path=None, path_poly=None, show_visits=False):
        fig, ax = plt.subplots(figsize=(10, 10))
    
        if show_visits:
            # Create a masked array for visit counts
            masked_visits = np.ma.masked_where(self.matrix == 99, self.visit_matrix)
            im = ax.imshow(masked_visits, cmap='viridis', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='Visit Count')
      
        # Plot the field
        field_polygon = Polygon(self.polygon, facecolor='lightgreen', edgecolor='green', alpha=0.5)
        ax.add_patch(field_polygon)
    
     # Plot the path if provided
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
    
        # Plot the path polygon if provided
        if path_poly:
            path_poly_patch = Polygon(path_poly, facecolor='red', edgecolor='red', alpha=0.3)
            ax.add_patch(path_poly_patch)
    
        # Set axis limits
        ax.set_xlim(self.bounding_box[0], self.bounding_box[1])
        ax.set_ylim(self.bounding_box[2], self.bounding_box[3])
        ax.set_aspect('equal', 'box')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        #ax.set_title('Field Visualization')
    
        # Add legend
        ax.legend()
    
        # Show the plot
        plt.show()


class Gym:
    def __init__(self):
        self.Env = None

    def initialize(self):
        env = Env(max_size=1000, num_points=8,vehicle_width=10, sub_steps=10)

        inital_path, initial_heading = self.env.random_point_on_polygon_perimeter(self.env.polygon)

    def step(self, visualize=False):
        observation = self.Env.matrix
        distance = np.random.randint(10, 100)
        steering_angle = np.random.randint(-60, 60)
        self.Env.extend_path(path=self.path_obj.path, distance=distance, steering_angle=steering_angle, heading=self.path_obj.heading)
        self.Env.update_matrix(self.field.polygon, self.path_obj.path_polygon)

        if visualize: 
            self.Env.visualize(path=self.path_obj.path, path_poly=self.path_obj.path_polygon, show_visits=True)

    def eval(self):
        values, counts = np.unique(self.Env.matrix, return_counts=True)
        obs = zip(values, counts)
        lst = list(zip(*obs))
        print(lst)
        
# Usage
gym = Gym()
gym.initialize()

for i in range(30): 
    gym.step(visualize=True)
gym.eval()



