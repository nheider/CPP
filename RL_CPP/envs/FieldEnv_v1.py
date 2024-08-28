import numpy as np 
import gymnasium as gym 
from gymnasium import spaces
from scipy.spatial import ConvexHull
import math 
import skgeom as sg
from skgeom import boolean_set
from skgeom.draw import draw
import cv2

class FieldEnv(gym.Env):
    def __init__(self, max_size=1000, num_points=8, vehicle_width=10, sub_steps=10, num_lidar_rays = 20):
        super(FieldEnv, self).__init__()

        self.max_size = max_size
        self.num_points = num_points
        self.vehicle_width = vehicle_width 
        self.sub_steps = sub_steps         
        self.num_lidar_rays = num_lidar_rays
        self.path = []
        self.sg_path_poly = None 
        self.sg_cover_poly = None 
        self.sg_field_poly = None 

        self.left_edge = []
        self.right_edge = []
        self.path_polygon = []

        # Normalized action space: segment distance and curvature 
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float64
        )

        # Normalized observation space: num_rays with normalized distances 
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_lidar_rays,),
            dtype=np.float64
        )

    def create_field(self): 
        points = np.random.randint(0, self.max_size, size=(self.num_points, 2))
        hull = ConvexHull(points)
        field_polygon = points[hull.vertices].tolist()
        field_polygon.append(field_polygon[0]) # Ensures Polygon is closed
        self.field_polygon = field_polygon
        self.sg_field_poly = sg.Polygon(field_polygon)

    def random_point_on_polygon_perimeter(self): # To determin the start point and heading of the agent 
        perimeter = 0
        for i in range(len(self.field_polygon)):
            x1, y1 = self.field_polygon[i]
            x2, y2 = self.field_polygon[(i + 1) % len(self.field_polygon)]
            perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        random_distance = np.random.uniform(0, perimeter)
        cumulative_distance = 0

        for i in range(len(self.field_polygon)-1):
            x1, y1 = self.field_polygon[i]
            x2, y2 = self.field_polygon[(i + 1) % len(self.field_polygon)]
            segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
            if cumulative_distance + segment_length >= random_distance:
                segment_fraction = (random_distance - cumulative_distance) / segment_length
                px = x1 + segment_fraction * (x2 - x1)
                py = y1 + segment_fraction * (y2 - y1)

                # Calculate the heading perpendicular to the edge
                edge_angle_radians = math.atan2(y2 - y1, x2 - x1)
                perpendicular_heading_radians = edge_angle_radians + math.radians(90)
                perpendicular_heading_degrees = math.degrees(perpendicular_heading_radians)

                start_point = [px, py]
                heading = perpendicular_heading_degrees % 360
                
                return start_point, heading
                
            cumulative_distance += segment_length
    
    def next_point_in_path(self, segment_len, segment_angle_degrees, start = False):
        x1, y1 = self.path[-1]
        total_angle_radians = math.radians(self.heading + segment_angle_degrees)

        # Calculate the endpoint
        x2 = x1 + segment_len * math.cos(total_angle_radians)
        y2 = y1 + segment_len * math.sin(total_angle_radians)

        # Calculate offsets for width
        offset_top_dx, offset_top_dy = self.calculate_offsets(total_angle_radians, self.vehicle_width)
        offset_bottom_dx, offset_bottom_dy = self.calculate_offsets(total_angle_radians, -self.vehicle_width)

        # Top and bottom endpoints
        x2_top = x2 + offset_top_dx
        y2_top = y2 + offset_top_dy
        x2_bot = x2 + offset_bottom_dx
        y2_bot = y2 + offset_bottom_dy

        # Edge case width for the first ever segment, that gets placed randomly # To do: rewrite function so this is not needed 
        if start: 
            initial_angle_radians = math.radians(self.heading)
            initial_offset_top_dx, initial_offset_top_dy = self.calculate_offsets(initial_angle_radians, self.vehicle_width)
            initial_offset_bottom_dx, initial_offset_bottom_dy = self.calculate_offsets(initial_angle_radians, -self.vehicle_width)
            
            x1_top = x1 + initial_offset_top_dx
            y1_top = y1 + initial_offset_top_dy
            x1_bot = x1 + initial_offset_bottom_dx
            y1_bot = y1 + initial_offset_bottom_dy

        else: 
            x1_top = y1_top = x1_bot = y1_bot = None 

        return [x2, y2], [x2_top, y2_top], [x2_bot, y2_bot], [x1_top, y1_top], [x1_bot, y1_bot]

    def calculate_offsets(self, angle_radians, vehicle_width): 
        offset_angle = angle_radians - math.radians(90)
        offset_dx = 0.5 * vehicle_width * math.cos(offset_angle)
        offset_dy = 0.5 * vehicle_width * math.sin(offset_angle)
        return offset_dx, offset_dy

    def steering_to_curve(self, distance, steering_angle):
        if steering_angle < -90 or steering_angle > 90:
            raise ValueError("only works with steering angles up to +/- 90 degrees")

        segment_len = distance / self.sub_steps
        segment_angle = steering_angle / self.sub_steps

        # new_path and new_polygon get used to update the cell visit count
        new_polygon = []
        new_left_edge = []
        new_right_edge = []

        for i in range(self.sub_steps):
            if i == 0:
                mid, top, bot, initial_top, initial_bot = self.next_point_in_path(segment_len, segment_angle, start=True)
                self.left_edge.append(initial_top)
                self.right_edge.append(initial_bot)

                new_left_edge.append(initial_top)
                new_right_edge.append(initial_bot)

            else:
                mid, top, bot, _, _ = self.next_point_in_path(segment_len, segment_angle)

            self.heading = (self.heading + segment_angle) % 360

            self.path.extend([mid])
            self.left_edge.extend([top])
            self.right_edge.extend([bot])

            new_left_edge.extend([top])
            new_right_edge.extend([bot])

        # Create the polygons: cover_polygon is the newly covered area, path polygon is the overall covered area 
        self.cover_polygon = new_left_edge + list(reversed(new_right_edge))
        if self.cover_polygon[0] != self.cover_polygon[-1]:
            self.cover_polygon.append(self.cover_polygon[0])
        self.sg_cover_poly = sg.Polygon(self.cover_polygon)

        if len(self.path_polygon == 0): 
            self.path_polygon = self.left_edge + list(reversed(self.right_edge))
            if self.path_polygon[0] != self.path_polygon[-1]:
                self.path_polygon.append(self.path_polygon[0])
            sg_path_poly = sg.Polygon(self.path_polygon)
        else: 
            self.sg_path_poly = boolean_set.join(self.sg_path_poly, self.sg_cover_poly)

    def extend_path(self, distance, steering_angle):
        self.steering_to_curve(distance=distance, steering_angle=steering_angle)
    
    def calc_intersect_area(self, field_poly, path_poly):
        pg1 = sg.Polygon(field_poly)
        pg2 = sg.Polygon(path_poly)
        intersect_area = boolean_set.intersect(pg1, pg2).area()
        return intersect_area
    
    def check_if_completed(self):
        self.completed = (self.field_area == self.coverage_area)

    def calc_path_intersection(): 
        # to do 
        return

    def get_lidar_distances(self):
        current_position = np.array(self.path[-1])
        lidar_distances = []

        for i in range(self.num_lidar_rays):
            angle = (self.heading + i * (360 / self.num_lidar_rays)) % 360
            angle_rad = np.radians(angle)
            
            # Calculate the end point of the ray
            ray_end = current_position + 1000 * np.array([np.cos(angle_rad), np.sin(angle_rad)])

            # Check for intersection with polygon edges
            min_distance = 1000  # Initialize with max distance

            for j in range(len(self.field_polygon)):
                p1 = np.array(self.field_polygon[j])
                p2 = np.array(self.field_polygon[(j + 1) % len(self.field_polygon)])
                
                # Calculate intersection using line segments
                intersection_point = self._line_intersection(current_position, ray_end, p1, p2)
                
                if intersection_point is not None:
                    distance = np.linalg.norm(intersection_point - current_position)
                    if distance < min_distance:
                        min_distance = distance
            
            lidar_distances.append(min_distance)

        return np.array(lidar_distances, dtype=np.float32)

    def _line_intersection(self, p1, p2, q1, q2):
        # Calculate the intersection of two line segments p1-p2 and q1-q2
        r = p2 - p1
        s = q2 - q1
        r_cross_s = np.cross(r, s)
        
        if r_cross_s == 0:
            return None  # Lines are parallel or collinear

        t = np.cross((q1 - p1), s) / r_cross_s
        u = np.cross((q1 - p1), r) / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point = p1 + t * r
            return intersection_point

        return None     

    def check_if_inside(self):
        point = self.path[-1]
        polygon = self.field_polygon

        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside



    def reset_environment(self):
        field = self.create_field()
        point, heading = self.random_point_on_polygon_perimeter()
        self.heading = heading 
        self.path = [point]

        self.sg_path_poly = None 
        self.sg_cover_poly = None 
        self.path_polygon = []



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_environment()

        obs = self.get_lidar_distances()
        info = {}  # To do: add some info 

        return obs, info

    def step(self, action):
        terminated = False
        truncated = False
        
        steering_angle = action[0] * 60 # Normalized Steering 
        
        distance = action[1] * 10

        self.extend_path(steering_angle=steering_angle, distance=distance)
    
        obs = self.get_lidar_distances()
                
        alpha = 1000  # Reward for new area covered
        beta = 0   # Penalty for overlap area !!! Overlap is not yet implemented 
        delta = 1000 # Large reward for completing the task
        psi = 100 # Large penalty for leaving the field
        norm =  abs(float(self.sg_field_poly.area()))# larger fields should get more reward by default 
        
        reward = (alpha * (abs(float(self.sg_cover_poly.area()))/norm)*10000)/ (distance*0.8) #- (beta * self.overlap_area) 

        if self.completed:
            reward += delta

        if self.check_if_inside() is not True: 
            reward -= psi
            terminated = True

        self.reward = reward
        
        return obs, reward, terminated, truncated, {} 
    
    def render(self):
        # Create a larger blank image for super-sampling
        scale_factor = 4
        img_size = (1000 * scale_factor, 800 * scale_factor)
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255  # White background

        poly = sg.Polygon(self.field_polygon)
        bounding_box = poly.bbox()
        # Define scale factors to map field coordinates to image coordinates
        margin = 0.1  # 10% margin
        plot_bbox = [
            bounding_box[0] - (bounding_box[1] - bounding_box[0]) * margin,
            bounding_box[1] + (bounding_box[1] - bounding_box[0]) * margin,
            bounding_box[2] - (bounding_box[3] - bounding_box[2]) * margin,
            bounding_box[3] + (bounding_box[3] - bounding_box[2]) * margin
        ]
        scale_x = img_size[0] / (plot_bbox[1] - plot_bbox[0])
        scale_y = img_size[1] / (plot_bbox[3] - plot_bbox[2])
        
        def map_to_img(point):
            x = int((point[0] - plot_bbox[0]) * scale_x)
            y = img_size[1] - int((point[1] - plot_bbox[2]) * scale_y)  # Invert y-axis
            return (x, y)

        # Draw the field polygon
        field_poly = np.array([map_to_img(p) for p in self.field_polygon], np.int32)
        cv2.fillPoly(img, [field_poly], color=(230, 250, 230))  # Light green
        cv2.polylines(img, [field_poly], isClosed=True, color=(0, 128, 0), thickness=3*scale_factor)  # Dark green border

        # Draw the path
        path_points = np.array([map_to_img(p) for p in self.path], np.int32)
        cv2.polylines(img, [path_points], isClosed=False, color=(220, 20, 60), thickness=3*scale_factor)  # Crimson path

        # Draw the path polygon
        path_poly = np.array([map_to_img(p) for p in self.path_polygon], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [path_poly], color=(220, 20, 60))  # Crimson fill
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)  # Blend for transparency

        # Draw gridlines
        grid_color = (152, 184, 183)
        for x in np.linspace(plot_bbox[0], plot_bbox[1], 11):
            start = map_to_img((x, plot_bbox[2]))
            end = map_to_img((x, plot_bbox[3]))
            cv2.line(img, start, end, grid_color, 1*scale_factor)
        for y in np.linspace(plot_bbox[2], plot_bbox[3], 11):
            start = map_to_img((plot_bbox[0], y))
            end = map_to_img((plot_bbox[1], y))
            cv2.line(img, start, end, grid_color, 1*scale_factor)

        # Resize the image to the original size
        img = cv2.resize(img, (1000, 800), interpolation=cv2.INTER_AREA)

        # Display the image
        #cv2.imshow('Plot', img)
        cv2.waitKey(1000)  # Wait for a key event (1ms delay)
    
        # Store the image if needed
        #self.img = img

        # Optionally save the image
        print("Outside:", self.outside)
        print("Rewards:", self.reward)
        cv2.imwrite('field_map_with_path.png', img)


    def close(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)