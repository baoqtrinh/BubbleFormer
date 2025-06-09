import cv2
import numpy as np
import pickle
import os
import random
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.floorplan_Loader import Load_floorplan_Plan_Graph

# Define color map (BGR order for OpenCV)
color_map = np.array([
    [222, 241, 244],   # living room 0
    [159, 182, 234],   # bedroom 1
    [95, 122, 224],    # kitchen 2
    [123, 121, 95],    # bathroom 3
    [143, 204, 242],   # balcony 4
    [92, 112, 107],    # storage 5
    [100, 100, 100],   # exterior wall
    [25, 255, 255],    # front door
    [150, 150, 150],   # interior wall
    [255, 255, 255],   # external
], dtype=np.uint8)

def extract_data_from_rplan(rplan_image_path):
    # Load 4-channel PNG
    img = cv2.imread(rplan_image_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 2:
        raise ValueError(f"Invalid or missing image: {rplan_image_path}")

    instance_channel = img[:,:,0]
    room_type_channel = img[:,:,1]

    # For segmentation, you can use instance_channel > 0 as foreground
    segmentation = (instance_channel > 0).astype(np.uint8) * 255

    # 1. Extract boundary graph
    boundary_graph = extract_boundary_graph(segmentation)

    # 2. Extract wall graph
    wall_graph = extract_wall_graph(img, segmentation)

    # 3. Extract inter_graph (connections between walls)
    inter_graph = create_inter_graph(wall_graph)

    # 4. Extract door information from room_type_channel (ID 15 for front door)
    door_info = extract_door_info_from_channel(room_type_channel)

    # 5. Extract room information
    rooms_info = extract_rooms_info_from_channels(instance_channel, room_type_channel)

    # 6. Calculate room circles
    room_circles = calculate_room_circles(rooms_info)

    # 7. Determine room connections
    connects = determine_room_connections(segmentation, rooms_info)

    # 8. Generate BFS traversal iterations
    allG_iteration = generate_bfs_iterations(wall_graph)

    # 9. Create window mask (placeholder)
    new_window_mask = create_window_mask(img, segmentation)

    # 10. Generate blur masks
    inside_blur, outside_blur = generate_blur_masks(segmentation)

    return {
        'main': [wall_graph, boundary_graph, inter_graph, door_info, room_circles,
                 rooms_info, connects, allG_iteration, new_window_mask],
        'blur': [inside_blur, outside_blur]
    }


def extract_boundary_graph(img):
    """
    Extract the boundary graph using Channel 4 (index 3).
    Channel 4 distinguishes between exterior (0) and interior (255) areas.
    """
    # Get boundary information directly from Channel 4 (index 3)
    boundary_channel = img[:,:,3]
    
    # Create a mask for the boundary (value 255 for interior area)
    boundary_mask = (boundary_channel == 255).astype(np.uint8) * 255
    
    # Find contours of the boundary mask using CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a graph representation
    boundary_graph = {}
    node_id = 0
    
    # Sample points along the contour to create nodes
    for contour in contours:
        # Change step from 5 to 1 to use all points from CHAIN_APPROX_SIMPLE
        for i in range(0, len(contour), 1):  # MODIFIED LINE
            x, y = contour[i][0]
            boundary_graph[node_id] = {
                'pos': [y, x],  # Note: [height, width] format
                'connect': []
            }
            
            # Connect to previous node
            if node_id > 0:
                boundary_graph[node_id]['connect'].append(node_id-1)
                boundary_graph[node_id-1]['connect'].append(node_id)
            
            node_id += 1
    
    # Connect the last node to the first node to close the loop
    if node_id > 0 and len(boundary_graph) > 1: # Ensure there's more than one node to form a loop
        boundary_graph[0]['connect'].append(node_id-1)
        boundary_graph[node_id-1]['connect'].append(0)
    
    return boundary_graph

def extract_wall_graph(floor_plan, segmentation):
    # Detect all walls
    edges = cv2.Canny(segmentation, 50, 150)
    
    # Dilate to connect any broken lines
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
    
    # Create wall graph
    wall_graph = {}
    node_id = 0
    
    # Process each line segment
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Create nodes at endpoints
        wall_graph[node_id] = {
            'pos': [y1, x1],
            'connect': [node_id+1]
        }
        
        wall_graph[node_id+1] = {
            'pos': [y2, x2],
            'connect': [node_id]
        }
        
        node_id += 2
    
    # Merge nodes that are very close to each other
    wall_graph = merge_nearby_nodes(wall_graph)
    
    return wall_graph

def merge_nearby_nodes(graph, threshold=5):
    # Merge nodes that are within threshold distance
    merged_graph = {}
    node_map = {}  # Maps original node IDs to new node IDs
    
    new_id = 0
    for node_id, node_data in graph.items():
        # Check if this node is close to any existing node
        merged = False
        for new_node_id, new_node_data in merged_graph.items():
            dist = np.sqrt((node_data['pos'][0] - new_node_data['pos'][0])**2 + 
                          (node_data['pos'][1] - new_node_data['pos'][1])**2)
            if dist < threshold:
                # Merge this node
                node_map[node_id] = new_node_id
                merged = True
                break
        
        if not merged:
            # Create a new node
            merged_graph[new_id] = {
                'pos': node_data['pos'],
                'connect': []
            }
            node_map[node_id] = new_id
            new_id += 1
    
    # Update connections
    for old_id, node_data in graph.items():
        new_id = node_map[old_id]
        for conn in node_data['connect']:
            if conn in node_map and node_map[conn] not in merged_graph[new_id]['connect']:
                merged_graph[new_id]['connect'].append(node_map[conn])
    
    return merged_graph

def create_inter_graph(wall_graph):
    # Inter graph represents internal connections
    inter_graph = {}
    
    # Copy nodes from wall graph
    for node_id, node_data in wall_graph.items():
        inter_graph[node_id] = {
            'pos': node_data['pos'],
            'connect': []
        }
    
    # Find junctions (nodes with 3+ connections)
    junctions = [node_id for node_id, node_data in wall_graph.items() 
                if len(node_data['connect']) >= 3]
    
    # Add connections between junctions
    for i, j1 in enumerate(junctions):
        for j2 in junctions[i+1:]:
            # Check if there's a path between junctions
            if path_exists(wall_graph, j1, j2):
                inter_graph[j1]['connect'].append(j2)
                inter_graph[j2]['connect'].append(j1)
    
    return inter_graph

def path_exists(graph, start, end, max_depth=5):
    # Simple BFS to check if there's a path within max_depth
    visited = set([start])
    queue = [(start, 0)]
    
    while queue:
        node, depth = queue.pop(0)
        
        if node == end:
            return True
        
        if depth >= max_depth:
            continue
        
        for neighbor in graph[node]['connect']:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return False

def extract_rooms_info(orig_image, segmentation):
    # Identify individual rooms
    ret, labels = cv2.connectedComponents(segmentation)
    
    rooms_info = []
    for label in range(1, ret):  # Skip background (0)
        # Create mask for this room
        room_mask = (labels == label).astype(np.uint8) * 255
        
        # Find centroid
        M = cv2.moments(room_mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue
        
        # Count pixels to determine area
        pixels = np.sum(room_mask > 0)
        
        # Determine room category using original image data
        category = determine_room_category(orig_image, labels, label)
        
        rooms_info.append({
            'pos': [cy, cx],  # [height, width] format
            'pixels': pixels,
            'category': category
        })
    
    return rooms_info

def determine_room_category(orig_image, segmentation, label):
    """
    Determine room category using RPLAN's encoding.
    RPLAN uses channel 1 (index 1) for room types.
    
    Room type mapping from RPLAN to BubbleFormer:
    0: Living room     -> 0
    1: Master room     -> 2 (Bedroom)
    2: Kitchen         -> 1
    3: Bathroom        -> 3
    4: Dining room     -> 4
    5: Child room      -> 2 (Bedroom)
    6: Study room      -> 5
    7: Second room     -> 2 (Bedroom)
    8: Guest room      -> 2 (Bedroom)
    9: Balcony         -> 6
    10: Entrance       -> 7
    11: Storage        -> 8
    """
    # Create mask for this room
    room_mask = (segmentation == label)
    
    # Get the room type from channel 1 of the original image
    room_type_channel = orig_image[:, :, 1]
    room_type_values = room_type_channel[room_mask]
    
    if room_type_values.size == 0:
        return 0  # Default to living room
    
    # Get the most common room type value in this room
    room_type = np.bincount(room_type_values.flatten()).argmax()
    
    # Map RPLAN categories to BubbleFormer categories
    rplan_to_bubble = {
        0: 0,  # Living room
        1: 2,  # Master room -> Bedroom
        2: 1,  # Kitchen
        3: 3,  # Bathroom
        4: 4,  # Dining room
        5: 2,  # Child room -> Bedroom
        6: 5,  # Study room
        7: 2,  # Second room -> Bedroom
        8: 2,  # Guest room -> Bedroom
        9: 6,  # Balcony
        10: 7, # Entrance
        11: 8  # Storage
    }
    
    return rplan_to_bubble.get(room_type, 0)  # Default to living room if not found

def determine_room_connections(segmentation, rooms_info):
    connects = []
    
    # Dilate each room and check for overlaps
    room_masks = []
    for i, room in enumerate(rooms_info):
        # Create mask for this room
        mask = np.zeros_like(segmentation)
        cv2.circle(mask, (room['pos'][1], room['pos'][0]), 
                  10, 255, -1)  # Small circle at centroid
        room_masks.append(mask)
    
    # Dilate masks and check for overlaps
    kernel = np.ones((15,15), np.uint8)
    for i, mask1 in enumerate(room_masks):
        dilated = cv2.dilate(mask1, kernel, iterations=1)
        for j, mask2 in enumerate(room_masks[i+1:], i+1):
            # If dilated masks overlap, rooms are connected
            if np.any(np.logical_and(dilated, mask2)):
                connects.append([i, j])
    
    return connects

def generate_bfs_iterations(graph):
    # Start from a random node
    start_node = random.choice(list(graph.keys()))
    
    # BFS traversal
    visited = set([start_node])
    queue = [start_node]
    iterations = {'iteration': []}
    
    while queue:
        current_level = []
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.pop(0)
            current_level.append(node)
            
            for neighbor in graph[node]['connect']:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if current_level:
            iterations['iteration'].append(current_level)
    
    return iterations

def generate_blur_masks(segmentation):
    # Create inside blur mask
    inside_mask = (segmentation > 0).astype(np.uint8) * 255
    inside_blur = cv2.GaussianBlur(inside_mask, (31, 31), 0)
    
    # Create outside blur mask
    outside_mask = (segmentation == 0).astype(np.uint8) * 255
    outside_blur = cv2.GaussianBlur(outside_mask, (31, 31), 0)
    
    return inside_blur, outside_blur

def save_pickle_files(output_data, output_dir, filename, blur_output_dir):
    # Save main pickle file
    main_path = os.path.join(output_dir, filename + '.pkl')
    with open(main_path, 'wb') as f:
        pickle.dump(output_data['main'], f)
    
    # Save blur pickle file
    blur_path = os.path.join(blur_output_dir, filename + '_blur.pkl')
    with open(blur_path, 'wb') as f:
        pickle.dump(output_data['blur'], f)
    
    print(f"Saved pickle files to {main_path} and {blur_path}")

def process_rplan_dataset(rplan_dir, output_dir, blur_output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(blur_output_dir, exist_ok=True)

    for filename in os.listdir(rplan_dir):
        if filename.endswith('.png') and not '_' in filename:
            image_path = os.path.join(rplan_dir, filename)
            output_data = extract_data_from_rplan(image_path)
            base_name = filename.replace('.png', '')
            save_pickle_files(output_data, output_dir, base_name, blur_output_dir)

def extract_door_info(door_image):
    # Extract door information from the door image
    # In RPLAN, doors are marked with a specific color (ID 15)
    door_info = []
    
    # Convert to grayscale if needed
    if len(door_image.shape) > 2:
        # Look for front door color (ID 15 in RPLAN)
        door_mask = np.all(door_image == color_map[7], axis=2)
    else:
        door_mask = (door_image == 15)
    
    # Find door contours
    contours, _ = cv2.findContours(door_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get door endpoints
        if len(contour) >= 2:
            # Get two points that are furthest apart
            x1, y1 = contour[0][0]
            x2, y2 = contour[-1][0]
            
            door_info.append({
                'pos1': [y1, x1],  # [height, width] format
                'pos2': [y2, x2]
            })
    
    return door_info

def detect_doors(floor_plan, segmentation):
    # Fallback method if no door image is provided
    # This is a simplified method that may not be accurate
    edges = cv2.Canny(segmentation, 50, 150)
    
    # Find small line segments that could be doors
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=10, maxLineGap=5)
    
    door_info = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Only consider lines of a certain length (doors are typically smaller than walls)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if 10 < length < 30:  # Adjust these thresholds based on your dataset
                door_info.append({
                    'pos1': [y1, x1],
                    'pos2': [y2, x2]
                })
    
    return door_info

def calculate_room_circles(rooms_info):
    # Create room circles based on room information
    room_circles = []
    
    for room in rooms_info:
        # Calculate radius based on room area
        radius = round(np.sqrt(room['pixels'] / np.pi))
        
        room_circles.append({
            'pos': room['pos'],
            'radius': radius,
            'category': room['category']
        })
    
    return room_circles

def create_window_mask(floor_plan, segmentation):
    # In RPLAN, windows might not be explicitly labeled
    # We'll create a placeholder window mask
    return np.zeros_like(segmentation)

def extract_room_adjacency_graph(instance_channel, kernel_size=10):
    """
    Given the instance channel (channel 0) of the 4-channel PNG,
    returns a dictionary mapping each room instance ID to a set of adjacent instance IDs.
    """
    room_ids = np.unique(instance_channel)
    room_ids = room_ids[room_ids != 0]  # Exclude background

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    masks = {}
    for rid in room_ids:
        mask = (instance_channel == rid).astype(np.uint8)
        masks[rid] = cv2.dilate(mask, kernel, iterations=1)

    adjacency = {rid: set() for rid in room_ids}
    for i, rid1 in enumerate(room_ids):
        for rid2 in room_ids[i+1:]:
            if np.any(masks[rid1] & masks[rid2]):
                adjacency[rid1].add(rid2)
                adjacency[rid2].add(rid1)
    return adjacency

def extract_door_info_from_channel(room_type_channel):
    # ID 15 = front door, ID 17 = interior door (from your table)
    door_info = []
    for door_id in [15, 17]:
        mask = (room_type_channel == door_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 2:
                x1, y1 = contour[0][0]
                x2, y2 = contour[-1][0]
                door_info.append({'pos1': [y1, x1], 'pos2': [y2, x2], 'type': int(door_id)})
    return door_info

def extract_rooms_info_from_channels(instance_channel, room_type_channel):
    rooms_info = []
    room_ids = np.unique(instance_channel)
    room_ids = room_ids[room_ids != 0]
    for rid in room_ids:
        mask = (instance_channel == rid)
        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            continue
        cy, cx = np.mean(coords, axis=0)
        pixels = np.sum(mask)
        room_type_vals = room_type_channel[mask]
        if room_type_vals.size == 0:
            category = 0
        else:
            category = np.bincount(room_type_vals.flatten()).argmax()
        rooms_info.append({'pos': [int(cy), int(cx)], 'pixels': int(pixels), 'category': int(category)})
    return rooms_info

if __name__ == "__main__":
    # Set paths
    rplan_dir = "D:/Desktop/RPLAN/dataset/floorplan_dataset"
    output_dir = "D:/Desktop/RPLAN/dataset/RPLAN_data_compact"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    blur_output_dir = output_dir.replace("RPLAN_data_compact", "RPLAN_data_compact_blur")
    os.makedirs(blur_output_dir, exist_ok=True)
    
    # Process dataset
    process_rplan_dataset(rplan_dir, output_dir, blur_output_dir)
    
    # Test loading
    print("Testing sample file loading...")
    sample_files = os.listdir(output_dir)
    if sample_files:
        sample_path = os.path.join(output_dir, sample_files[0])
        try:
            fp_loader = Load_floorplan_Plan_Graph(sample_path)
            outputs = fp_loader.get_output()
            print(f"Successfully loaded sample data with {outputs['labels'].shape[0]} rooms")
        except Exception as e:
            print(f"Error loading sample: {e}")
    else:
        print("No sample files generated")