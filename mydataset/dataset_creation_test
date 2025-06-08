import cv2
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

# Load the 4-channel image
img = cv2.imread(r'0.png', cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError("Image not found or path is incorrect.")

ch_instance = img[:,:,0]  # channel 1 (Python index 0) is actually the instance channel
ch_roomtype = img[:,:,1]  # channel 2 (Python index 1) is the room type

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

# Create a blank color image
color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Assign color to each room instance (0 is background/other)
room_ids = np.unique(ch_instance)
room_ids = room_ids[room_ids != 0]

for rid in room_ids:
    mask = (ch_instance == rid)
    room_type_vals = ch_roomtype[mask]
    if room_type_vals.size == 0:
        continue
    room_type_val = np.bincount(room_type_vals.flatten()).argmax()
    if room_type_val < len(color_map):
        color_img[mask] = color_map[room_type_val]

# Visualize exterior wall (id 16) and front door (id 15) from channel 2
exterior_wall_mask = (ch_roomtype == 14)
front_door_mask = (ch_roomtype == 15)
wall_fd_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
# Use color from color_map for exterior wall and front door
wall_fd_img[exterior_wall_mask] = color_map[6]      # exterior wall color from color_map
wall_fd_img[front_door_mask] = color_map[7]         # front door color from color_map

# Save with input image name as prefix
img_name = os.path.splitext(os.path.basename('0.png'))[0]

cv2.imwrite(f'{img_name}_room_instances_colored.png', color_img)
print(f"Saved colored room instance image as '{img_name}_room_instances_colored.png'.")

cv2.imwrite(f'{img_name}_frontdoor_exwall.png', wall_fd_img)
print(f"Saved front door and exterior wall mask as '{img_name}_frontdoor_exwall.png'.")

adjacency = {rid: set() for rid in room_ids}
kernel = np.ones((10, 10), np.uint8)

masks = {}
for rid in room_ids:
    mask = (ch_instance == rid).astype(np.uint8)
    masks[rid] = cv2.dilate(mask, kernel, iterations=1)

for i, rid1 in enumerate(room_ids):
    for rid2 in room_ids[i+1:]:
        if np.any(masks[rid1] & masks[rid2]):
            adjacency[rid1].add(rid2)
            adjacency[rid2].add(rid1)

print("Room adjacency graph (instance IDs):")
for rid, neighbors in adjacency.items():
    print(f"Room {rid} adjacent to: {list(neighbors)}")

room_type_names = [
    "Living room", "Master room", "Kitchen", "Bathroom", "Dining room", "Child room",
    "Study room", "Second room", "Guest room", "Balcony", "Entrance", "Storage"
]

node_pos = {}
node_color = []
node_labels = {}
for rid in room_ids:
    mask = (ch_instance == rid)
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        centroid = (0, 0)
    else:
        centroid = coords.mean(axis=0)[::-1]  # (x, y) order for plotting
        centroid = (centroid[0], img.shape[0] - centroid[1])  # invert y for display
    node_pos[rid] = centroid

    # Use the color of the room type for this node, convert BGR to RGB for matplotlib
    room_type_vals = ch_roomtype[mask]
    if room_type_vals.size == 0:
        color = [0.5, 0.5, 0.5]
        label = "Unknown"
    else:
        room_type_val = np.bincount(room_type_vals.flatten()).argmax()
        bgr = color_map[room_type_val]
        rgb = bgr[::-1]  # BGR to RGB
        color = rgb / 255.0
        label = room_type_names[room_type_val] if room_type_val < len(room_type_names) else str(room_type_val)
    node_color.append(color)
    node_labels[rid] = label

# Build and draw the graph  
G = nx.Graph()
for rid in room_ids:
    G.add_node(rid)
for rid, neighbors in adjacency.items():
    for nbr in neighbors:
        G.add_edge(rid, nbr)

plt.figure(figsize=(8, 8))
nx.draw(
    G, pos=node_pos, labels=node_labels, with_labels=True, node_color=node_color,
    edge_color='gray', node_size=800, font_size=10
)
plt.title("Room Adjacency Graph (Centroid & Color)")
plt.savefig(f"{img_name}_room_adjacency_graph.png")
plt.close()
print(f"Saved NetworkX room adjacency graph image as '{img_name}_room_adjacency_graph.png'.")

# Print room count by type
room_type_count = {name: 0 for name in room_type_names}
for rid in room_ids:
    mask = (ch_instance == rid)
    room_type_vals = ch_roomtype[mask]
    if room_type_vals.size == 0:
        continue
    room_type_val = np.bincount(room_type_vals.flatten()).argmax()
    if room_type_val < len(room_type_names):
        room_type_count[room_type_names[room_type_val]] += 1

print("Room count for each type:")
for k, v in room_type_count.items():
    print(f"{k}: {v} rooms")