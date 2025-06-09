import cv2
import numpy as np
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import traceback
from skimage.morphology import skeletonize
from skimage.morphology import binary_closing, binary_dilation
from skimage.util import invert
from skimage import measure

# Import graph extraction functions
from data_new import extract_boundary_graph, extract_wall_graph # Assuming merge_nearby_nodes is used by extract_wall_graph

def visualize_nx_graph(graph_data, output_path, graph_title="Graph Visualization", img_shape=None, node_color='red', edge_color='green', show_labels_mod=10):
    """
    Create and save a NetworkX visualization of a generic graph.
    
    Args:
        graph_data: Dictionary representation of the graph
        output_path: Where to save the NetworkX visualization
        graph_title: Title for the plot
        img_shape: Optional shape of original image for setting plot limits
        node_color: Color for graph nodes
        edge_color: Color for graph edges
        show_labels_mod: Show label for every Nth node. If 0 or None, no labels.
    """
    # Create a new NetworkX graph
    G = nx.Graph()
    
    if not graph_data:
        print(f"Graph data is empty for {graph_title}. Skipping visualization.")
        # Create an empty image with a message
        fig, ax = plt.subplots(figsize=(10,10))
        ax.text(0.5, 0.5, f"{graph_title}\n(No data to display)", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=16, color='gray')
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    # Add nodes with positions
    for node_id, node_data in graph_data.items():
        y, x = node_data['pos']
        G.add_node(node_id, pos=(x, y))
    
    # Add edges (connections between nodes)
    for node_id, node_data in graph_data.items():
        for connected_id in node_data.get('connect', []): # Use .get for safety
            if node_id < connected_id:  # Add each edge only once
                G.add_edge(node_id, connected_id)
    
    # Set up the plot
    plt.figure(figsize=(10, 10))
    
    # Get node positions for drawing
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    nx.draw(G, pos, 
           node_size=30,
           node_color=node_color,
           with_labels=False,
           width=1.0,
           edge_color=edge_color)
    
    # Add a few node labels for reference
    if show_labels_mod and show_labels_mod > 0:
        labels = {node_id: str(node_id) for node_id in graph_data.keys() if node_id % show_labels_mod == 0}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='blue')
    
    # Set the plot limits if image shape is provided
    if img_shape:
        plt.xlim(0, img_shape[1])  # Width dimension
        plt.ylim(img_shape[0], 0)  # Height dimension (inverted for image coordinates)
    
    plt.title(f"{graph_title} ({len(graph_data)} nodes)")
    plt.axis('off')  # Hide the axes
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {graph_title} visualization to {output_path}")


def extract_skeleton_graph(floor_plan):
    """Extract a graph from wall edges using Canny detection and component analysis."""
    # Use channel 1 which contains semantic classes
    semantic_channel = floor_plan[:, :, 1]
    
    # Create wall mask (classes 12, 14, 15, 16, 17)
    wall_classes = [12, 14, 15, 16, 17]
    wall_mask = np.zeros_like(semantic_channel, dtype=np.uint8)
    for wall_class in wall_classes:
        wall_mask[semantic_channel == wall_class] = 255

    # Create a visualization of the wall mask as a color image
    walls_vis = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
    walls_vis[wall_mask == 255] = [0, 0, 255]  # Red walls
    
    # Apply Canny edge detection to the wall mask
    edges = cv2.Canny(wall_mask, 50, 150)
    
    # Optional: dilate edges slightly to ensure connectivity
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create a visualization of the edges
    skeleton_vis = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    skeleton_vis[edges > 0] = [0, 255, 0]  # Green edges
    
    # Find connected components in the edge image
    labeled_edges = measure.label(edges, connectivity=2)
    num_components = labeled_edges.max()
    
    print(f"Found {num_components} connected edge components")
    
    # Extract nodes and create individual graphs for each component
    all_nodes = []
    component_graphs = []
    
    h, w = edges.shape
    
    for component_id in range(1, num_components + 1):
        print(f"Processing component {component_id}/{num_components}")
        
        # Get the current component
        component_mask = (labeled_edges == component_id).astype(np.uint8) * 255
        
        # Find endpoints and junctions in this component
        component_nodes = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if component_mask[y, x] > 0:
                    patch = component_mask[y-1:y+2, x-1:x+2]
                    neighbors = np.sum(patch > 0) - 1
                    if neighbors == 1 or neighbors >= 3:
                        component_nodes.append((y, x))
                        skeleton_vis[y, x] = [255, 0, 0]  # Red nodes
        
        print(f"Component {component_id} has {len(component_nodes)} nodes")
        
        # If component has nodes, create a graph for it
        if len(component_nodes) >= 2:
            component_graph = {}
            for i, (y, x) in enumerate(component_nodes):
                node_id = len(all_nodes) + i
                component_graph[node_id] = {
                    'pos': [y, x],
                    'connect': [],
                    'component': component_id
                }
            
            # Connect nodes within this component using simple distance
            node_ids = list(component_graph.keys())
            print(f"Connecting {len(node_ids)} nodes within component {component_id}")
            
            connections_made = 0
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    id1, id2 = node_ids[i], node_ids[j]
                    y1, x1 = component_graph[id1]['pos']
                    y2, x2 = component_graph[id2]['pos']
                    
                    distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                    if distance < 50:  # Connect if close enough
                        component_graph[id1]['connect'].append(id2)
                        component_graph[id2]['connect'].append(id1)
                        connections_made += 1
                        # Visualize component connections in yellow
                        cv2.line(skeleton_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            print(f"Made {connections_made} connections within component {component_id}")
            component_graphs.append(component_graph)
            all_nodes.extend(component_nodes)
        elif len(component_nodes) == 1:
            # Single node component
            y, x = component_nodes[0]
            node_id = len(all_nodes)
            component_graph = {
                node_id: {
                    'pos': [y, x],
                    'connect': [],
                    'component': component_id
                }
            }
            component_graphs.append(component_graph)
            all_nodes.extend(component_nodes)
            print(f"Component {component_id} has single node")
    
    print(f"Total nodes found: {len(all_nodes)}")
    print(f"Total component graphs: {len(component_graphs)}")
    
    # Merge all component graphs into one
    wall_graph = {}
    for component_graph in component_graphs:
        wall_graph.update(component_graph)
    
    print(f"Combined graph has {len(wall_graph)} nodes")
    
    # Now merge nearby nodes from different components
    merge_distance = 15  # Distance threshold for merging nodes
    nodes_to_merge = []
    
    # Find pairs of nodes that should be merged
    node_ids = list(wall_graph.keys())
    print(f"Checking for nodes to merge among {len(node_ids)} nodes")
    
    merge_checks = 0
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            merge_checks += 1
            if merge_checks % 1000 == 0:
                print(f"Processed {merge_checks} merge checks...")
                
            id1, id2 = node_ids[i], node_ids[j]
            
            # Only merge nodes from different components
            if wall_graph[id1]['component'] != wall_graph[id2]['component']:
                y1, x1 = wall_graph[id1]['pos']
                y2, x2 = wall_graph[id2]['pos']
                
                distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                if distance < merge_distance:
                    nodes_to_merge.append((id1, id2))
                    # Visualize merge connections in cyan
                    cv2.line(skeleton_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    print(f"Found {len(nodes_to_merge)} pairs of nodes to merge")
    
    # Perform the merging
    for idx, (id1, id2) in enumerate(nodes_to_merge):
        if idx % 10 == 0:
            print(f"Merging pair {idx+1}/{len(nodes_to_merge)}")
            
        if id1 in wall_graph and id2 in wall_graph:
            # Connect the two nodes
            if id2 not in wall_graph[id1]['connect']:
                wall_graph[id1]['connect'].append(id2)
            if id1 not in wall_graph[id2]['connect']:
                wall_graph[id2]['connect'].append(id1)
            
            # Also connect all neighbors of id2 to id1
            for neighbor in wall_graph[id2]['connect']:
                if neighbor != id1 and neighbor in wall_graph:
                    if neighbor not in wall_graph[id1]['connect']:
                        wall_graph[id1]['connect'].append(neighbor)
                    # Update neighbor's connections
                    if id1 not in wall_graph[neighbor]['connect']:
                        wall_graph[neighbor]['connect'].append(id1)
                    # Remove old connection to id2
                    if id2 in wall_graph[neighbor]['connect']:
                        wall_graph[neighbor]['connect'].remove(id2)
            
            # Remove id2 from the graph
            del wall_graph[id2]
    
    print(f"After merging: {len(wall_graph)} nodes remain")
    
    # Clean up the graph by removing component info and renumbering
    final_graph = {}
    node_mapping = {}
    new_id = 0
    
    for old_id, node_data in wall_graph.items():
        node_mapping[old_id] = new_id
        final_graph[new_id] = {
            'pos': node_data['pos'],
            'connect': []
        }
        new_id += 1
    
    # Update connections with new IDs
    for old_id, node_data in wall_graph.items():
        new_id = node_mapping[old_id]
        for connected_old_id in node_data['connect']:
            if connected_old_id in node_mapping:
                connected_new_id = node_mapping[connected_old_id]
                if connected_new_id not in final_graph[new_id]['connect']:
                    final_graph[new_id]['connect'].append(connected_new_id)
    
    print(f"Final graph has {len(final_graph)} nodes")
    return final_graph, skeleton_vis, walls_vis

def extract_space_canny_graph(floor_plan):
    """Extract a graph from space regions using Canny detection."""
    # Use channel 1 which contains semantic classes
    semantic_channel = floor_plan[:, :, 1]
    
    # Print unique values to understand what we're working with
    unique_values = np.unique(semantic_channel)
    print(f"Unique semantic values: {unique_values}")
    
    # Create space mask - include room/space classes, exclude walls and background
    wall_classes = [12, 14, 15, 16, 17]  # Wall classes
    background_class = 0
    
    # Start with all pixels as potential space
    space_mask = np.ones_like(semantic_channel, dtype=np.uint8) * 255
    
    # Remove wall pixels
    for wall_class in wall_classes:
        space_mask[semantic_channel == wall_class] = 0
    
    # Remove background
    space_mask[semantic_channel == background_class] = 0
    
    # Create a visualization of the space mask
    space_vis = cv2.cvtColor(space_mask, cv2.COLOR_GRAY2BGR)
    space_vis[space_mask == 255] = [0, 255, 0]  # Green spaces
    
    # Apply Canny edge detection to the space mask
    edges = cv2.Canny(space_mask, 50, 150)
    
    # Create a visualization of the edges
    edges_vis = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    edges_vis[edges > 0] = [255, 0, 0]  # Red edges
    
    # Find connected components in the edge image
    labeled_edges = measure.label(edges, connectivity=2)
    num_components = labeled_edges.max()
    
    print(f"Found {num_components} space edge components")
    
    # Create a graph that follows the actual edge structure
    space_graph = {}
    node_id = 0
    h, w = edges.shape
    
    # Process each connected component
    for component_id in range(1, num_components + 1):
        # Get the current component
        component_mask = (labeled_edges == component_id).astype(np.uint8) * 255
        
        # Find corner points and junctions in this component
        component_nodes = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if component_mask[y, x] > 0:
                    # Check 8-neighborhood to find corners/junctions
                    patch = component_mask[y-1:y+2, x-1:x+2]
                    neighbors = np.sum(patch > 0) - 1  # Subtract center pixel
                    
                    # Find endpoints (1 neighbor) or junctions (3+ neighbors)
                    if neighbors == 1 or neighbors >= 3:
                        component_nodes.append((y, x))
                        edges_vis[y, x] = [0, 0, 255]  # Blue nodes
        
        print(f"Component {component_id} has {len(component_nodes)} corner/junction nodes")
        
        if len(component_nodes) == 0:
            continue
            
        # Add nodes to graph
        component_start_id = node_id
        node_positions = {}
        for i, (y, x) in enumerate(component_nodes):
            space_graph[node_id] = {
                'pos': [y, x],
                'connect': []
            }
            node_positions[node_id] = (y, x)
            node_id += 1
        
        # Connect nodes that are actually connected via the edge structure
        component_node_ids = list(range(component_start_id, node_id))
        
        for i, id1 in enumerate(component_node_ids):
            for j, id2 in enumerate(component_node_ids):
                if i >= j:
                    continue
                    
                y1, x1 = node_positions[id1]
                y2, x2 = node_positions[id2]
                
                # Check if there's a path between these nodes along the component
                # Use simple distance check for now, but ensure they're reasonably close
                distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                
                if distance < 100:  # Reasonable connection distance
                    # Check if there's roughly a straight line of edge pixels between them
                    # Simple line check - sample points along the line
                    steps = int(distance)
                    if steps > 0:
                        connected = True
                        for step in range(1, steps):
                            t = step / steps
                            check_y = int(y1 + t * (y2 - y1))
                            check_x = int(x1 + t * (x2 - x1))
                            
                            # Check if this point is close to an edge pixel
                            found_edge = False
                            for dy in [-2, -1, 0, 1, 2]:
                                for dx in [-2, -1, 0, 1, 2]:
                                    cy, cx = check_y + dy, check_x + dx
                                    if (0 <= cy < h and 0 <= cx < w and 
                                        component_mask[cy, cx] > 0):
                                        found_edge = True
                                        break
                                if found_edge:
                                    break
                            
                            if not found_edge:
                                connected = False
                                break
                        
                        if connected:
                            space_graph[id1]['connect'].append(id2)
                            space_graph[id2]['connect'].append(id1)
                            
                            # Visualize connection
                            cv2.line(edges_vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
    
    print(f"Created space graph with {len(space_graph)} nodes")
    return space_graph, edges_vis, space_vis

def debug_floorplan_graphs(image_path, output_dir):
    """Extract and visualize boundary and wall graphs for a floor plan."""
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    if img.ndim < 3 or img.shape[2] < 4: 
        print(f"Error: Image {image_path} does not have enough channels (needs at least 4). Found shape {img.shape}.")
        return
        
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing floor plan: {base_name}")
    print(f"Image shape: {img.shape}")

    # Debug: Print unique values in each channel
    print("Channel 0 unique values:", np.unique(img[:,:,0]))
    print("Channel 1 unique values:", np.unique(img[:,:,1]))
    print("Channel 2 unique values:", np.unique(img[:,:,2]))
    print("Channel 3 unique values:", np.unique(img[:,:,3]))

    try:
        # 1. Extract and Visualize Boundary Graph
        boundary_graph = extract_boundary_graph(img) 
        visualize_nx_graph(
            boundary_graph,
            os.path.join(output_dir, f"{base_name}_boundary_nx.png"),
            graph_title="Boundary Graph NetworkX",
            img_shape=img.shape,
            node_color='red',
            edge_color='green',
            show_labels_mod=10 
        )
        print(f"Successfully processed boundary graph with {len(boundary_graph)} nodes")

        # 2. Create simple space Canny graph
        print("\n--- Creating Simple Space Canny Graph ---")
        space_graph, space_edges_vis, space_vis = extract_space_canny_graph(img)
        
        # Save space region visualization
        space_vis_path = os.path.join(output_dir, f"{base_name}_space_region.png")
        cv2.imwrite(space_vis_path, space_vis)
        print(f"Saved space region visualization to: {space_vis_path}")
        
        # Save space edges visualization
        space_edges_vis_path = os.path.join(output_dir, f"{base_name}_space_edges.png")
        cv2.imwrite(space_edges_vis_path, space_edges_vis)
        print(f"Saved space edges visualization to: {space_edges_vis_path}")
        
        # Visualize space graph
        visualize_nx_graph(
            space_graph,
            os.path.join(output_dir, f"{base_name}_space_graph.png"),
            graph_title="Space Canny Graph",
            img_shape=img.shape,
            node_color='blue',
            edge_color='yellow',
            show_labels_mod=1  # Show all labels since it's a small graph
        )
        print(f"Successfully processed space graph with {len(space_graph)} nodes")
        
    except Exception as e:
        print(f"Error processing graphs for {base_name}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "D:/Desktop/RPLAN/dataset/floorplan_dataset/0.png" 
    
    output_dir = "D:/Desktop/RPLAN/dataset/debug_visualizations"
    
    debug_floorplan_graphs(image_path, output_dir)
    print(f"\nDebug visualizations saved to {output_dir}")