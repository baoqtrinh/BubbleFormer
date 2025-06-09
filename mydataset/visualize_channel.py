import cv2
import os
import sys

def visualize_channel4(image_path, output_path):
    """
    Load a 4-channel image, extract channel 4 (index 3) and save it as a grayscale image.
    
    Args:
        image_path: Path to the input 4-channel image.
        output_path: Path to save the extracted Channel 4 image.
    """
    # Load the image in unchanged mode to get all channels
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Check if the image has at least 4 channels
    if img.shape[2] < 4:
        print(f"Error: Image {image_path} does not have 4 channels")
        return
    
    # Extract Channel 4 (index 3)
    channel4 = img[:,:,3]
    
    # Save the raw channel 4 as a grayscale image
    cv2.imwrite(output_path, channel4)
    print(f"Saved Channel 4 visualization to {output_path}")

def visualize_channel4_contours(image_path, output_path):
    """
    Load a 4-channel image, extract channel 4 (index 3), compute contours from it, 
    draw the contours on a color image and save it.
    
    Args:
        image_path: Path to the input 4-channel image.
        output_path: Path to save the contour visualization.
    """
    # Load the image in unchanged mode
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    if img.shape[2] < 4:
        print(f"Error: Image {image_path} does not have 4 channels")
        return
    
    # Extract Channel 4 (index 3)
    channel4 = img[:,:,3]
    
    # Threshold channel 4 to create a binary mask.
    # Adjust the threshold value if needed.
    ret, thresh = cv2.threshold(channel4, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert channel4 to BGR so we can draw colored contours
    contour_img = cv2.cvtColor(channel4, cv2.COLOR_GRAY2BGR)
    
    # Draw contours on the image (green color, thickness 2)
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
    
    # Save the result
    cv2.imwrite(output_path, contour_img)
    print(f"Saved Channel 4 contour visualization to {output_path}")

def visualize_contour_sampling(image_path, output_path, step=5):
    """
    Extract channel4 contours, sample every `step`th point,
    and draw both the full contour (green) and sampled points (red).
    """
    import cv2, numpy as np, os

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        print("Invalid image")
        return

    ch4 = img[:,:,3]
    _, th = cv2.threshold(ch4, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found")
        return

    # pick the largest contour
    contour = max(contours, key=lambda c: cv2.contourArea(c))

    # convert to BGR for drawing
    vis = cv2.cvtColor(ch4, cv2.COLOR_GRAY2BGR)

    # draw full contour
    cv2.drawContours(vis, [contour], -1, (0,255,0), 1)

    # draw sampled points
    for i in range(0, len(contour), step):
        x, y = contour[i][0]
        cv2.circle(vis, (x,y), 3, (0,0,255), -1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"Saved contour sampling viz to {output_path}")

if __name__ == "__main__":
    # If an image path is provided as command-line argument, use it; otherwise, use default.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "D:/Desktop/RPLAN/dataset/floorplan_dataset/0.png"
    
    # Create output directory for visualizations if not exists
    output_dir = "D:/Desktop/RPLAN/dataset/debug_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Export channel4 raw visualization
    channel4_output = os.path.join(output_dir, f"{base_name}_channel4.png")
    visualize_channel4(image_path, channel4_output)
    
    # Export contour visualization from channel4
    contours_output = os.path.join(output_dir, f"{base_name}_channel4_contours.png")
    visualize_channel4_contours(image_path, contours_output)

    # visualize sampled contour points
    samp_out = os.path.join(output_dir, f"{base_name}_contour_sampling.png")
    visualize_contour_sampling(image_path, samp_out, step=5)