import torch
import numpy as np
from PIL import Image
import os
from folder_paths import get_temp_directory
import math
import json

# Global cache to store node extra data (crop values from frontend)
# Key: node_id, Value: dict with crop_center_x, crop_center_y, crop_width, crop_height
_node_extra_cache = {}

def setup_api_routes(server):
    """Set up API routes to receive crop data from frontend"""
    from aiohttp import web
    
    @server.routes.post("/oa360clip/set_crop")
    async def set_crop(request):
        """Receive crop data from frontend and store in cache"""
        try:
            data = await request.json()
            node_id = data.get('node_id')
            # Convert node_id to string to ensure consistent key type
            node_id_str = str(node_id) if node_id is not None else None
            if node_id_str:
                cache_entry = {
                    'crop_center_x': data.get('crop_center_x'),
                    'crop_center_y': data.get('crop_center_y'),
                    'crop_width': data.get('crop_width'),
                    'crop_height': data.get('crop_height'),
                }
                _node_extra_cache[node_id_str] = cache_entry
                return web.json_response({"status": "ok"})
            return web.json_response({"status": "error", "message": "node_id required"}, status=400)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

# Hook into execution to populate cache from workflow's node extra data
def populate_cache_from_workflow(prompt, node_id=None):
    """Populate cache from workflow's node extra data before execution"""
    try:
        if prompt:
            # prompt is a dict of {node_id: node_data}
            if node_id:
                # Populate for specific node
                node_data = prompt.get(node_id)
                if node_data and 'extra' in node_data:
                    extra = node_data['extra']
                    if extra:
                        cache_data = {
                            'crop_center_x': extra.get('crop_center_x'),
                            'crop_center_y': extra.get('crop_center_y'),
                            'crop_width': extra.get('crop_width'),
                            'crop_height': extra.get('crop_height'),
                        }
                        # Only cache if we have at least one valid value
                        if any(v is not None for v in cache_data.values()):
                            _node_extra_cache[str(node_id)] = cache_data
            else:
                # Populate for all OA360Clip nodes
                for nid, node_data in prompt.items():
                    class_type = node_data.get("class_type")
                    if class_type == "OA360Clip" and "extra" in node_data:
                        extra = node_data['extra']
                        if extra:
                            cache_data = {
                                'crop_center_x': extra.get('crop_center_x'),
                                'crop_center_y': extra.get('crop_center_y'),
                                'crop_width': extra.get('crop_width'),
                                'crop_height': extra.get('crop_height'),
                            }
                            # Only cache if we have at least one valid value
                            if any(v is not None for v in cache_data.values()):
                                _node_extra_cache[str(nid)] = cache_data
    except Exception:
        pass

# 360 Image Projection Functions (reused from original code)
def equirectangular_to_spherical(x, y, img_width, img_height):
    """
    Convert equirectangular coordinates to spherical coordinates (latitude, longitude).
    x, y: pixel coordinates in equirectangular image
    Returns: (latitude in degrees, longitude in degrees)
    """
    # Normalize to [0, 1]
    u = x / img_width
    v = y / img_height
    
    # Convert to spherical coordinates
    # Longitude: 0-360 degrees, centered at 180 (left edge is -180, right edge is +180)
    longitude = (u * 360.0) - 180.0
    
    # Latitude: -90 to +90 degrees
    # Standard equirectangular: top (v=0) = +90° (north pole), bottom (v=1) = -90° (south pole)
    latitude = 90.0 - (v * 180.0)  # Standard: top = +90°, bottom = -90°
    
    return latitude, longitude


def spherical_to_cartesian(lat, lon):
    """
    Convert spherical coordinates to 3D cartesian coordinates.
    lat, lon: in degrees
    Returns: (x, y, z) unit vector
    Standard convention:
    - x = east (longitude 0°)
    - y = north (latitude +90° = north pole)
    - z = up from equator
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Standard spherical to cartesian:
    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.sin(lat_rad)  # +90° lat = +1, -90° lat = -1
    z = -math.cos(lat_rad) * math.sin(lon_rad)
    
    return x, y, z


def output_pixel_to_equirectangular(out_x, out_y, output_width, output_height,
                                     crop_left, crop_top, crop_width, crop_height,
                                     img_width, img_height,
                                     forward, up, right, fov_horizontal, fov_vertical):
    """
    Map an output pixel coordinate to equirectangular image coordinate using standard pinhole projection.
    
    Args:
        out_x, out_y: Output pixel coordinates (0 to output_width-1, 0 to output_height-1)
        output_width, output_height: Output image dimensions
        crop_left, crop_top, crop_width, crop_height: Crop region in equirectangular image
        img_width, img_height: Full equirectangular image dimensions
        forward, up, right: Camera basis vectors (tuples)
        fov_horizontal, fov_vertical: Field of view in degrees
    
    Returns:
        (x_eq, y_eq): Equirectangular pixel coordinates, or (None, None) if behind projection plane
    """
    # Standard pinhole camera projection
    # Pixel → normalized device coordinates [-1, 1]
    u = (out_x - output_width / 2.0) / (output_width / 2.0)
    v = (out_y - output_height / 2.0) / (output_height / 2.0)
    
    # NDC → 3D ray in camera space (pinhole projection)
    fov_h_rad = math.radians(fov_horizontal)
    fov_v_rad = math.radians(fov_vertical)
    cam_dir_x = u * math.tan(fov_h_rad / 2.0)
    cam_dir_y = -v * math.tan(fov_v_rad / 2.0)  # Negative v because screen y is inverted
    cam_dir_z = 1.0
    cam_dir_len = math.sqrt(cam_dir_x**2 + cam_dir_y**2 + cam_dir_z**2)
    if cam_dir_len < 0.001:
        return None, None  # Behind projection plane
    cam_dir_x /= cam_dir_len
    cam_dir_y /= cam_dir_len
    cam_dir_z /= cam_dir_len
    
    # Rotate to world space
    world_dir_x = -cam_dir_x * right[0] + cam_dir_y * up[0] + cam_dir_z * forward[0]
    world_dir_y = -cam_dir_x * right[1] + cam_dir_y * up[1] + cam_dir_z * forward[1]
    world_dir_z = -cam_dir_x * right[2] + cam_dir_y * up[2] + cam_dir_z * forward[2]
    
    # Convert to lat/lon
    target_lat_rad = math.asin(max(-1.0, min(1.0, world_dir_y)))
    target_lon_rad = math.atan2(-world_dir_z, world_dir_x)
    target_lat = math.degrees(target_lat_rad)
    target_lon = math.degrees(target_lon_rad)
    
    # Convert to equirectangular coordinates
    u_eq = (target_lon + 180.0) / 360.0
    v_eq = (90.0 - target_lat) / 180.0
    x_eq = int(u_eq * img_width) % img_width
    y_eq = int(v_eq * img_height)
    y_eq = max(0, min(img_height - 1, y_eq))
    
    return x_eq, y_eq


def equirectangular_to_perspective(equirect_img, crop_left, crop_top, crop_width, crop_height, 
                                    output_width, output_height, img_width, img_height):
    """
    Extract a perspective view from an equirectangular 360 image using gnomonic projection.
    
    Args:
        equirect_img: numpy array of shape (H, W, C) in range [0, 255]
        crop_left, crop_top: crop region top-left corner
        crop_width, crop_height: crop region dimensions
        output_width, output_height: desired output dimensions
        img_width, img_height: full equirectangular image dimensions
    
    Returns:
        output_image: numpy array of shape (output_height, output_width, C) in range [0, 255]
    """
    # Calculate center of crop in spherical coordinates
    center_x = crop_left + crop_width / 2.0
    center_y = crop_top + crop_height / 2.0
    
    center_lat, center_lon = equirectangular_to_spherical(center_x, center_y, img_width, img_height)
    
    # Limit viewing angle to avoid extreme distortion
    max_viewing_angle = 85.0  # degrees from forward direction
    
    # Calculate actual crop bounds for verification
    crop_right_pixel = crop_left + crop_width
    crop_bottom_pixel = crop_top + crop_height
    
    # Convert crop corners to spherical to verify
    top_left_lat, top_left_lon = equirectangular_to_spherical(crop_left, crop_top, img_width, img_height)
    top_right_lat, top_right_lon = equirectangular_to_spherical(crop_right_pixel, crop_top, img_width, img_height)
    bottom_left_lat, bottom_left_lon = equirectangular_to_spherical(crop_left, crop_bottom_pixel, img_width, img_height)
    bottom_right_lat, bottom_right_lon = equirectangular_to_spherical(crop_right_pixel, crop_bottom_pixel, img_width, img_height)
    
    # Check maximum angular distance from center to any corner
    center_cart = spherical_to_cartesian(center_lat, center_lon)
    max_angle_from_center = 0.0
    for corner_lat, corner_lon in [(top_left_lat, top_left_lon), (top_right_lat, top_right_lon),
                                   (bottom_left_lat, bottom_left_lon), (bottom_right_lat, bottom_right_lon)]:
        corner_cart = spherical_to_cartesian(corner_lat, corner_lon)
        dot_product = corner_cart[0] * center_cart[0] + corner_cart[1] * center_cart[1] + corner_cart[2] * center_cart[2]
        angle_rad = math.acos(max(-1.0, min(1.0, dot_product)))
        angle_deg = math.degrees(angle_rad)
        max_angle_from_center = max(max_angle_from_center, angle_deg)
    
    if max_angle_from_center > max_viewing_angle:
        print(f"[OA360Clip] WARNING: Crop region extends beyond {max_viewing_angle}° from center!")
        print(f"[OA360Clip] Gnomonic projection will have severe distortion near the edges.")
    
    # Calculate angular width from crop corners
    top_left_lat_rad = math.radians(top_left_lat)
    top_right_lat_rad = math.radians(top_right_lat)
    
    # Longitude span in degrees
    lon_span = abs(top_right_lon - top_left_lon)
    if lon_span > 180:
        lon_span = 360 - lon_span
    
    # Latitude span in degrees
    lat_span = abs(top_left_lat - bottom_left_lat)
    
    # Calculate FOV
    fov_horizontal = lon_span
    fov_horizontal = min(fov_horizontal, max_viewing_angle * 2.0)
    
    fov_vertical = lat_span
    fov_vertical = min(fov_vertical, max_viewing_angle * 2.0)
    
    # Additional safety: if the crop region is very close to a pole, further limit FOV
    if abs(center_lat) > 75.0:
        max_fov_near_pole = 60.0
        fov_horizontal = min(fov_horizontal, max_fov_near_pole)
        fov_vertical = min(fov_vertical, max_fov_near_pole)
    
    # Precompute center direction in cartesian
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)
    
    # Create output image
    output = np.zeros((output_height, output_width, equirect_img.shape[2]), dtype=np.uint8)
    
    # Pre-compute camera basis vectors
    center_cart = spherical_to_cartesian(center_lat, center_lon)
    cx, cy, cz = center_cart
    forward = (cx, cy, cz)
    
    # Calculate "up" direction: from center towards top edge
    top_edge_lat = (top_left_lat + top_right_lat) / 2.0
    top_edge_lon = center_lon
    top_edge_cart = spherical_to_cartesian(top_edge_lat, top_edge_lon)
    tx, ty, tz = top_edge_cart
    to_top = (tx - cx, ty - cy, tz - cz)
    to_top_dot_forward = to_top[0] * forward[0] + to_top[1] * forward[1] + to_top[2] * forward[2]
    up_proj = (to_top[0] - to_top_dot_forward * forward[0],
              to_top[1] - to_top_dot_forward * forward[1],
              to_top[2] - to_top_dot_forward * forward[2])
    up_len = math.sqrt(up_proj[0]**2 + up_proj[1]**2 + up_proj[2]**2)
    if up_len > 0.001:
        up = (up_proj[0] / up_len, up_proj[1] / up_len, up_proj[2] / up_len)
    else:
        world_up = (0.0, 1.0, 0.0)
        up_dot_forward = world_up[0] * forward[0] + world_up[1] * forward[1] + world_up[2] * forward[2]
        up_proj = (world_up[0] - up_dot_forward * forward[0],
                  world_up[1] - up_dot_forward * forward[1],
                  world_up[2] - up_dot_forward * forward[2])
        up_len = math.sqrt(up_proj[0]**2 + up_proj[1]**2 + up_proj[2]**2)
        if up_len > 0.001:
            up = (up_proj[0] / up_len, up_proj[1] / up_len, up_proj[2] / up_len)
        else:
            up = (1.0, 0.0, 0.0)
    
    right = (forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0])
    
    for out_y in range(output_height):
        for out_x in range(output_width):
            # Use the unified mapping function with standard pinhole projection
            x_eq, y_eq = output_pixel_to_equirectangular(
                out_x, out_y, output_width, output_height,
                crop_left, crop_top, crop_width, crop_height,
                img_width, img_height,
                forward, up, right,
                fov_horizontal, fov_vertical
            )
            
            if x_eq is None:
                # Behind projection plane - use black
                output[out_y, out_x] = [0, 0, 0] if equirect_img.shape[2] == 3 else [0, 0, 0, 0]
                continue
            
            output[out_y, out_x] = equirect_img[y_eq, x_eq]
    
    return output


class OA360Clip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "output_width": ("INT", {"default": 1280, "min": 1, "max": 8192}),
                "output_height": ("INT", {"default": 720, "min": 1, "max": 8192}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "clip"
    CATEGORY = "image/transform"
    DESCRIPTION = "Extract a perspective view from a 360 equirectangular image with draggable crop zone."
    
    @classmethod
    def IS_CHANGED(cls, image, output_width=1280, output_height=720, node_id=None):
        # This will be called to check if the node needs re-execution
        # We use a hash of the crop properties stored in the node instance
        # Since we can't access instance properties here, we'll use a simpler approach
        # The frontend will mark the workflow as dirty when crop changes
        return float("nan")  # Always re-execute when inputs change

    def clip(
        self,
        image: torch.Tensor,
        output_width: int = 1280,
        output_height: int = 720,
        node_id=None,
    ):
        batch_size, current_height, current_width, channels = image.shape
        
        # Try to populate cache from workflow's node extra data if not already in cache
        # This happens when workflow is loaded or when serialize() sends the data
        if node_id and node_id not in _node_extra_cache:
            try:
                # Access the current execution context to get the workflow
                from comfy_execution.utils import get_executing_context
                context = get_executing_context()
                if context:
                    # We can't directly access dynprompt from here, but the cache should
                    # be populated via the API endpoint or serialize()
                    pass
            except Exception:
                pass

        # Get crop region from UI state (set by frontend interaction)
        # Default to center of image with reasonable size
        # Maximum crop width is 25% of image width
        max_crop_width = int(current_width * 0.25)
        
        # Get crop region from multiple sources in priority order:
        # 1) Global cache (set from workflow's node extra data via frontend serialize)
        # 2) Instance attributes (set from previous execution)
        # 3) Defaults
        
        # Try to get from global cache first (this is populated from workflow's node extra data)
        # Convert node_id to string to match cache keys
        node_id_str = str(node_id) if node_id is not None else None
        node_extra = _node_extra_cache.get(node_id_str, {}) if node_id_str else {}
        crop_center_x = node_extra.get('crop_center_x')
        crop_center_y = node_extra.get('crop_center_y')
        crop_width = node_extra.get('crop_width')
        crop_height = node_extra.get('crop_height')
        
        # Fallback to instance attributes (set from previous execution)
        if crop_center_x is None:
            crop_center_x = getattr(self, 'crop_center_x', None)
        if crop_center_y is None:
            crop_center_y = getattr(self, 'crop_center_y', None)
        if crop_width is None:
            crop_width = getattr(self, 'crop_width', None)
        if crop_height is None:
            crop_height = getattr(self, 'crop_height', None)
        
        # If still not set, use defaults (first run or after reset)
        if crop_center_x is None:
            crop_center_x = current_width // 2
        if crop_center_y is None:
            crop_center_y = current_height // 2
        if crop_width is None:
            crop_width = min(512, max_crop_width)
        if crop_height is None:
            crop_height = min(512, current_height)
        
        # Enforce maximum width constraint
        crop_width = min(crop_width, max_crop_width)
        
        # Handle wrapping for center_x (equirectangular images wrap horizontally)
        # Normalize center_x to [0, current_width) range
        crop_center_x = crop_center_x % current_width
        if crop_center_x < 0:
            crop_center_x += current_width

        # Calculate crop_left and crop_top from center
        # Allow wrapping for crop_left (it can be negative or > current_width)
        crop_left = crop_center_x - crop_width // 2
        crop_top = max(0, min(crop_center_y - crop_height // 2, current_height - crop_height))
        
        # Handle wrapping for crop_left - normalize to [0, current_width) range
        # crop_left can be negative or > current_width due to wrapping
        crop_left = crop_left % current_width
        if crop_left < 0:
            crop_left += current_width
        
        # Ensure crop fits within image bounds and respects max width (25%)
        # Note: crop_left is now normalized, so we can calculate normally
        max_crop_width = int(current_width * 0.25)
        crop_width = min(crop_width, current_width - crop_left, max_crop_width)
        crop_height = min(crop_height, current_height - crop_top)
        
        # Recalculate center if crop was adjusted
        crop_center_x = crop_left + crop_width // 2
        crop_center_y = crop_top + crop_height // 2
        
        # Store the final crop values on the instance so they persist for next execution
        # This ensures the frontend changes are preserved
        self.crop_center_x = crop_center_x
        self.crop_center_y = crop_center_y
        self.crop_width = crop_width
        self.crop_height = crop_height

        # Convert torch tensor to numpy for processing
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # Calculate scale factors for frontend
        scale_factors = None
        if crop_width > 0 and crop_height > 0:
            # Calculate crop corners in spherical coordinates
            center_x = crop_left + crop_width / 2.0
            center_y = crop_top + crop_height / 2.0
            center_lat, center_lon = equirectangular_to_spherical(center_x, center_y, current_width, current_height)
            
            top_left_lat, top_left_lon = equirectangular_to_spherical(crop_left, crop_top, current_width, current_height)
            top_right_lat, top_right_lon = equirectangular_to_spherical(crop_left + crop_width, crop_top, current_width, current_height)
            bottom_left_lat, bottom_left_lon = equirectangular_to_spherical(crop_left, crop_top + crop_height, current_width, current_height)
            
            lon_span = abs(top_right_lon - top_left_lon)
            if lon_span > 180:
                lon_span = 360 - lon_span
            lat_span = abs(top_left_lat - bottom_left_lat)
            
            fov_h = min(lon_span, 170.0)
            fov_v = min(lat_span, 170.0)
            
            fov_h_rad = math.radians(fov_h)
            fov_v_rad = math.radians(fov_v)
            scale_x = math.tan(fov_h_rad / 2.0)
            scale_y = math.tan(fov_v_rad / 2.0)
            
            scale_factors = {
                "top_scale_x": scale_x,
                "top_scale_y": scale_y,
                "bottom_scale_x": scale_x,
                "bottom_scale_y": scale_y,
                "fov_horizontal": fov_h,
                "fov_vertical": fov_v
            }
        
        # Apply equirectangular to perspective projection
        perspective_img = equirectangular_to_perspective(
            img_np,
            crop_left,
            crop_top,
            crop_width,
            crop_height,
            output_width,
            output_height,
            current_width,
            current_height
        )
        
        # Convert back to torch tensor and move to correct device
        perspective_tensor = torch.from_numpy(perspective_img.astype(np.float32) / 255.0).unsqueeze(0)
        perspective_tensor = perspective_tensor.to(image.device)
        
        # Handle batch dimension
        if batch_size > 1:
            cropped_image = perspective_tensor.repeat(batch_size, 1, 1, 1)
        else:
            cropped_image = perspective_tensor

        # Save preview image for frontend
        original_filename = None
        if batch_size > 0:
            img_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            temp_dir = get_temp_directory()
            filename_hash = hash(f"{node_id}_{current_width}x{current_height}")
            original_filename = f"oa360clip_original_{abs(filename_hash)}.png"
            filepath = os.path.join(temp_dir, original_filename)
            os.makedirs(temp_dir, exist_ok=True)
            try:
                pil_image.save(filepath)
            except Exception as e:
                print(f"[OA360Clip] Error saving preview image: {e}")
                original_filename = None

        # Save output preview image
        output_filename = None
        if batch_size > 0:
            output_array = (cropped_image[0].cpu().numpy() * 255).astype(np.uint8)
            output_pil = Image.fromarray(output_array)
            temp_dir = get_temp_directory()
            output_hash = hash(f"{node_id}_output_{current_width}x{current_height}")
            output_filename = f"oa360clip_output_{abs(output_hash)}.png"
            output_filepath = os.path.join(temp_dir, output_filename)
            try:
                output_pil.save(output_filepath)
            except Exception as e:
                print(f"[OA360Clip] Error saving output preview image: {e}")
                output_filename = None

        # Only send crop_info back to frontend if values haven't changed
        # This prevents overwriting user's drag changes
        # Check if the stored values match what we calculated
        stored_center_x = getattr(self, 'crop_center_x', None)
        stored_center_y = getattr(self, 'crop_center_y', None)
        stored_width = getattr(self, 'crop_width', None)
        stored_height = getattr(self, 'crop_height', None)
        
        # Only send back if values are significantly different (more than 1 pixel)
        # This allows for minor adjustments but prevents overwriting user changes
        values_changed = (
            stored_center_x is None or stored_center_y is None or stored_width is None or stored_height is None or
            abs(stored_center_x - crop_center_x) > 1 or
            abs(stored_center_y - crop_center_y) > 1 or
            abs(stored_width - crop_width) > 1 or
            abs(stored_height - crop_height) > 1
        )
        
        crop_info_for_frontend = {
            "center_x": crop_center_x,
            "center_y": crop_center_y,
            "width": crop_width,
            "height": crop_height,
            "output_width": output_width,
            "output_height": output_height,
            "scale_factors": scale_factors,
            "original_size": [current_width, current_height],
            "cropped_size": [output_width, output_height],
        }

        images_custom = []
        if original_filename:
            images_custom.append({
                "filename": original_filename,
                "subfolder": "",
                "type": "temp"
            })
        if output_filename:
            images_custom.append({
                "filename": output_filename,
                "subfolder": "",
                "type": "temp"
            })
        
        return {
            "ui": {
                "images_custom": images_custom,
                "crop_info": [crop_info_for_frontend]
            },
            "result": (cropped_image,),
        }


NODE_CLASS_MAPPINGS = {
    "OA360Clip": OA360Clip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OA360Clip": "OA 360 Clip",
}

WEB_DIRECTORY = "./web"

