import { app } from "../../scripts/app.js";

// 360 Projection utilities (reused from original code)
function equirectangularToSpherical(x, y, imgWidth, imgHeight) {
  const u = x / imgWidth;
  const v = y / imgHeight;
  const longitude = (u * 360.0) - 180.0;
  const latitude = 90.0 - (v * 180.0);
  return { latitude, longitude };
}

function sphericalToCartesian(lat, lon) {
  const latRad = (lat * Math.PI) / 180.0;
  const lonRad = (lon * Math.PI) / 180.0;
  const x = Math.cos(latRad) * Math.cos(lonRad);
  const y = Math.sin(latRad);
  const z = -Math.cos(latRad) * Math.sin(lonRad);
  return { x, y, z };
}

function outputPixelToEquirectangular(
  outX, outY, outputWidth, outputHeight,
  centerLat, centerLon,
  fovHorizontal, fovVertical,
  imgWidth, imgHeight
) {
  const u = (outX - outputWidth / 2.0) / (outputWidth / 2.0);
  const v = (outY - outputHeight / 2.0) / (outputHeight / 2.0);
  
  const fovHRad = (fovHorizontal * Math.PI) / 180.0;
  const fovVRad = (fovVertical * Math.PI) / 180.0;
  let camDirX = u * Math.tan(fovHRad / 2.0);
  let camDirY = -v * Math.tan(fovVRad / 2.0);
  let camDirZ = 1.0;
  
  const camDirLen = Math.sqrt(camDirX * camDirX + camDirY * camDirY + camDirZ * camDirZ);
  if (camDirLen < 0.001) {
    return null;
  }
  
  camDirX /= camDirLen;
  camDirY /= camDirLen;
  camDirZ /= camDirLen;
  
  const centerCart = sphericalToCartesian(centerLat, centerLon);
  const forward = { x: centerCart.x, y: centerCart.y, z: centerCart.z };
  
  const topLat = centerLat + fovVertical / 2.0;
  const topCart = sphericalToCartesian(topLat, centerLon);
  const toTop = {
    x: topCart.x - forward.x,
    y: topCart.y - forward.y,
    z: topCart.z - forward.z
  };
  const toTopDotForward = toTop.x * forward.x + toTop.y * forward.y + toTop.z * forward.z;
  const upProj = {
    x: toTop.x - toTopDotForward * forward.x,
    y: toTop.y - toTopDotForward * forward.y,
    z: toTop.z - toTopDotForward * forward.z
  };
  const upLen = Math.sqrt(upProj.x * upProj.x + upProj.y * upProj.y + upProj.z * upProj.z);
  let up;
  if (upLen > 0.001) {
    up = { x: upProj.x / upLen, y: upProj.y / upLen, z: upProj.z / upLen };
  } else {
    const worldUp = { x: 0.0, y: 1.0, z: 0.0 };
    const upDotForward = worldUp.x * forward.x + worldUp.y * forward.y + worldUp.z * forward.z;
    const upProj2 = {
      x: worldUp.x - upDotForward * forward.x,
      y: worldUp.y - upDotForward * forward.y,
      z: worldUp.z - upDotForward * forward.z
    };
    const upLen2 = Math.sqrt(upProj2.x * upProj2.x + upProj2.y * upProj2.y + upProj2.z * upProj2.z);
    if (upLen2 > 0.001) {
      up = { x: upProj2.x / upLen2, y: upProj2.y / upLen2, z: upProj2.z / upLen2 };
    } else {
      up = { x: 1.0, y: 0.0, z: 0.0 };
    }
  }
  
  const right = {
    x: forward.y * up.z - forward.z * up.y,
    y: forward.z * up.x - forward.x * up.z,
    z: forward.x * up.y - forward.y * up.x
  };
  
  const worldDirX = -camDirX * right.x + camDirY * up.x + camDirZ * forward.x;
  const worldDirY = -camDirX * right.y + camDirY * up.y + camDirZ * forward.y;
  const worldDirZ = -camDirX * right.z + camDirY * up.z + camDirZ * forward.z;
  
  const targetLatRad = Math.asin(Math.max(-1.0, Math.min(1.0, worldDirY)));
  const targetLonRad = Math.atan2(-worldDirZ, worldDirX);
  const targetLat = (targetLatRad * 180.0) / Math.PI;
  const targetLon = (targetLonRad * 180.0) / Math.PI;
  
  const uEq = (targetLon + 180.0) / 360.0;
  const vEq = (90.0 - targetLat) / 180.0;
  let xEq = Math.floor(uEq * imgWidth) % imgWidth;
  if (xEq < 0) xEq += imgWidth;
  let yEq = Math.floor(vEq * imgHeight);
  yEq = Math.max(0, Math.min(imgHeight - 1, yEq));
  
  return { x: xEq, y: yEq };
}

function calculateFOVFromCrop(centerX, centerY, cropWidth, cropHeight, imgWidth, imgHeight) {
  const cropLeft = centerX - cropWidth / 2.0;
  const cropTop = centerY - cropHeight / 2.0;
  const topLeft = equirectangularToSpherical(cropLeft, cropTop, imgWidth, imgHeight);
  const topRight = equirectangularToSpherical(cropLeft + cropWidth, cropTop, imgWidth, imgHeight);
  const bottomLeft = equirectangularToSpherical(cropLeft, cropTop + cropHeight, imgWidth, imgHeight);
  
  let lonSpan = Math.abs(topRight.longitude - topLeft.longitude);
  if (lonSpan > 180) {
    lonSpan = 360 - lonSpan;
  }
  const latSpan = Math.abs(topLeft.latitude - bottomLeft.latitude);
  
  return {
    horizontal: Math.min(lonSpan, 170.0),
    vertical: Math.min(latSpan, 170.0)
  };
}

function getPreviewArea(node) {
  if (!node.size || node.size[0] <= 0 || node.size[1] <= 0) {
    return null;
  }
  
  // IMPORTANT: This function returns coordinates relative to the node (for use in onDrawForeground)
  // NOT absolute canvas coordinates
  const titleBarHeight = 30;
  const widgetHeight = (node.widgets?.filter((w) => !w.hidden) || []).length * 25;
  const padding = 10;
  const bottomPadding = 15; // Extra padding at bottom
  const spacing = 20;
  const previewY = titleBarHeight + widgetHeight + padding; // Relative to node top
  
  // Calculate available space - content scales to fit
  const availableWidth = node.size[0] - 40;
  const availableHeight = node.size[1] - previewY - padding - bottomPadding;
  
  // Calculate preview area (background container)
  const previewAreaHeight = Math.max(100, availableHeight);
  const previewAreaWidth = availableWidth;
  
  // Calculate actual displayed image size maintaining aspect ratio
  let displayedImageWidth = previewAreaWidth;
  let displayedImageHeight = previewAreaHeight;
  let displayedImageX = (node.size[0] - previewAreaWidth) / 2;
  let displayedImageY = previewY;
  
  if (node.properties.actualImageWidth && node.properties.actualImageHeight) {
    const imageAspect = node.properties.actualImageWidth / node.properties.actualImageHeight;
    const availableAspect = previewAreaWidth / previewAreaHeight;
    
    if (imageAspect > availableAspect) {
      // Image is wider - fit to width
      displayedImageWidth = previewAreaWidth;
      displayedImageHeight = previewAreaWidth / imageAspect;
    } else {
      // Image is taller - fit to height
      displayedImageHeight = previewAreaHeight;
      displayedImageWidth = previewAreaHeight * imageAspect;
    }
    
    // Center the displayed image within the preview area
    displayedImageX = (node.size[0] - displayedImageWidth) / 2;
    displayedImageY = previewY + (previewAreaHeight - displayedImageHeight) / 2;
  }
  
  // Return the displayed image area (not the preview container)
  // This is what we use for coordinate calculations
  return {
    imageX: displayedImageX, // Centered horizontally (relative to node)
    imageY: displayedImageY, // Relative to node top edge
    imageWidth: displayedImageWidth,
    imageHeight: displayedImageHeight,
    // Also include preview area for reference (background container)
    x: (node.size[0] - previewAreaWidth) / 2,
    y: previewY,
    width: previewAreaWidth,
    height: previewAreaHeight
  };
}

function getPreviewLocalPos(nodePos, canvasPos, preview) {
  // Convert canvas coordinates to local preview coordinates
  // canvasPos is in absolute canvas coordinates
  // preview.imageX and preview.imageY are relative to node, so we need nodePos + preview.imageX/Y to get absolute
  // Use displayed image position (not preview area) for coordinate conversion
  const imageAbsX = nodePos[0] + preview.imageX;
  const imageAbsY = nodePos[1] + preview.imageY;
  return {
    x: canvasPos[0] - imageAbsX,
    y: canvasPos[1] - imageAbsY
  };
}

// Point-in-polygon test using ray casting algorithm
function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersect = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

// Get the 4 corner points of the output rectangle mapped to equirectangular
function getCornerPoints(cropInfo, imageWidth, imageHeight) {
  const { center_x, center_y, width, height, output_width, output_height } = cropInfo;
  
  if (center_x === undefined || center_y === undefined || !width || !height || !imageWidth || !imageHeight) {
    return null;
  }
  
  const centerX = center_x;
  const centerY = center_y;
  const center = equirectangularToSpherical(centerX, centerY, imageWidth, imageHeight);
  const fov = calculateFOVFromCrop(centerX, centerY, width, height, imageWidth, imageHeight);
  
  const outputWidth = output_width || width;
  const outputHeight = output_height || height;
  
  // Get the 4 corners of the output rectangle
  const corners = [
    { outX: 0, outY: 0 }, // Top-left
    { outX: outputWidth - 1, outY: 0 }, // Top-right
    { outX: outputWidth - 1, outY: outputHeight - 1 }, // Bottom-right
    { outX: 0, outY: outputHeight - 1 } // Bottom-left
  ];
  
  const mappedCorners = [];
  for (const corner of corners) {
    const mapped = outputPixelToEquirectangular(
      corner.outX, corner.outY, outputWidth, outputHeight,
      center.latitude, center.longitude,
      fov.horizontal, fov.vertical,
      imageWidth, imageHeight
    );
    if (mapped) {
      mappedCorners.push(mapped);
    }
  }
  
  return mappedCorners.length === 4 ? mappedCorners : null;
}

// Get all outline points for drawing and hit detection
function getOutlinePoints(cropInfo, imageWidth, imageHeight) {
  const { center_x, center_y, width, height, output_width, output_height } = cropInfo;
  
  if (center_x === undefined || center_y === undefined || !width || !height || !imageWidth || !imageHeight) {
    return [];
  }
  
  const centerX = center_x;
  const centerY = center_y;
  const center = equirectangularToSpherical(centerX, centerY, imageWidth, imageHeight);
  const fov = calculateFOVFromCrop(centerX, centerY, width, height, imageWidth, imageHeight);
  
  const outputWidth = output_width || width;
  const outputHeight = output_height || height;
  
  // Use more samples for better edge detection
  const numSamples = 100;
  const points = [];
  
  // Sample edges of output rectangle
  for (let i = 0; i <= numSamples; i++) {
    const outX = (i / numSamples) * (outputWidth - 1);
    const outY = 0;
    const mapped = outputPixelToEquirectangular(
      outX, outY, outputWidth, outputHeight,
      center.latitude, center.longitude,
      fov.horizontal, fov.vertical,
      imageWidth, imageHeight
    );
    if (mapped) {
      // Normalize X to handle wrapping (ensure it's in [0, imageWidth))
      mapped.x = ((mapped.x % imageWidth) + imageWidth) % imageWidth;
      points.push(mapped);
    }
  }
  
  for (let i = 1; i <= numSamples; i++) {
    const outX = outputWidth - 1;
    const outY = (i / numSamples) * (outputHeight - 1);
    const mapped = outputPixelToEquirectangular(
      outX, outY, outputWidth, outputHeight,
      center.latitude, center.longitude,
      fov.horizontal, fov.vertical,
      imageWidth, imageHeight
    );
    if (mapped) {
      mapped.x = ((mapped.x % imageWidth) + imageWidth) % imageWidth;
      points.push(mapped);
    }
  }
  
  for (let i = 1; i <= numSamples; i++) {
    const outX = (1 - i / numSamples) * (outputWidth - 1);
    const outY = outputHeight - 1;
    const mapped = outputPixelToEquirectangular(
      outX, outY, outputWidth, outputHeight,
      center.latitude, center.longitude,
      fov.horizontal, fov.vertical,
      imageWidth, imageHeight
    );
    if (mapped) {
      mapped.x = ((mapped.x % imageWidth) + imageWidth) % imageWidth;
      points.push(mapped);
    }
  }
  
  for (let i = 1; i < numSamples; i++) {
    const outX = 0;
    const outY = (1 - i / numSamples) * (outputHeight - 1);
    const mapped = outputPixelToEquirectangular(
      outX, outY, outputWidth, outputHeight,
      center.latitude, center.longitude,
      fov.horizontal, fov.vertical,
      imageWidth, imageHeight
    );
    if (mapped) {
      mapped.x = ((mapped.x % imageWidth) + imageWidth) % imageWidth;
      points.push(mapped);
    }
  }
  
  return points;
}

function drawProjectionOutline(ctx, cropInfo, previewArea, imageWidth, imageHeight) {
  const points = getOutlinePoints(cropInfo, imageWidth, imageHeight);
  
  if (points.length === 0) return [];
  
  // Use displayed image size for coordinate conversion (maintains aspect ratio)
  const scaleX = previewArea.imageWidth / imageWidth;
  const scaleY = previewArea.imageHeight / imageHeight;
  
  // Convert to preview coordinates
  // Note: previewArea.imageX and previewArea.imageY are relative to node, so we add them directly
  const previewPoints = points.map(p => ({
    x: p.x * scaleX + previewArea.imageX,
    y: p.y * scaleY + previewArea.imageY,
    origX: p.x, // Keep original X for wrap detection
    origY: p.y,
    localX: p.x * scaleX, // Local coordinates relative to displayed image (for point-in-polygon)
    localY: p.y * scaleY
  }));
  
  ctx.save();
  ctx.strokeStyle = "rgba(0, 200, 255, 0.8)";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  
  // Draw outline, skipping segments that wrap around the edge
  // A segment wraps if the X difference is more than half the image width
  const wrapThreshold = imageWidth * 0.5;
  
  ctx.beginPath();
  let lastValidPoint = null;
  
  for (let i = 0; i < previewPoints.length; i++) {
    const point = previewPoints[i];
    
    if (lastValidPoint === null) {
      // First point - start the path
      ctx.moveTo(point.x, point.y);
      lastValidPoint = point;
    } else {
      // Check if this segment wraps around the image edge
      const dx = Math.abs(point.origX - lastValidPoint.origX);
      
      if (dx > wrapThreshold) {
        // This segment wraps around - don't draw it, start a new path segment
        ctx.stroke(); // Draw current path
        ctx.beginPath(); // Start new path
        ctx.moveTo(point.x, point.y);
        lastValidPoint = point;
      } else {
        // Normal segment - continue the line
        ctx.lineTo(point.x, point.y);
        lastValidPoint = point;
      }
    }
  }
  
  // Don't close the path if it wraps - just stroke what we have
  ctx.stroke();
  
  // Draw corner handles
  const cornerPoints = getCornerPoints(cropInfo, imageWidth, imageHeight);
  if (cornerPoints) {
    const handleSize = 14; // Increased from 8 for easier clicking
    ctx.fillStyle = "#00ccff";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    
    for (const corner of cornerPoints) {
      // Use displayed image coordinates (maintains aspect ratio)
      const x = corner.x * scaleX + previewArea.imageX;
      const y = corner.y * scaleY + previewArea.imageY;
      
      ctx.beginPath();
      ctx.arc(x, y, handleSize / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
  
  ctx.restore();
  
  // Store preview points for hit detection
  return previewPoints;
}

// Check if a point is near a corner handle
function getCornerHit(cornerPoints, previewArea, imageWidth, imageHeight, localX, localY) {
  if (!cornerPoints || cornerPoints.length !== 4) return null;
  
  // Use displayed image size for coordinate conversion (maintains aspect ratio)
  const scaleX = previewArea.imageWidth / imageWidth;
  const scaleY = previewArea.imageHeight / imageHeight;
  const handleSize = 14; // Match the handle size for hit detection
  const handleRadius = handleSize / 2 + 2; // Slightly larger than visual handle for easier clicking
  
  // localX and localY are already relative to displayed image (from getPreviewLocalPos)
  // So we don't need to add previewArea.imageX/imageY
  
  const cornerNames = ["topLeft", "topRight", "bottomRight", "bottomLeft"];
  
  for (let i = 0; i < cornerPoints.length; i++) {
    const corner = cornerPoints[i];
    // Convert corner from equirectangular coordinates to displayed image coordinates
    const x = corner.x * scaleX;
    const y = corner.y * scaleY;
    
    const dx = localX - x;
    const dy = localY - y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    if (dist <= handleRadius) {
      return { index: i, name: cornerNames[i] };
    }
  }
  
  return null;
}

app.registerExtension({
  name: "oa.360clip",
  
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "OA360Clip") return;
    
    function initNodeState(node) {
      node.serialize_widgets = true;
      node.properties = node.properties || {};
      
      if (!node.image) {
        node.image = new Image();
        node.image.src = "";
        node.imageLoaded = false;
      }
      
      if (node.dragging === undefined) node.dragging = false;
      if (node.dragStartPos === undefined) node.dragStartPos = null;
      if (node.dragMode === undefined) node.dragMode = null;
      if (node.originalDragStart === undefined) node.originalDragStart = null;
      if (node.originalDragEnd === undefined) node.originalDragEnd = null;
      
          const defaults = {
            // Don't set crop values until image is loaded
            output_width: 1280,
            output_height: 720,
            actualImageWidth: 0,
            actualImageHeight: 0,
          };
      
      Object.keys(defaults).forEach((key) => {
        if (node.properties[key] === undefined) {
          node.properties[key] = defaults[key];
        }
      });
      
      // Clear invalid crop values if image isn't loaded yet
      // This prevents old/invalid values from a saved workflow from being serialized
      if (!node.properties.actualImageWidth || !node.properties.actualImageHeight ||
          node.properties.actualImageWidth === 0 || node.properties.actualImageHeight === 0) {
        // Clear crop values if image dimensions are invalid
        if (node.properties.crop_center_x !== undefined || node.properties.crop_center_y !== undefined ||
            node.properties.crop_width !== undefined || node.properties.crop_height !== undefined) {
          // Check if values are clearly invalid (way too large)
          const imgWidth = node.properties.actualImageWidth || 0;
          const imgHeight = node.properties.actualImageHeight || 0;
          const crop_center_x = node.properties.crop_center_x;
          const crop_width = node.properties.crop_width;
          
          // If image isn't loaded or values are clearly invalid, clear them
          if (imgWidth === 0 || imgHeight === 0 || 
              (crop_center_x !== undefined && crop_center_x > imgWidth * 2 && imgWidth > 0) ||
              (crop_width !== undefined && crop_width > imgWidth * 0.5 && imgWidth > 0)) {
            delete node.properties.crop_center_x;
            delete node.properties.crop_center_y;
            delete node.properties.crop_width;
            delete node.properties.crop_height;
          }
        }
      }
    }
    
    const originalNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (originalNodeCreated) {
        originalNodeCreated.apply(this, arguments);
      }
      initNodeState(this);
      
      // Explicitly allow the node to be resized
      this.resizable = true;
      
      // Initialize image object for preview if not already initialized
      if (!this.image) {
        this.image = new Image();
        this.image.src = "";
        this.imageLoaded = false;
      } else {
        // Ensure imageLoaded is false if image exists but hasn't loaded
        this.imageLoaded = false;
      }
      
      
      // Force immediate redraw after a short delay to ensure node is fully initialized
      setTimeout(() => {
        if (this.setDirtyCanvas) {
          this.setDirtyCanvas(true);
        }
        if (this.graph && this.graph.setDirtyCanvas) {
          this.graph.setDirtyCanvas(true);
        }
      }, 50);
    };
    
    const originalOnAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      if (originalOnAdded) {
        originalOnAdded.apply(this, arguments);
      }
      
      // Only set initial size if node doesn't already have a size
      // This prevents overriding a size that was set by user resize
      if (!this.size || this.size[0] === 0 || this.size[1] === 0) {
        const computedSize = this.computeSize();
        if (computedSize && computedSize[0] > 0 && computedSize[1] > 0) {
          this.size = computedSize;
        } else {
          this.size = [320, 400];
        }
      }
      
      // Ensure node is not collapsed
      if (this.flags) {
        this.flags.collapsed = false;
      }
      
      // Force immediate redraw and also schedule one for next frame
      this.setDirtyCanvas(true);
      if (this.graph) {
        this.graph.setDirtyCanvas(true);
      }
      
      // Also schedule a redraw for next frame to ensure it shows
      requestAnimationFrame(() => {
        this.setDirtyCanvas(true);
        if (this.graph) {
          this.graph.setDirtyCanvas(true);
        }
      });
    };
    
    const originalComputeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function (out) {
      // Let ComfyUI handle resizing - we just provide a default size if needed
      // If out is provided (user is resizing), accept it
      if (out && Array.isArray(out) && out.length >= 2) {
        return out;
      }
      
      // If node already has a size, return it (let user's resize persist)
      if (this.size && Array.isArray(this.size) && this.size.length >= 2 && this.size[0] > 0 && this.size[1] > 0) {
        return this.size;
      }
      
      // Only compute a default size if node has no size yet
      const titleBarHeight = 30;
      const widgetHeight = (this.widgets?.filter((w) => !w.hidden) || []).length * 25;
      const padding = 10;
      const bottomPadding = 15;
      const minWidth = 320;
      const minPreviewHeight = 100;
      const minHeight = titleBarHeight + widgetHeight + padding + minPreviewHeight + bottomPadding;
      
      return [minWidth, minHeight];
    };
    
    nodeType.prototype.onDrawForeground = function (ctx) {
      // Only fix size if it's completely invalid (0 or undefined)
      // Otherwise, let ComfyUI handle the size - our content will scale to fit
      if (!this.size || this.size[0] === 0 || this.size[1] === 0) {
        const computedSize = this.computeSize();
        if (computedSize && computedSize[0] > 0 && computedSize[1] > 0) {
          this.size = computedSize;
        } else {
          this.size = [320, 400];
        }
      }
      
      // Ensure position is valid
      if (!this.pos || this.pos[0] === undefined || this.pos[1] === undefined) {
        return;
      }
      
      if (this.flags?.collapsed) {
        return;
      }
      
      // Calculate preview area - always ensure it's valid
      // IMPORTANT: In onDrawForeground, coordinates are relative to the node (0,0 is top-left of node)
      // NOT absolute canvas coordinates
      const widgetHeight = (this.widgets?.filter((w) => !w.hidden) || []).length * 25;
      const titleBarHeight = 30; // Standard ComfyUI title bar height
      const padding = 10;
      const bottomPadding = 15; // Extra padding at bottom to avoid touching edge
      const previewY = titleBarHeight + widgetHeight + padding; // Relative to node top
      
      // Draw input image preview with crop box
      // Check if image is loaded and complete FIRST
      // IMPORTANT: Check imageLoaded flag first, as image object might exist but not be loaded
      const hasImage = this.imageLoaded && this.image && this.image.complete && this.image.naturalWidth > 0 && this.image.naturalHeight > 0;
      
      // Calculate available space - content scales to fit, node size stays fixed
      const availableWidth = Math.max(160, (this.size[0] || 320) - 40);
      const availableHeight = (this.size[1] || 400) - previewY - padding - bottomPadding;
      
      // Calculate preview area (background container)
      const previewAreaHeight = Math.max(100, availableHeight);
      const previewAreaWidth = availableWidth;
      
      // Calculate actual displayed image size maintaining aspect ratio
      let displayedImageWidth = previewAreaWidth;
      let displayedImageHeight = previewAreaHeight;
      let displayedImageX = ((this.size[0] || 320) - previewAreaWidth) / 2;
      let displayedImageY = previewY;
      
      if (hasImage && this.properties.actualImageWidth && this.properties.actualImageHeight) {
        const imageAspect = this.properties.actualImageWidth / this.properties.actualImageHeight;
        const availableAspect = previewAreaWidth / previewAreaHeight;
        
        if (imageAspect > availableAspect) {
          // Image is wider - fit to width
          displayedImageWidth = previewAreaWidth;
          displayedImageHeight = previewAreaWidth / imageAspect;
        } else {
          // Image is taller - fit to height
          displayedImageHeight = previewAreaHeight;
          displayedImageWidth = previewAreaHeight * imageAspect;
        }
        
        // Center the displayed image within the preview area
        displayedImageX = ((this.size[0] || 320) - displayedImageWidth) / 2;
        displayedImageY = previewY + (previewAreaHeight - displayedImageHeight) / 2;
      }
      
      // Preview area (background container) - centered horizontally
      const preview = {
        x: ((this.size[0] || 320) - previewAreaWidth) / 2, // Centered horizontally
        y: previewY, // Relative to node top edge
        width: previewAreaWidth,
        height: previewAreaHeight,
        // Actual displayed image position and size (maintains aspect ratio)
        imageX: displayedImageX,
        imageY: displayedImageY,
        imageWidth: displayedImageWidth,
        imageHeight: displayedImageHeight
      };
      
      // Always draw the preview area background and border first
      // Make sure we don't draw over widgets - clip to preview area
      ctx.save();
      ctx.beginPath();
      ctx.rect(preview.x, preview.y, preview.width, preview.height);
      ctx.clip();
      ctx.fillStyle = "#222";
      ctx.fillRect(preview.x, preview.y, preview.width, preview.height);
      ctx.strokeStyle = "#555";
      ctx.lineWidth = 1;
      ctx.strokeRect(preview.x, preview.y, preview.width, preview.height);
      ctx.restore(); // This restores the clipping region
      
      if (hasImage) {
        ctx.save();
        // Draw image maintaining aspect ratio
        ctx.drawImage(this.image, preview.imageX, preview.imageY, preview.imageWidth, preview.imageHeight);
        ctx.restore();
        
        // Draw crop box
        const cropInfo = {
          center_x: this.properties.crop_center_x || 640,
          center_y: this.properties.crop_center_y || 360,
          width: this.properties.crop_width || 512,
          height: this.properties.crop_height || 512,
          output_width: this.properties.output_width || 1280,
          output_height: this.properties.output_height || 720,
        };
        
        if (this.properties.actualImageWidth && this.properties.actualImageHeight) {
          // Use displayed image size for drawing (maintains aspect ratio)
          // Pass preview object directly - it already has imageX, imageY, imageWidth, imageHeight
          const outlinePoints = drawProjectionOutline(ctx, cropInfo, preview, this.properties.actualImageWidth, this.properties.actualImageHeight);
          // Store outline points for hit detection - convert to local coordinates
          // outlinePoints are in preview coordinates (x, y relative to node), convert to local (relative to displayed image)
          this._outlinePoints = outlinePoints.map(p => ({
            x: p.localX || (p.x - preview.imageX),
            y: p.localY || (p.y - preview.imageY)
          }));
          this._cornerPoints = getCornerPoints(cropInfo, this.properties.actualImageWidth, this.properties.actualImageHeight);
        }
      } else {
        // Show message when image not loaded - ALWAYS draw this
        ctx.save();
        ctx.fillStyle = "#ffffff"; // Explicit white color
        ctx.font = "bold 14px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const msg = "Run with input to show image, then drag clip region to desired point";
        const lines = msg.split(", ");
        const centerY = preview.y + preview.height / 2;
        const lineHeight = 20;
        const totalHeight = (lines.length - 1) * lineHeight;
        const startY = centerY - totalHeight / 2;
        
        // Draw text with shadow for better visibility
        ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        
        lines.forEach((line, i) => {
          const y = startY + i * lineHeight;
          const x = preview.x + preview.width / 2;
          ctx.fillText(line.trim(), x, y);
        });
        
        // Reset shadow
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.restore();
      }
    };
    
    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      if (originalOnExecuted) {
        originalOnExecuted.apply(this, arguments);
      }
      
      // Handle input image
      const imageInfo = message?.images_custom?.[0];
      if (!imageInfo) {
        // No image - clear and show message
        this.image.src = "";
        this.imageLoaded = false;
        this.properties.actualImageWidth = 0;
        this.properties.actualImageHeight = 0;
        this.setDirtyCanvas(true);
      } else {
        // Load input image
        const imageUrl = app.api.apiURL(
          `/view?filename=${imageInfo.filename}&type=${imageInfo.type || "temp"}&subfolder=${imageInfo.subfolder || ""}&rand=${Date.now()}`
        );
        
        // Don't clear the image immediately - keep the old one visible while new one loads
        // Only set imageLoaded to false after we start loading the new image
        const wasLoaded = this.imageLoaded;
        
        // Create a new image object for loading, so we don't clear the current one
        const newImage = new Image();
        newImage.onload = () => {
            // Swap to the new image only after it's loaded
            this.image = newImage;
            this.imageLoaded = true;
            const newWidth = this.image.naturalWidth;
            const newHeight = this.image.naturalHeight;

            this.properties.actualImageWidth = newWidth;
            this.properties.actualImageHeight = newHeight;

            // Validate and reset crop region if invalid
            const maxCropWidth = newWidth * 0.25;
            const currentCropCenterX = this.properties.crop_center_x;
            const currentCropCenterY = this.properties.crop_center_y;
            const currentCropWidth = this.properties.crop_width;
            const currentCropHeight = this.properties.crop_height;
            
            // Check if current crop values are valid for this image
            // Normalize crop_center_x first to check if it's reasonable
            const normalizedCenterX = currentCropCenterX !== undefined ? 
              ((currentCropCenterX % newWidth) + newWidth) % newWidth : undefined;
            const isValid = 
              currentCropCenterX !== undefined && currentCropCenterY !== undefined &&
              currentCropWidth !== undefined && currentCropHeight !== undefined &&
              normalizedCenterX !== undefined && normalizedCenterX >= 0 && normalizedCenterX < newWidth &&
              currentCropCenterY >= 0 && currentCropCenterY <= newHeight &&
              currentCropWidth > 0 && currentCropWidth <= maxCropWidth &&
              currentCropHeight > 0 && currentCropHeight <= newHeight;
            
            // Initialize or reset crop region if not set or invalid
            if (!isValid || !this.properties.crop_center_x || !this.properties.crop_center_y) {
              this.properties.crop_center_x = newWidth / 2;
              this.properties.crop_center_y = newHeight / 2;
            } else if (normalizedCenterX !== undefined && normalizedCenterX !== currentCropCenterX) {
              // Normalize crop_center_x if it's out of bounds (wrapped)
              this.properties.crop_center_x = normalizedCenterX;
            }
            if (!isValid || !this.properties.crop_width || !this.properties.crop_height) {
              this.properties.crop_width = Math.min(512, maxCropWidth);
              this.properties.crop_height = Math.min(512, newHeight);
            } else {
              // Ensure crop width doesn't exceed max (25% of image width)
              if (this.properties.crop_width > maxCropWidth) {
                this.properties.crop_width = maxCropWidth;
              }
              // Ensure crop height doesn't exceed image height
              if (this.properties.crop_height > newHeight) {
                this.properties.crop_height = newHeight;
              }
            }
          
          // Don't resize the node - it should stay at its current size
          // Content will scale to fit available space
          
          this.setDirtyCanvas(true);
        };
        
        newImage.onerror = () => {
          // If new image fails to load, keep the old one if it exists
          if (!wasLoaded) {
            this.imageLoaded = false;
          }
          console.warn("[OA360Clip] Image failed to load");
          this.setDirtyCanvas(true);
        };

        // Start loading the new image
        newImage.src = imageUrl;
        // Don't set imageLoaded to false here - keep showing the old image until new one loads
        this.setDirtyCanvas(true);
      }
      
          // Update crop info from backend
      // IMPORTANT: Never update crop values from backend while dragging - they are always controlled by user interaction
      // Only update when not dragging to avoid race conditions
      if (message && message.crop_info && message.crop_info.length > 0 && !this.dragging) {
        const cropInfo = message.crop_info[0];
        if (cropInfo.original_size) {
          this.properties.actualImageWidth = cropInfo.original_size[0];
          this.properties.actualImageHeight = cropInfo.original_size[1];
        }
        
        // NEVER update crop values from backend - they are always controlled by user interaction
        // The backend should use the values we send via serialize(), not overwrite them
        // Only update image dimensions and output dimensions from backend
        
        if (cropInfo.output_width !== undefined) {
          this.properties.output_width = cropInfo.output_width;
          const outputWidthWidget = this.widgets?.find(w => w.name === "output_width");
          if (outputWidthWidget) outputWidthWidget.value = cropInfo.output_width;
        }
        if (cropInfo.output_height !== undefined) {
          this.properties.output_height = cropInfo.output_height;
          const outputHeightWidget = this.widgets?.find(w => w.name === "output_height");
          if (outputHeightWidget) outputHeightWidget.value = cropInfo.output_height;
        }
        this.setDirtyCanvas(true);
      }
    };
    
    // Serialize crop properties to send to backend via extra_data
    const originalSerialize = nodeType.prototype.serialize;
    nodeType.prototype.serialize = function () {
      const data = originalSerialize ? originalSerialize.apply(this, arguments) : {};
      data.extra = data.extra || {};
      
      // Only send crop properties if image is loaded and values are valid
      const imgWidth = this.properties.actualImageWidth || 0;
      const imgHeight = this.properties.actualImageHeight || 0;
      const hasValidImage = imgWidth > 0 && imgHeight > 0;
      
      // Don't serialize crop values if image isn't loaded - this prevents invalid values from being saved
      if (!hasValidImage) {
        return data;
      }
      
      // Validate crop values before sending
      const crop_center_x = this.properties.crop_center_x;
      const crop_center_y = this.properties.crop_center_y;
      const crop_width = this.properties.crop_width;
      const crop_height = this.properties.crop_height;
      
      // Only send if all values are defined and reasonable
      // Allow crop_center_x to wrap, but normalize it
      // But crop_width should never exceed 25% of image width
      const maxCropWidth = imgWidth * 0.25;
      if (crop_center_x !== undefined && crop_center_y !== undefined &&
          crop_width !== undefined && crop_height !== undefined &&
          crop_center_x >= 0 && // Allow wrapping, but we'll normalize it
          crop_center_y >= 0 && crop_center_y <= imgHeight &&
          crop_width > 0 && crop_width <= maxCropWidth &&
          crop_height > 0 && crop_height <= imgHeight) {
        // Normalize crop_center_x to [0, imgWidth) range for storage
        // This handles wrapping correctly
        const normalizedCenterX = ((crop_center_x % imgWidth) + imgWidth) % imgWidth;
        data.extra.crop_center_x = normalizedCenterX;
        data.extra.crop_center_y = crop_center_y;
        data.extra.crop_width = crop_width;
        data.extra.crop_height = crop_height;
        
        // Debug logging removed - serialize() is called frequently by ComfyUI for auto-save
        // This is normal behavior, no need to log every call
      }
      
      // Also send crop data to backend via API endpoint for immediate access
      // This ensures the backend has the values before execution
      // Only send if we have valid crop values (same validation as above)
      if (this.id && hasValidImage && 
          crop_center_x !== undefined && crop_center_y !== undefined &&
          crop_width !== undefined && crop_height !== undefined &&
          crop_center_x >= 0 && crop_center_x < imgWidth * 10 &&
          crop_center_y >= 0 && crop_center_y <= imgHeight &&
          crop_width > 0 && crop_width <= imgWidth * 0.25 &&
          crop_height > 0 && crop_height <= imgHeight) {
        // Normalize crop_center_x for API call too
        const normalizedCenterX = ((crop_center_x % imgWidth) + imgWidth) % imgWidth;
        fetch(app.api.apiURL("/oa360clip/set_crop"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            node_id: this.id,
            crop_center_x: normalizedCenterX,
            crop_center_y: crop_center_y,
            crop_width: crop_width,
            crop_height: crop_height,
          })
        }).catch(err => console.warn("[OA360Clip] Failed to send crop data:", err));
      }
      
      return data;
    };
    
    // Handle widget changes for output dimensions
    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      if (originalOnConfigure) {
        originalOnConfigure.apply(this, arguments);
      }
      
      const outputWidthWidget = this.widgets?.find(w => w.name === "output_width");
      const outputHeightWidget = this.widgets?.find(w => w.name === "output_height");
      
      if (outputWidthWidget) {
        const originalCallback = outputWidthWidget.callback;
        outputWidthWidget.callback = (value) => {
          this.properties.output_width = value;
          if (originalCallback) originalCallback(value);
          this.setDirtyCanvas(true);
          this.graph.setDirtyCanvas(true);
        };
      }
      
      if (outputHeightWidget) {
        const originalCallback = outputHeightWidget.callback;
        outputHeightWidget.callback = (value) => {
          this.properties.output_height = value;
          if (originalCallback) originalCallback(value);
          this.setDirtyCanvas(true);
          this.graph.setDirtyCanvas(true);
        };
      }
    };
    
    // Handle mouse events for dragging crop outline
    const originalOnMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (e, pos, canvas) {
      if (originalOnMouseDown) {
        const result = originalOnMouseDown.apply(this, arguments);
        if (result) return result;
      }
      
      if (this.flags?.collapsed) return false;
      
      const preview = getPreviewArea(this);
      if (!preview) return false;
      
      const local = getPreviewLocalPos(this.pos, [e.canvasX, e.canvasY], preview);
      
          // Check if click is within the displayed image area (maintains aspect ratio)
          const isInPreviewArea = local.x >= 0 && local.y >= 0 && local.x < preview.imageWidth && local.y < preview.imageHeight;
      
      if (isInPreviewArea) {
        // Use displayed image size for coordinate conversion (maintains aspect ratio)
        const scaleX = (this.properties.actualImageWidth || 1) / preview.imageWidth;
        const scaleY = (this.properties.actualImageHeight || 1) / preview.imageHeight;
        
        // Check if clicking on a corner handle
        const cornerHit = getCornerHit(
          this._cornerPoints,
          preview,
          this.properties.actualImageWidth || 1,
          this.properties.actualImageHeight || 1,
          local.x,
          local.y
        );
        
        if (cornerHit) {
          // Dragging a corner to resize
          this.dragging = true;
          this.dragMode = "corner";
          this.dragCornerIndex = cornerHit.index;
          this.dragStartPos = [e.canvasX, e.canvasY];
          const cropWidth = this.properties.crop_width || 512;
          const cropHeight = this.properties.crop_height || 512;
          const centerX = this.properties.crop_center_x || 640;
          const centerY = this.properties.crop_center_y || 360;
          this.originalCropLeft = centerX - cropWidth / 2;
          this.originalCropTop = centerY - cropHeight / 2;
          this.originalCropWidth = cropWidth;
          this.originalCropHeight = cropHeight;
          return true;
        }
        
        // Check if clicking inside the outline (for dragging the crop region)
        if (this._outlinePoints && this._outlinePoints.length > 0) {
          // Convert local coordinates to preview coordinates for point-in-polygon test
          const point = { x: local.x, y: local.y };
          if (pointInPolygon(point, this._outlinePoints)) {
            // Dragging the entire crop region
            this.dragging = true;
            this.dragMode = "move";
            this.dragStartPos = [e.canvasX, e.canvasY];
            this.dragStartLocal = { x: local.x, y: local.y };
            const cropWidth = this.properties.crop_width || 512;
            const cropHeight = this.properties.crop_height || 512;
            const centerX = this.properties.crop_center_x || 640;
            const centerY = this.properties.crop_center_y || 360;
            // Store original values for relative movement calculation
            this.originalCropCenterX = centerX;
            this.originalCropCenterY = centerY;
            this.originalCropLeft = centerX - cropWidth / 2;
            this.originalCropTop = centerY - cropHeight / 2;
            this.originalCropWidth = cropWidth;
            this.originalCropHeight = cropHeight;
            return true;
          }
        }
        
        // Clicking outside the outline but inside preview area - jump crop region to mouse position
        // and start dragging so user can continue dragging
        // Convert mouse position to equirectangular coordinates and set as new center
        // Note: scaleX and scaleY are already defined above, reuse them
        
        const imgWidth = this.properties.actualImageWidth || 0;
        const imgHeight = this.properties.actualImageHeight || 0;
        
        // Don't allow interaction if image isn't loaded yet
        if (!imgWidth || !imgHeight) {
          return false;
        }
        
        // Convert local preview coordinates to equirectangular coordinates
        let newCenterX = local.x * scaleX;
        let newCenterY = local.y * scaleY;
        
        // Handle wrapping for X (equirectangular wraps horizontally)
        while (newCenterX < 0) newCenterX += imgWidth;
        while (newCenterX >= imgWidth) newCenterX -= imgWidth;
        
        // Clamp Y to image bounds
        newCenterY = Math.max(0, Math.min(imgHeight, newCenterY));
        
        // Update crop center to mouse position
        this.properties.crop_center_x = newCenterX;
        this.properties.crop_center_y = newCenterY;
        
        // Start dragging mode so user can continue dragging
        this.dragging = true;
        this.dragMode = "move";
        this.dragStartPos = [e.canvasX, e.canvasY];
        this.dragStartLocal = { x: local.x, y: local.y };
        const cropWidth = this.properties.crop_width || 512;
        const cropHeight = this.properties.crop_height || 512;
        // Store original center for relative movement calculation
        this.originalCropCenterX = newCenterX;
        this.originalCropCenterY = newCenterY;
        this.originalCropLeft = newCenterX - cropWidth / 2;
        this.originalCropTop = newCenterY - cropHeight / 2;
        
        // Send updated crop data to backend immediately
        if (this.id) {
          fetch(app.api.apiURL("/oa360clip/set_crop"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              node_id: this.id,
              crop_center_x: this.properties.crop_center_x,
              crop_center_y: this.properties.crop_center_y,
              crop_width: this.properties.crop_width,
              crop_height: this.properties.crop_height,
            })
          }).catch(() => {});
        }
        
        // Mark node as changed to trigger re-execution
        this.setDirtyCanvas(true);
        if (this.graph) {
          this.graph.setDirtyCanvas(true);
          if (this.graph.setDirty) {
            this.graph.setDirty(true);
          }
        }
        
        return true; // Consume the event to prevent node dragging
      }
      
      return false;
    };
    
    const originalOnMouseMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (e, pos, canvas) {
      if (originalOnMouseMove) {
        originalOnMouseMove.apply(this, arguments);
      }
      
      // If we were dragging but mouse button is now up, clear dragging state
      if (this.dragging && e.buttons !== 1) {
        this.dragging = false;
        this.dragMode = null;
        this.dragCornerIndex = undefined;
        this.dragStartPos = null;
        this.originalCropLeft = null;
        this.originalCropTop = null;
        this.originalCropWidth = null;
        this.originalCropHeight = null;
        this.originalCropCenterX = null;
        this.originalCropCenterY = null;
        this.dragStartLocal = null;
        return false;
      }
      
      // Only process dragging if mouse button is still down (buttons === 1 means left button)
      if (this.dragging && this.dragStartPos && e.buttons === 1) {
        const preview = getPreviewArea(this);
        const local = getPreviewLocalPos(this.pos, [e.canvasX, e.canvasY], preview);
        
        const imgWidth = this.properties.actualImageWidth || 0;
        const imgHeight = this.properties.actualImageHeight || 0;
        
        // Don't allow dragging if image isn't loaded yet
        if (!imgWidth || !imgHeight || !preview.imageWidth || !preview.imageHeight) {
          return false;
        }
        
        // Use displayed image size for coordinate conversion (maintains aspect ratio)
        const scaleX = imgWidth / preview.imageWidth;
        const scaleY = imgHeight / preview.imageHeight;
        
        const dx = (e.canvasX - this.dragStartPos[0]) * scaleX;
        const dy = (e.canvasY - this.dragStartPos[1]) * scaleY;
        
        if (this.dragMode === "move") {
          // Move the entire crop region
          // Center point can wrap around horizontally (equirectangular wraps)
          const cropWidth = this.properties.crop_width || 512;
          const cropHeight = this.properties.crop_height || 512;
          
          // Allow X to wrap around (equirectangular images wrap horizontally)
          let newCenterX = this.originalCropLeft + cropWidth / 2 + dx;
          // Wrap around if it goes outside bounds
          while (newCenterX < 0) newCenterX += imgWidth;
          while (newCenterX >= imgWidth) newCenterX -= imgWidth;
          
          // Y doesn't wrap (latitude), but allow it to go near edges
          const newCenterY = Math.max(0, Math.min(
            imgHeight,
            this.originalCropTop + cropHeight / 2 + dy
          ));
          
          this.properties.crop_center_x = newCenterX;
          this.properties.crop_center_y = newCenterY;
        } else if (this.dragMode === "corner" && this.dragCornerIndex !== undefined) {
          // Resize by dragging a corner - MUST maintain output aspect ratio
          const imgWidth = this.properties.actualImageWidth || 0;
          const imgHeight = this.properties.actualImageHeight || 0;
          
          // Don't allow corner dragging if image isn't loaded yet
          if (!imgWidth || !imgHeight) {
            return false;
          }
          const minSize = 32; // Minimum crop size
          const maxWidth = imgWidth * 0.25; // Maximum crop width is 25% of image width
          
          // Get output aspect ratio
          const outputWidth = this.properties.output_width || 1280;
          const outputHeight = this.properties.output_height || 720;
          const outputAspect = outputWidth / outputHeight;
          
          // Get the new corner position in displayed image coordinates
          // local.x and local.y are already relative to displayed image (from getPreviewLocalPos)
          // Convert to equirectangular coordinates using displayed image size
          const newCornerEqX = local.x / (preview.imageWidth / imgWidth);
          const newCornerEqY = local.y / (preview.imageHeight / imgHeight);
          
          // Calculate new crop region based on which corner is being dragged
          // We'll adjust the crop region to keep the opposite corner fixed AND maintain aspect ratio
          let newLeft = this.originalCropLeft;
          let newTop = this.originalCropTop;
          let newWidth = this.originalCropWidth;
          let newHeight = this.originalCropHeight;
          
          if (this.dragCornerIndex === 0) {
            // Top-left corner: adjust left and top, keep bottom-right fixed
            const fixedX = this.originalCropLeft + this.originalCropWidth;
            const fixedY = this.originalCropTop + this.originalCropHeight;
            
            // Calculate distance from fixed corner
            const dx = fixedX - newCornerEqX;
            const dy = fixedY - newCornerEqY;
            
            // Calculate size maintaining aspect ratio
            if (Math.abs(dx) > Math.abs(dy * outputAspect)) {
              // Width is the limiting factor
              newWidth = Math.max(minSize, Math.min(dx, maxWidth));
              newHeight = newWidth / outputAspect;
            } else {
              // Height is the limiting factor
              newHeight = Math.max(minSize, Math.min(dy, imgHeight));
              newWidth = newHeight * outputAspect;
              // Ensure width doesn't exceed max
              if (newWidth > maxWidth) {
                newWidth = maxWidth;
                newHeight = newWidth / outputAspect;
              }
            }
            
            newLeft = fixedX - newWidth;
            newTop = fixedY - newHeight;
          } else if (this.dragCornerIndex === 1) {
            // Top-right corner: adjust top and width, keep bottom-left fixed
            const fixedX = this.originalCropLeft;
            const fixedY = this.originalCropTop + this.originalCropHeight;
            
            const dx = newCornerEqX - fixedX;
            const dy = fixedY - newCornerEqY;
            
            if (Math.abs(dx) > Math.abs(dy * outputAspect)) {
              newWidth = Math.max(minSize, Math.min(dx, Math.min(imgWidth - fixedX, maxWidth)));
              newHeight = newWidth / outputAspect;
            } else {
              newHeight = Math.max(minSize, Math.min(dy, fixedY));
              newWidth = newHeight * outputAspect;
              // Ensure width doesn't exceed max
              if (newWidth > maxWidth) {
                newWidth = maxWidth;
                newHeight = newWidth / outputAspect;
              }
            }
            
            newLeft = fixedX;
            newTop = fixedY - newHeight;
          } else if (this.dragCornerIndex === 2) {
            // Bottom-right corner: adjust width and height, keep top-left fixed
            const fixedX = this.originalCropLeft;
            const fixedY = this.originalCropTop;
            
            const dx = newCornerEqX - fixedX;
            const dy = newCornerEqY - fixedY;
            
            if (Math.abs(dx) > Math.abs(dy * outputAspect)) {
              newWidth = Math.max(minSize, Math.min(dx, Math.min(imgWidth - fixedX, maxWidth)));
              newHeight = newWidth / outputAspect;
            } else {
              newHeight = Math.max(minSize, Math.min(dy, imgHeight - fixedY));
              newWidth = newHeight * outputAspect;
              // Ensure width doesn't exceed max
              if (newWidth > maxWidth) {
                newWidth = maxWidth;
                newHeight = newWidth / outputAspect;
              }
            }
            
            newLeft = fixedX;
            newTop = fixedY;
          } else if (this.dragCornerIndex === 3) {
            // Bottom-left corner: adjust left and height, keep top-right fixed
            const fixedX = this.originalCropLeft + this.originalCropWidth;
            const fixedY = this.originalCropTop;
            
            const dx = fixedX - newCornerEqX;
            const dy = newCornerEqY - fixedY;
            
            if (Math.abs(dx) > Math.abs(dy * outputAspect)) {
              newWidth = Math.max(minSize, Math.min(dx, Math.min(fixedX, maxWidth)));
              newHeight = newWidth / outputAspect;
            } else {
              newHeight = Math.max(minSize, Math.min(dy, imgHeight - fixedY));
              newWidth = newHeight * outputAspect;
              // Ensure width doesn't exceed max
              if (newWidth > maxWidth) {
                newWidth = maxWidth;
                newHeight = newWidth / outputAspect;
              }
            }
            
            newLeft = fixedX - newWidth;
            newTop = fixedY;
          }
          
          // Clamp to image bounds
          newLeft = Math.max(0, Math.min(newLeft, imgWidth - minSize));
          newTop = Math.max(0, Math.min(newTop, imgHeight - minSize));
          
          // Enforce maximum width constraint (25% of image width)
          if (newWidth > maxWidth) {
            newWidth = maxWidth;
            newHeight = newWidth / outputAspect;
          }
          
          if (newLeft + newWidth > imgWidth) {
            newWidth = imgWidth - newLeft;
            newHeight = newWidth / outputAspect;
            // Re-check max width after clamping
            if (newWidth > maxWidth) {
              newWidth = maxWidth;
              newHeight = newWidth / outputAspect;
            }
          }
          if (newTop + newHeight > imgHeight) {
            newHeight = imgHeight - newTop;
            newWidth = newHeight * outputAspect;
            // Re-check max width after clamping
            if (newWidth > maxWidth) {
              newWidth = maxWidth;
              newHeight = newWidth / outputAspect;
            }
          }
          
          // Ensure minimum size
          if (newWidth < minSize) {
            newWidth = minSize;
            newHeight = newWidth / outputAspect;
          }
          if (newHeight < minSize) {
            newHeight = minSize;
            newWidth = newHeight * outputAspect;
            // Re-check max width after ensuring minimum
            if (newWidth > maxWidth) {
              newWidth = maxWidth;
              newHeight = newWidth / outputAspect;
            }
          }
          
            // Update center and dimensions
            this.properties.crop_center_x = newLeft + newWidth / 2;
            this.properties.crop_center_y = newTop + newHeight / 2;
            this.properties.crop_width = newWidth;
            this.properties.crop_height = newHeight;
          
          // Don't send API calls during dragging - only update on drag end to avoid race conditions
          // Mark node as changed to trigger re-execution
          this.setDirtyCanvas(true);
          if (this.graph) {
            this.graph.setDirtyCanvas(true);
            // Trigger workflow change to force re-execution
            if (this.graph.setDirty) {
              this.graph.setDirty(true);
            }
          }
        }
        
        this.setDirtyCanvas(true);
        this.graph.setDirtyCanvas(true);
        return true;
      }
      
      return false;
    };
    
    const originalOnMouseUp = nodeType.prototype.onMouseUp;
    nodeType.prototype.onMouseUp = function (e, pos, canvas) {
      if (originalOnMouseUp) {
        originalOnMouseUp.apply(this, arguments);
      }
      
      if (this.dragging) {
        // Send crop data to backend when drag ends (not during dragging to avoid race conditions)
        const imgWidth = this.properties.actualImageWidth || 0;
        const imgHeight = this.properties.actualImageHeight || 0;
        if (this.id && imgWidth > 0 && imgHeight > 0 && (
          this.properties.crop_center_x !== undefined ||
          this.properties.crop_center_y !== undefined ||
          this.properties.crop_width !== undefined ||
          this.properties.crop_height !== undefined
        )) {
          // Normalize crop_center_x for API call
          const crop_center_x = this.properties.crop_center_x;
          let normalizedCenterX = crop_center_x;
          if (crop_center_x !== undefined && imgWidth > 0) {
            normalizedCenterX = ((crop_center_x % imgWidth) + imgWidth) % imgWidth;
          }
          
          fetch(app.api.apiURL("/oa360clip/set_crop"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              node_id: this.id,
              crop_center_x: normalizedCenterX,
              crop_center_y: this.properties.crop_center_y,
              crop_width: this.properties.crop_width,
              crop_height: this.properties.crop_height,
            })
          }).catch(() => {});
        }
        
        this.dragging = false;
        this.dragMode = null;
        this.dragCornerIndex = undefined;
        this.dragStartPos = null;
        this.originalCropLeft = null;
        this.originalCropTop = null;
        this.originalCropWidth = null;
        this.originalCropHeight = null;
        this.originalCropCenterX = null;
        this.originalCropCenterY = null;
        this.dragStartLocal = null;
        this.graph.setDirtyCanvas(true);
        return true;
      }
      
      return false;
    };
  },
});

