from fastapi import UploadFile, HTTPException
from PIL import Image
import os
import aiofiles
from typing import List, Tuple
import magic
import hashlib
from datetime import datetime

ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/jpg', 
    'image/png'
}

MAX_FILE_SIZE = 10 * 1024 * 1024

UPLOAD_DIR = "uploads"
TRAINING_DIR = os.path.join(UPLOAD_DIR, "training")
TEMP_DIR = os.path.join(UPLOAD_DIR, "temp")


def ensure_upload_directories():
    """Create upload directories if they don't exist"""
    for directory in [UPLOAD_DIR, TRAINING_DIR, TEMP_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    # Create subdirectories for each class
    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = os.path.join(TRAINING_DIR, class_name.lower())
        os.makedirs(class_dir, exist_ok=True)


async def validate_image(file: UploadFile) -> None:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Check content type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )
    
    # Read first chunk to validate it's actually an image
    content = await file.read(1024)  # Read first 1KB
    await file.seek(0)  # Reset file pointer
    
    try:
        # Use python-magic to detect file type
        mime_type = magic.from_buffer(content, mime=True)
        if mime_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File content doesn't match declared type. Detected: {mime_type}"
            )
    except Exception:
        # Fallback to PIL validation
        try:
            await file.seek(0)
            content = await file.read()
            await file.seek(0)
            
            # Try to open with PIL
            image = Image.open(file.file)
            image.verify()
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )


async def save_uploaded_file(file: UploadFile, label: str) -> str:
    """
    Save uploaded file to training directory
    
    Args:
        file: Uploaded file
        label: Class label (NORMAL or PNEUMONIA)
        
    Returns:
        str: Path where file was saved
    """
    ensure_upload_directories()
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
    filename = f"{timestamp}_{file_hash}_{file.filename}"
    
    # Determine save directory based on label
    label_dir = os.path.join(TRAINING_DIR, label.lower())
    file_path = os.path.join(label_dir, filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
        await file.seek(0)  # Reset for potential reuse
    
    return file_path


def get_image_info(image_path: str) -> dict:
    """
    Get information about an image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "size_bytes": os.path.getsize(image_path)
            }
    except Exception as e:
        return {"error": str(e)}


def validate_image_dimensions(image_path: str, min_size: Tuple[int, int] = (224, 224)) -> bool:
    """
    Validate image dimensions for model input
    
    Args:
        image_path: Path to image file
        min_size: Minimum required dimensions (width, height)
        
    Returns:
        bool: True if dimensions are valid
    """
    try:
        with Image.open(image_path) as img:
            return img.width >= min_size[0] and img.height >= min_size[1]
    except Exception:
        return False


def cleanup_temp_files(max_age_hours: int = 24):
    """
    Clean up temporary files older than specified age
    
    Args:
        max_age_hours: Maximum age of files to keep (in hours)
    """
    import time
    
    if not os.path.exists(TEMP_DIR):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                except Exception:
                    pass  # Ignore errors


def get_training_data_stats() -> dict:
    """
    Get statistics about training data
    
    Returns:
        dict: Training data statistics
    """
    ensure_upload_directories()
    
    stats = {
        "total_files": 0,
        "normal_count": 0,
        "pneumonia_count": 0,
        "classes": {}
    }
    
    for class_name in ["normal", "pneumonia"]:
        class_dir = os.path.join(TRAINING_DIR, class_name)
        if os.path.exists(class_dir):
            file_count = len([f for f in os.listdir(class_dir) 
                            if os.path.isfile(os.path.join(class_dir, f))])
            stats["classes"][class_name] = file_count
            stats["total_files"] += file_count
            
            if class_name == "normal":
                stats["normal_count"] = file_count
            elif class_name == "pneumonia":
                stats["pneumonia_count"] = file_count
    
    return stats


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        str: Formatted file size
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    import re
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace potentially dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename


# Initialize upload directories on import
ensure_upload_directories()
