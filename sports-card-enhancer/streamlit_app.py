"""
Sports Card Image Enhancement - Streamlit Prototype
A rapid deployment prototype for the sports card enhancement application.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import zipfile
from typing import List, Tuple, Optional
import tempfile
import os
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="CardEnhance AI - Sports Card Restoration",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for holographic theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #00ffff;
        --secondary-color: #ff00ff;
        --bg-color: #000000;
    }
    
    .main {
        background: linear-gradient(135deg, #000000 0%, #0a0a0a 50%, #000000 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.9);
        border-right: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.2), rgba(255, 0, 255, 0.2));
        border: 1px solid rgba(0, 255, 255, 0.5);
        color: #00ffff;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.4), rgba(255, 0, 255, 0.4));
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* File uploader */
    .css-1cpxqw2 {
        background: rgba(0, 255, 255, 0.05);
        border: 2px dashed rgba(0, 255, 255, 0.3);
        border-radius: 12px;
    }
    
    .css-1cpxqw2:hover {
        border-color: rgba(0, 255, 255, 0.6);
        background: rgba(0, 255, 255, 0.1);
    }
    
    /* Checkboxes */
    .stCheckbox > label > span {
        color: #00ffff;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 8px;
    }
    
    .stSuccess {
        background: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.3);
        border-radius: 8px;
    }
    
    .stError {
        background: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.3);
        border-radius: 8px;
    }
    
    /* Metric cards */
    .css-1xarl3l {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Separator */
    hr {
        border-color: rgba(0, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)


class BlemishDetector:
    """Detect blemishes in card images."""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.min_defect_size = int(10 + (1 - sensitivity) * 40)
    
    def detect_scratches(self, image: np.ndarray) -> List[Tuple]:
        """Detect scratches using edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Dilate to connect segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find lines
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, 
                                threshold=int(20 + (1-self.sensitivity)*30),
                                minLineLength=int(30 + (1-self.sensitivity)*50),
                                maxLineGap=10)
        
        blemishes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > self.min_defect_size:
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2-x1)+5, abs(y2-y1)+5
                    confidence = min(1.0, length/100) * self.sensitivity
                    blemishes.append(('scratch', confidence, (x, y, w, h)))
        
        return blemishes
    
    def detect_dust(self, image: np.ndarray) -> List[Tuple]:
        """Detect dust particles."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inverted = 255 - gray
        
        _, thresh = cv2.threshold(inverted, int(200 + self.sensitivity*30), 255, 
                                  cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blemishes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_defect_size or area > 500:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.5:
                    confidence = min(1.0, area/100) * self.sensitivity
                    blemishes.append(('dust', confidence, (x, y, w, h)))
        
        return blemishes
    
    def detect_all(self, image: np.ndarray) -> List[Tuple]:
        """Detect all blemish types."""
        blemishes = []
        blemishes.extend(self.detect_scratches(image))
        blemishes.extend(self.detect_dust(image))
        return blemishes


class ImageEnhancer:
    """Image enhancement operations."""
    
    @staticmethod
    def remove_blemishes(image: np.ndarray, blemishes: List[Tuple]) -> np.ndarray:
        """Remove detected blemishes using inpainting."""
        if not blemishes:
            return image
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for blemish in blemishes:
            _, _, (x, y, bw, bh) = blemish
            padding = 5
            x1, y1 = max(0, x-padding), max(0, y-padding)
            x2, y2 = min(w, x+bw+padding), min(h, y+bh+padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Dilate mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Inpaint
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return result
    
    @staticmethod
    def sharpen(image: np.ndarray, amount: float) -> np.ndarray:
        """Apply sharpening."""
        pil_img = Image.fromarray(image)
        factor = 1 + amount
        kernel = np.array([[-1, -1, -1],
                          [-1, factor + 8, -1],
                          [-1, -1, -1]]) * (amount / 8)
        kernel[1, 1] = factor
        sharpened = pil_img.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=factor))
        return np.array(sharpened)
    
    @staticmethod
    def adjust_color_temperature(image: np.ndarray, temperature: float) -> np.ndarray:
        """Adjust color temperature."""
        pil_img = Image.fromarray(image)
        r, g, b = pil_img.split()
        r = r.point(lambda i: min(255, int(i * (1 + temperature * 0.1))))
        b = b.point(lambda i: min(255, int(i * (1 - temperature * 0.1))))
        return np.array(Image.merge('RGB', (r, g, b)))
    
    @staticmethod
    def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust saturation."""
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_img)
        return np.array(enhancer.enhance(factor))
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, amount: float) -> np.ndarray:
        """Adjust contrast."""
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        factor = 0.5 + amount
        return np.array(enhancer.enhance(factor))
    
    @staticmethod
    def reduce_noise(image: np.ndarray, strength: float) -> np.ndarray:
        """Apply noise reduction."""
        h = int(3 + strength * 7)
        h = h if h % 2 == 1 else h + 1
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    @staticmethod
    def upscale(image: np.ndarray, factor: int) -> np.ndarray:
        """Upscale image."""
        h, w = image.shape[:2]
        new_h, new_w = h * factor, w * factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def enhance_image(image: np.ndarray, settings: dict) -> Tuple[np.ndarray, List]:
    """Apply full enhancement pipeline."""
    result = image.copy()
    blemishes = []
    
    # Blemish removal
    if settings['blemish_removal']:
        detector = BlemishDetector(settings['blemish_sensitivity'])
        blemishes = detector.detect_all(result)
        if blemishes:
            result = ImageEnhancer.remove_blemishes(result, blemishes)
    
    # Noise reduction
    if settings['noise_reduction']:
        result = ImageEnhancer.reduce_noise(result, settings['noise_reduction_strength'])
    
    # Color correction
    if settings['color_correction']:
        result = ImageEnhancer.adjust_color_temperature(result, settings['color_temperature'])
        result = ImageEnhancer.adjust_saturation(result, settings['saturation'])
    
    # Contrast enhancement
    if settings['contrast_enhancement']:
        result = ImageEnhancer.adjust_contrast(result, settings['contrast_amount'])
    
    # Sharpening
    if settings['sharpening']:
        result = ImageEnhancer.sharpen(result, settings['sharpening_amount'])
    
    # Upscaling
    if settings['upscaling'] and settings['upscale_factor'] > 1:
        result = ImageEnhancer.upscale(result, settings['upscale_factor'])
    
    return result, blemishes


def create_download_zip(images: List[Tuple[str, np.ndarray]], format: str, quality: int, dpi: int = 300) -> bytes:
    """Create ZIP file with enhanced images."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, image in images:
            img_buffer = io.BytesIO()
            pil_img = Image.fromarray(image)
            
            if format == 'jpg':
                pil_img.save(img_buffer, format='JPEG', quality=quality, dpi=(dpi, dpi))
                ext = 'jpg'
            elif format == 'webp':
                pil_img.save(img_buffer, format='WEBP', quality=quality)
                ext = 'webp'
            elif format == 'tiff':
                pil_img.save(img_buffer, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))
                ext = 'tiff'
            else:
                pil_img.save(img_buffer, format='PNG', dpi=(dpi, dpi))
                ext = 'png'
            
            # Update filename with new extension
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}_enhanced.{ext}"
            
            zip_file.writestr(new_filename, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">✨ CardEnhance AI</h1>
        <p style="color: #888; font-size: 1.2rem;">Sports Card Restoration Studio</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Settings
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Enhancement Settings</h2>", unsafe_allow_html=True)
        
        # Presets
        st.subheader("Quick Presets")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restoration", use_container_width=True):
                st.session_state.update({
                    'blemish_removal': True,
                    'blemish_sensitivity': 0.8,
                    'sharpening': True,
                    'sharpening_amount': 0.6,
                    'color_correction': True,
                    'contrast_enhancement': True,
                    'noise_reduction': True,
                    'upscaling': False
                })
        with col2:
            if st.button("⬆️ Upscale", use_container_width=True):
                st.session_state.update({
                    'blemish_removal': False,
                    'sharpening': True,
                    'sharpening_amount': 0.8,
                    'color_correction': False,
                    'contrast_enhancement': True,
                    'contrast_amount': 0.5,
                    'noise_reduction': False,
                    'upscaling': True,
                    'upscale_factor': 2
                })
        
        st.markdown("---")
        
        # Blemish Removal
        st.subheader("🔍 Blemish Removal")
        blemish_removal = st.checkbox("Enable blemish removal", 
                                      value=st.session_state.get('blemish_removal', True),
                                      key='blemish_removal')
        if blemish_removal:
            blemish_sensitivity = st.slider("Detection Sensitivity", 0.0, 1.0, 
                                           st.session_state.get('blemish_sensitivity', 0.7),
                                           key='blemish_sensitivity')
        
        # Sharpening
        st.subheader("✨ Sharpening")
        sharpening = st.checkbox("Enable sharpening", 
                                value=st.session_state.get('sharpening', True),
                                key='sharpening')
        if sharpening:
            sharpening_amount = st.slider("Sharpening Amount", 0.0, 1.0,
                                         st.session_state.get('sharpening_amount', 0.5),
                                         key='sharpening_amount')
        
        # Color Correction
        st.subheader("🎨 Color Correction")
        color_correction = st.checkbox("Enable color correction",
                                      value=st.session_state.get('color_correction', True),
                                      key='color_correction')
        if color_correction:
            color_temperature = st.slider("Color Temperature", -1.0, 1.0,
                                         st.session_state.get('color_temperature', 0.0),
                                         key='color_temperature')
            saturation = st.slider("Saturation", 0.0, 2.0,
                                  st.session_state.get('saturation', 1.0),
                                  key='saturation')
        
        # Contrast
        st.subheader("◐ Contrast Enhancement")
        contrast_enhancement = st.checkbox("Enable contrast enhancement",
                                          value=st.session_state.get('contrast_enhancement', True),
                                          key='contrast_enhancement')
        if contrast_enhancement:
            contrast_amount = st.slider("Contrast Amount", 0.0, 1.0,
                                       st.session_state.get('contrast_amount', 0.3),
                                       key='contrast_amount')
        
        # Noise Reduction
        st.subheader("🔇 Noise Reduction")
        noise_reduction = st.checkbox("Enable noise reduction",
                                     value=st.session_state.get('noise_reduction', True),
                                     key='noise_reduction')
        if noise_reduction:
            noise_reduction_strength = st.slider("Noise Reduction Strength", 0.0, 1.0,
                                                st.session_state.get('noise_reduction_strength', 0.5),
                                                key='noise_reduction_strength')
        
        # Upscaling
        st.subheader("⬆️ AI Upscaling")
        upscaling = st.checkbox("Enable upscaling",
                               value=st.session_state.get('upscaling', False),
                               key='upscaling')
        if upscaling:
            upscale_factor = st.select_slider("Upscale Factor", options=[1, 2, 4],
                                             value=st.session_state.get('upscale_factor', 2),
                                             key='upscale_factor')
        
        st.markdown("---")
        
        # Output Settings
        st.subheader("💾 Output Settings")
        output_format = st.selectbox("Output Format", ["png", "jpg", "webp", "tiff"], index=0)
        output_quality = st.slider("Output Quality", 50, 100, 95)
        output_dpi = st.select_slider("Output DPI", options=[72, 150, 300, 600, 1200], value=300,
                                      help="Higher DPI for print, lower for web/screen")
    
    # Main content
    # File uploader
    st.subheader("📤 Upload Images")
    uploaded_files = st.file_uploader(
        "Drag and drop your sports card images here",
        type=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp', 'webp', 'zip'],
        accept_multiple_files=True,
        help="Supported formats: JPG, PNG, TIFF, BMP, WebP, ZIP archives (Max 50MB per file)"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
        
        # Process button
        if st.button("🚀 Start Enhancement", type="primary", use_container_width=True):
            # Collect settings
            settings = {
                'blemish_removal': st.session_state.get('blemish_removal', True),
                'blemish_sensitivity': st.session_state.get('blemish_sensitivity', 0.7),
                'sharpening': st.session_state.get('sharpening', True),
                'sharpening_amount': st.session_state.get('sharpening_amount', 0.5),
                'color_correction': st.session_state.get('color_correction', True),
                'color_temperature': st.session_state.get('color_temperature', 0.0),
                'saturation': st.session_state.get('saturation', 1.0),
                'contrast_enhancement': st.session_state.get('contrast_enhancement', True),
                'contrast_amount': st.session_state.get('contrast_amount', 0.3),
                'noise_reduction': st.session_state.get('noise_reduction', True),
                'noise_reduction_strength': st.session_state.get('noise_reduction_strength', 0.5),
                'upscaling': st.session_state.get('upscaling', False),
                'upscale_factor': st.session_state.get('upscale_factor', 2),
            }
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            enhanced_images = []
            total_blemishes = 0
            
            # Collect all images (including from ZIPs)
            all_images = []
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
            
            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                
                if ext == '.zip':
                    # Extract images from ZIP
                    status_text.text(f"Extracting {uploaded_file.name}...")
                    try:
                        zip_bytes = uploaded_file.read()
                        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
                            for file_info in zip_ref.infolist():
                                if file_info.is_dir() or file_info.filename.startswith('__MACOSX'):
                                    continue
                                filename = os.path.basename(file_info.filename)
                                if filename.startswith('.'):
                                    continue
                                file_ext = os.path.splitext(filename)[1].lower()
                                if file_ext not in allowed_extensions:
                                    continue
                                try:
                                    img_data = zip_ref.read(file_info)
                                    all_images.append((filename, img_data))
                                except Exception:
                                    continue
                    except Exception as e:
                        st.warning(f"Could not extract {uploaded_file.name}: {e}")
                else:
                    # Regular image file
                    all_images.append((uploaded_file.name, uploaded_file.read()))
            
            if not all_images:
                st.error("No valid images found to process.")
            else:
                total_count = len(all_images)
                
                for i, (filename, file_bytes) in enumerate(all_images):
                    status_text.text(f"Processing {filename}... ({i+1}/{total_count})")
                    
                    try:
                        # Load image
                        img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if image is None:
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Enhance
                        enhanced, blemishes = enhance_image(image, settings)
                        enhanced_images.append((filename, enhanced))
                        total_blemishes += len(blemishes)
                    except Exception as e:
                        st.warning(f"Failed to process {filename}: {e}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_count)
                
                status_text.text("✅ Enhancement complete!")
                progress_bar.empty()
            
            # Store results
            st.session_state['enhanced_images'] = enhanced_images
            st.session_state['all_images'] = all_images  # Store originals for comparison
            st.session_state['settings'] = settings
            st.session_state['output_format'] = output_format
            st.session_state['output_quality'] = output_quality
            st.session_state['output_dpi'] = output_dpi
            
            # Success message
            if enhanced_images:
                st.success(f"🎉 Enhanced {len(enhanced_images)} image(s)! Detected and removed {total_blemishes} blemishes.")
        
        # Display results
        if 'enhanced_images' in st.session_state and st.session_state['enhanced_images']:
            st.markdown("---")
            st.subheader("📊 Results")
            
            enhanced_images = st.session_state['enhanced_images']
            all_images = st.session_state.get('all_images', [])
            
            # Create a mapping from filename to original bytes
            original_map = {name: data for name, data in all_images}
            
            # Create comparison columns
            for filename, enhanced in enhanced_images:
                st.markdown(f"**{filename}**")
                
                col1, col2 = st.columns(2)
                
                # Try to get original image
                original_bytes = original_map.get(filename)
                if original_bytes:
                    try:
                        img_array = np.asarray(bytearray(original_bytes), dtype=np.uint8)
                        original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if original is not None:
                            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                            with col1:
                                st.markdown("<p style='text-align: center; color: #888;'>Original</p>", 
                                           unsafe_allow_html=True)
                                st.image(original, use_column_width=True)
                    except Exception:
                        pass
                
                with col2:
                    st.markdown("<p style='text-align: center; color: #00ffff;'>Enhanced</p>", 
                               unsafe_allow_html=True)
                    st.image(enhanced, use_column_width=True)
                
                st.markdown("---")
            
            # Download button
            st.subheader("💾 Download")
            
            zip_data = create_download_zip(
                enhanced_images,
                st.session_state['output_format'],
                st.session_state['output_quality'],
                st.session_state.get('output_dpi', 300)
            )
            
            st.download_button(
                label=f"📥 Download All ({len(enhanced_images)} images)",
                data=zip_data,
                file_name=f"enhanced_cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #666;">
            <p style="font-size: 4rem; margin-bottom: 1rem;">📸</p>
            <h3 style="color: #888;">No Images Uploaded</h3>
            <p>Upload your sports card scans to begin enhancement</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>CardEnhance AI v1.0.0 | Powered by OpenCV & Deep Learning</p>
        <p style="font-size: 0.8rem;">Supported formats: JPG, PNG, TIFF, BMP | Max file size: 50MB</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
