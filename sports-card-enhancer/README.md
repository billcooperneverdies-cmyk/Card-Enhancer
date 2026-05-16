# Sports Card Image Enhancement Web Application

A state-of-the-art web application for bulk enhancement of scanned sports card images, featuring AI-powered blemish detection and removal, color correction, sharpening, and upscaling.

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![React](https://img.shields.io/badge/react-18+-cyan)
![FastAPI](https://img.shields.io/badge/fastapi-0.109+-teal)

## Features

### Core Capabilities
- **Bulk Image Upload**: Support for JPG, PNG, TIFF, BMP, WebP formats
- **ZIP Archive Support**: Upload ZIP files containing multiple card images for batch processing
- **AI Blemish Detection**: Automatically detects scratches, dust, scuffs, print artifacts, and border damage
- **Blemish Removal**: Intelligent inpainting to restore damaged areas
- **Image Enhancement**:
  - Sharpening and detail enhancement
  - Color correction and temperature adjustment
  - Contrast enhancement
  - Noise reduction
  - AI upscaling (2x, 4x)
- **Holographic Preservation**: Special handling to preserve holographic foil regions
- **Batch Processing**: Process multiple cards simultaneously with job queue
- **Real-time Preview**: Before/after comparison with interactive slider
- **Configurable Output**: Choose format (PNG, JPG, WebP, TIFF), quality, and DPI

### User Interface
- **Holographic Black Theme**: Futuristic UI with Three.js-powered background effects
- **Drag & Drop Upload**: Intuitive file upload with preview
- **Job Monitor**: Real-time progress tracking for all processing jobs
- **Granular Controls**: Fine-tune enhancement settings per job
- **Responsive Design**: Works on desktop and tablet devices

## Architecture

### Production Build (React + FastAPI)
```
├── Frontend (React + TypeScript + Vite)
│   ├── Three.js holographic background
│   ├── Framer Motion animations
│   ├── Tailwind CSS + shadcn/ui components
│   └── Real-time WebSocket updates
│
└── Backend (FastAPI + Python)
    ├── Image ingestion pipeline
    ├── Blemish detection engine (OpenCV)
    ├── Enhancement service
    ├── Batch processor with async queue
    └── WebSocket progress notifications
```

### Streamlit Prototype
- Single-file Python application
- Rapid deployment for testing
- Same core enhancement algorithms

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Redis (optional, for production Celery)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

```bash
cd app

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Streamlit Prototype

```bash
# Install Streamlit dependencies
pip install -r requirements-streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and status |
| `/upload` | POST | Upload images for enhancement |
| `/enhance` | POST | Upload and start enhancement job |
| `/status/{job_id}` | GET | Get job status and progress |
| `/download/{job_id}` | GET | Get download URL for completed job |
| `/download/{job_id}/file` | GET | Direct file download |
| `/preview` | POST | Generate quick preview |
| `/ws/{job_id}` | WebSocket | Real-time progress updates |
| `/health` | GET | Health check |

## Enhancement Settings

### Blemish Removal
- **Detection Sensitivity**: 0.0 - 1.0 (higher = more aggressive detection)
- **Types Detected**: Scratches, dust, scuffs, print artifacts, border damage

### Sharpening
- **Amount**: 0.0 - 1.0 (strength of sharpening effect)

### Color Correction
- **Temperature**: -1.0 to 1.0 (cooler to warmer)
- **Saturation**: 0.0 to 2.0 (grayscale to oversaturated)

### Contrast Enhancement
- **Amount**: 0.0 - 1.0 (strength of contrast adjustment)

### Noise Reduction
- **Strength**: 0.0 - 1.0 (higher = more smoothing)

### Upscaling
- **Factor**: 1x, 2x, or 4x (uses Lanczos interpolation, Real-ESRGAN optional)

### Output Settings
- **Format**: PNG, JPG, WebP, or TIFF
- **Quality**: 50-100% (for lossy formats)
- **DPI**: 72, 150, 300, 600, or 1200 (for print/archival purposes)

## Project Structure

```
.
├── app/                          # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   │   ├── HolographicBackground.tsx
│   │   │   ├── BatchUploader.tsx
│   │   │   ├── EnhancementSettings.tsx
│   │   │   ├── JobMonitor.tsx
│   │   │   └── ImagePreview.tsx
│   │   ├── services/
│   │   │   └── api.ts            # API client
│   │   ├── App.tsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                      # FastAPI backend
│   ├── app/
│   │   ├── api/                  # API routes
│   │   ├── core/                 # Configuration
│   │   ├── models/               # Pydantic schemas
│   │   ├── services/             # Business logic
│   │   │   ├── blemish_detector.py
│   │   │   ├── enhancement_service.py
│   │   │   └── batch_processor.py
│   │   ├── utils/                # Utilities
│   │   │   └── image_utils.py
│   │   └── main.py               # FastAPI app
│   └── requirements.txt
│
├── streamlit_app.py              # Streamlit prototype
├── requirements-streamlit.txt
└── README.md
```

## Technology Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **Three.js + React Three Fiber** - 3D background effects
- **Framer Motion** - Animations

### Backend
- **FastAPI** - Web framework
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **Pillow** - Image manipulation
- **PyTorch** - Deep learning (optional)
- **WebSockets** - Real-time updates

### Algorithms
- **Blemish Detection**: Canny edge detection + Hough transform
- **Inpainting**: OpenCV Telea/Navier-Stokes methods
- **Sharpening**: Unsharp mask + kernel convolution
- **Noise Reduction**: Non-local means denoising
- **Upscaling**: Lanczos interpolation (Real-ESRGAN optional)

## Performance Considerations

- **Client-side preprocessing**: Images are resized before upload
- **Async batch processing**: Multiple jobs handled concurrently
- **WebSocket updates**: Real-time progress without polling
- **Virtualized lists**: Efficient rendering of large job lists
- **Tile-based processing**: For memory-efficient upscaling

## Deployment

### Frontend (Static)
```bash
cd app
npm run build
# Deploy dist/ folder to CDN or static hosting
```

### Backend (Docker)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```env
# Backend
DEBUG=false
MAX_FILE_SIZE=52428800
MAX_BATCH_SIZE=100
MAX_CONCURRENT_JOBS=4
REDIS_URL=redis://localhost:6379/0

# Frontend
VITE_API_URL=http://localhost:8000
```

## Future Enhancements

- [ ] Integration with Real-ESRGAN for super-resolution
- [ ] Deep learning-based inpainting (LaMa)
- [ ] Automatic card border detection and cropping
- [ ] EXIF metadata preservation
- [ ] Cloud storage integration (S3, R2)
- [ ] User authentication and job history
- [ ] Mobile app (React Native)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenCV community for computer vision algorithms
- Real-ESRGAN team for super-resolution research
- shadcn/ui for beautiful React components
