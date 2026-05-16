import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Download, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Maximize2, RotateCcw, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import type { ProcessingJob } from '../App';

interface ImagePreviewProps {
  job: ProcessingJob;
}

export function ImagePreview({ job }: ImagePreviewProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [zoom, setZoom] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);

  const currentFile = job.files[currentIndex];
  const totalFiles = job.files.length;

  // Generate object URL for the original file
  const originalUrl = currentFile ? URL.createObjectURL(currentFile) : '';
  const enhancedUrl = job.results?.[currentIndex] || '';

  useEffect(() => {
    return () => {
      if (originalUrl) URL.revokeObjectURL(originalUrl);
    };
  }, [originalUrl]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    updateSliderPosition(e);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      updateSliderPosition(e);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const updateSliderPosition = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = (x / rect.width) * 100;
    setSliderPosition(Math.max(0, Math.min(100, percentage)));
  };

  const handlePrevious = () => {
    setCurrentIndex(prev => (prev > 0 ? prev - 1 : totalFiles - 1));
    setSliderPosition(50);
  };

  const handleNext = () => {
    setCurrentIndex(prev => (prev < totalFiles - 1 ? prev + 1 : 0));
    setSliderPosition(50);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.25, 0.5));
  };

  const resetView = () => {
    setZoom(1);
    setSliderPosition(50);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="glass-dark border-b border-cyan-500/20 px-6 py-4">
        <div className="flex items-center justify-between max-w-6xl mx-auto">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-magenta-500/20 border border-cyan-500/30 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="font-semibold tracking-wide text-white">
                {currentFile?.name}
              </h3>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <Badge variant="outline" className="border-cyan-500/30 text-cyan-400 bg-cyan-500/5 text-xs tracking-wider">
                  {currentIndex + 1} / {totalFiles}
                </Badge>
                <span className="mono">{currentFile ? (currentFile.size / 1024).toFixed(1) : 0} KB</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Zoom Controls */}
            <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-black/50 border border-gray-800">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleZoomOut}
                disabled={zoom <= 0.5}
                className="w-8 h-8 hover:bg-cyan-500/10"
              >
                <ZoomOut className="w-4 h-4" />
              </Button>
              <span className="text-sm text-cyan-400 w-14 text-center mono">
                {Math.round(zoom * 100)}%
              </span>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleZoomIn}
                disabled={zoom >= 3}
                className="w-8 h-8 hover:bg-cyan-500/10"
              >
                <ZoomIn className="w-4 h-4" />
              </Button>
            </div>

            <Button
              variant="ghost"
              size="icon"
              onClick={resetView}
              className="w-8 h-8 hover:bg-cyan-500/10"
              title="Reset view"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>

            <div className="w-px h-6 bg-gray-800 mx-1" />

            {/* Navigation */}
            {totalFiles > 1 && (
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handlePrevious}
                  className="w-8 h-8 hover:bg-cyan-500/10"
                >
                  <ChevronLeft className="w-5 h-5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleNext}
                  className="w-8 h-8 hover:bg-cyan-500/10"
                >
                  <ChevronRight className="w-5 h-5" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Comparison View */}
      <div className="flex-1 flex items-center justify-center p-6 overflow-hidden bg-gradient-to-b from-black to-gray-950">
        <motion.div
          ref={containerRef}
          className="relative rounded-xl overflow-hidden cursor-ew-resize select-none shadow-2xl"
          style={{ transform: `scale(${zoom})` }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: zoom }}
          transition={{ duration: 0.3 }}
        >
          {/* Border glow effect */}
          <div className="absolute inset-0 rounded-xl border-2 border-cyan-500/30 pointer-events-none z-20" />
          
          {/* Original Image (Background) */}
          <img
            src={originalUrl}
            alt="Original"
            className="max-w-full max-h-[70vh] object-contain"
            draggable={false}
          />

          {/* Enhanced Image (Foreground - clipped) */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
          >
            {enhancedUrl ? (
              <img
                src={enhancedUrl}
                alt="Enhanced"
                className="max-w-full max-h-[70vh] object-contain"
                draggable={false}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-black/80">
                <span className="text-gray-500 tracking-wide">Enhanced version loading...</span>
              </div>
            )}
          </div>

          {/* Slider Handle */}
          <div
            className="absolute top-0 bottom-0 w-1 cursor-ew-resize z-10"
            style={{ 
              left: `${sliderPosition}%`,
              background: 'linear-gradient(180deg, #00ffff 0%, #ff00ff 50%, #00ffff 100%)',
              boxShadow: '0 0 20px rgba(0, 255, 255, 0.8), 0 0 40px rgba(255, 0, 255, 0.5)'
            }}
          >
            {/* Handle grip */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 rounded-full flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, #00ffff, #ff00ff)',
                boxShadow: '0 0 30px rgba(0, 255, 255, 0.8)'
              }}
            >
              <div className="flex gap-0.5">
                <div className="w-0.5 h-4 bg-black rounded" />
                <div className="w-0.5 h-4 bg-black rounded" />
              </div>
            </div>
          </div>

          {/* Labels */}
          <div className="absolute top-4 left-4 z-10">
            <Badge className="bg-black/80 text-gray-300 border border-gray-700 tracking-wider text-xs backdrop-blur-sm">
              ORIGINAL
            </Badge>
          </div>
          <div className="absolute top-4 right-4 z-10">
            <Badge className="bg-gradient-to-r from-cyan-500/80 to-magenta-500/80 text-white border-0 tracking-wider text-xs backdrop-blur-sm">
              <Sparkles className="w-3 h-3 mr-1" />
              ENHANCED
            </Badge>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <div className="glass-dark border-t border-cyan-500/20 px-6 py-4">
        <div className="flex items-center justify-between max-w-6xl mx-auto">
          <div className="flex items-center gap-6 text-sm text-gray-500">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400/50" />
              <span className="tracking-wide">Drag slider to compare</span>
            </span>
            <span className="mono text-xs">
              Position: {sliderPosition.toFixed(0)}%
            </span>
          </div>
          
          {enhancedUrl && (
            <Button
              className="neon-button bg-gradient-to-r from-green-500/20 to-cyan-500/20 hover:from-green-500/30 hover:to-cyan-500/30 text-green-400 border border-green-500/50 tracking-wider"
              onClick={() => window.open(enhancedUrl, '_blank')}
            >
              <Download className="w-4 h-4 mr-2" />
              DOWNLOAD ENHANCED
            </Button>
          )}
        </div>
      </div>

      {/* Thumbnail strip for multiple images */}
      {totalFiles > 1 && (
        <div className="glass-dark border-t border-cyan-500/20 px-6 py-3">
          <div className="flex items-center gap-2 overflow-x-auto max-w-6xl mx-auto">
            {job.files.map((file, index) => {
              const thumbUrl = URL.createObjectURL(file);
              return (
                <motion.button
                  key={index}
                  onClick={() => {
                    setCurrentIndex(index);
                    setSliderPosition(50);
                  }}
                  className={`relative flex-shrink-0 w-16 h-20 rounded-lg overflow-hidden border-2 transition-all ${
                    index === currentIndex 
                      ? 'border-cyan-400 shadow-lg shadow-cyan-500/30' 
                      : 'border-gray-800 hover:border-gray-600'
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <img
                    src={thumbUrl}
                    alt={file.name}
                    className="w-full h-full object-cover"
                  />
                  {index === currentIndex && (
                    <div className="absolute inset-0 bg-cyan-500/20" />
                  )}
                  <div className="absolute bottom-0.5 right-0.5 w-5 h-5 rounded bg-black/70 flex items-center justify-center">
                    <span className="text-xs mono text-cyan-400">{index + 1}</span>
                  </div>
                </motion.button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
