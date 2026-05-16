import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Image as ImageIcon, FileArchive, Check, Archive, Cpu, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { toast } from 'sonner';

interface BatchUploaderProps {
  onFilesSelected: (files: File[]) => void;
}

const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp'];
const ALLOWED_ARCHIVE_TYPES = ['application/zip', 'application/x-zip-compressed'];
const ALLOWED_TYPES = [...ALLOWED_IMAGE_TYPES, ...ALLOWED_ARCHIVE_TYPES];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB for images
const MAX_ZIP_SIZE = 500 * 1024 * 1024; // 500MB for ZIP files

export function BatchUploader({ onFilesSelected }: BatchUploaderProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [previews, setPreviews] = useState<Map<string, string>>(new Map());
  const inputRef = useRef<HTMLInputElement>(null);

  const isZipFile = (file: File) => {
    return ALLOWED_ARCHIVE_TYPES.includes(file.type) || file.name.toLowerCase().endsWith('.zip');
  };

  const generatePreview = (file: File) => {
    if (isZipFile(file)) {
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviews(prev => new Map(prev).set(file.name, e.target?.result as string));
    };
    reader.readAsDataURL(file);
  };

  const validateFile = (file: File): string | null => {
    const isZip = isZipFile(file);
    
    if (!ALLOWED_TYPES.includes(file.type) && !isZip) {
      return `Unsupported format: ${file.type || 'unknown'}`;
    }
    
    const maxSize = isZip ? MAX_ZIP_SIZE : MAX_FILE_SIZE;
    if (file.size > maxSize) {
      const maxMB = maxSize / 1024 / 1024;
      return `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB (max ${maxMB}MB)`;
    }
    return null;
  };

  const handleFiles = useCallback((newFiles: FileList | null) => {
    if (!newFiles) return;

    const validFiles: File[] = [];
    const errors: string[] = [];

    Array.from(newFiles).forEach(file => {
      const error = validateFile(file);
      if (error) {
        errors.push(`${file.name}: ${error}`);
      } else {
        validFiles.push(file);
        generatePreview(file);
      }
    });

    if (errors.length > 0) {
      errors.forEach(error => toast.error(error));
    }

    if (validFiles.length > 0) {
      setFiles(prev => [...prev, ...validFiles]);
      const zipCount = validFiles.filter(f => isZipFile(f)).length;
      const imageCount = validFiles.length - zipCount;
      let message = `Added ${validFiles.length} file(s)`;
      if (zipCount > 0) {
        message += ` (${zipCount} ZIP archive${zipCount > 1 ? 's' : ''})`;
      }
      toast.success(message);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const removeFile = useCallback((index: number) => {
    setFiles(prev => {
      const file = prev[index];
      setPreviews(p => {
        const newMap = new Map(p);
        newMap.delete(file.name);
        return newMap;
      });
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  const clearAll = useCallback(() => {
    setFiles([]);
    setPreviews(new Map());
  }, []);

  const handleSubmit = useCallback(() => {
    if (files.length === 0) {
      toast.error('Please select at least one file');
      return;
    }
    onFilesSelected(files);
    clearAll();
  }, [files, onFilesSelected, clearAll]);

  const totalSize = files.reduce((acc, file) => acc + file.size, 0);

  return (
    <div className="h-full flex flex-col p-6">
      <div className="max-w-6xl mx-auto w-full flex-1 flex flex-col">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold tracking-wider mb-3">
            <span className="holographic-text">UPLOAD</span>
            <span className="text-white/90 ml-3">SPORTS CARDS</span>
          </h2>
          <p className="text-gray-500 tracking-wide">
            Drag and drop your scanned card images or click to browse
          </p>
        </div>

        {/* Drop Zone */}
        <motion.div
          className={`drop-zone relative rounded-2xl p-16 text-center mb-6 cursor-pointer overflow-hidden ${
            isDragging ? 'drag-over' : ''
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          whileHover={{ scale: 1.005 }}
          whileTap={{ scale: 0.995 }}
        >
          {/* Animated background grid */}
          <div className="absolute inset-0 grid-bg opacity-30" />
          
          {/* Corner brackets */}
          <div className="absolute top-4 left-4 w-8 h-8 border-l-2 border-t-2 border-cyan-500/50" />
          <div className="absolute top-4 right-4 w-8 h-8 border-r-2 border-t-2 border-magenta-500/50" />
          <div className="absolute bottom-4 left-4 w-8 h-8 border-l-2 border-b-2 border-cyan-500/50" />
          <div className="absolute bottom-4 right-4 w-8 h-8 border-r-2 border-b-2 border-magenta-500/50" />
          
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.tiff,.tif,.bmp,.webp,.zip"
            onChange={(e) => handleFiles(e.target.files)}
            className="hidden"
          />
          
          <motion.div
            animate={{ y: isDragging ? -15 : 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="relative z-10"
          >
            {/* Upload icon with glow */}
            <div className="relative w-24 h-24 mx-auto mb-6">
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-cyan-500/20 to-magenta-500/20 blur-xl" />
              <div className="relative w-full h-full rounded-full bg-black/50 border border-cyan-500/30 flex items-center justify-center">
                <Upload className={`w-10 h-10 transition-colors duration-300 ${
                  isDragging ? 'text-cyan-300' : 'text-cyan-500'
                }`} />
              </div>
              {/* Orbiting particles */}
              <motion.div
                className="absolute w-3 h-3 rounded-full bg-cyan-400"
                style={{ top: '50%', left: '50%' }}
                animate={{
                  x: [0, 40, 0, -40, 0],
                  y: [-40, 0, 40, 0, -40],
                }}
                transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
              />
              <motion.div
                className="absolute w-2 h-2 rounded-full bg-magenta-400"
                style={{ top: '50%', left: '50%' }}
                animate={{
                  x: [30, 0, -30, 0, 30],
                  y: [0, 30, 0, -30, 0],
                }}
                transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
              />
            </div>
            
            <h3 className="text-2xl font-semibold mb-3 tracking-wide">
              {isDragging ? (
                <span className="holographic-text">DROP FILES HERE</span>
              ) : (
                'Drag & Drop Images'
              )}
            </h3>
            <p className="text-gray-500 mb-6 text-sm tracking-wide">
              or click anywhere to browse from your computer
            </p>
            
            {/* Supported formats */}
            <div className="flex items-center justify-center gap-2 flex-wrap">
              {['JPG', 'PNG', 'TIFF', 'BMP', 'WebP'].map((format) => (
                <Badge 
                  key={format}
                  variant="outline" 
                  className="border-gray-700 text-gray-500 hover:border-cyan-500/50 hover:text-cyan-400 transition-colors text-xs tracking-wider"
                >
                  {format}
                </Badge>
              ))}
              <Badge 
                variant="outline" 
                className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10 text-xs tracking-wider"
              >
                <Archive className="w-3 h-3 mr-1" />
                ZIP
              </Badge>
            </div>
          </motion.div>
        </motion.div>

        {/* File List */}
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex-1 glass-card rounded-xl overflow-hidden flex flex-col"
          >
            {/* File List Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-cyan-500/20 bg-black/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
                  <FileArchive className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <span className="font-semibold tracking-wide">{files.length} file(s) selected</span>
                  <p className="text-xs text-gray-500 mono">
                    {(totalSize / 1024 / 1024).toFixed(2)} MB total
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAll}
                className="text-red-400 hover:text-red-300 hover:bg-red-500/10 tracking-wider"
              >
                <X className="w-4 h-4 mr-1" />
                CLEAR ALL
              </Button>
            </div>

            {/* File Grid */}
            <ScrollArea className="flex-1 p-4">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                <AnimatePresence>
                  {files.map((file, index) => (
                    <motion.div
                      key={`${file.name}-${index}`}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      layout
                      className="relative group"
                    >
                      <div className="sports-card bg-gray-900/80 border border-cyan-500/20 overflow-hidden">
                        {isZipFile(file) ? (
                          <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-cyan-900/30 via-black to-magenta-900/30">
                            <Archive className="w-12 h-12 text-cyan-400 mb-2" />
                            <span className="text-xs text-cyan-400 tracking-wider">ZIP ARCHIVE</span>
                          </div>
                        ) : previews.get(file.name) ? (
                          <img
                            src={previews.get(file.name)}
                            alt={file.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-900 to-black">
                            <ImageIcon className="w-12 h-12 text-gray-700" />
                          </div>
                        )}
                        
                        {/* Hover overlay */}
                        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <Button
                            variant="destructive"
                            size="icon"
                            className="rounded-full w-10 h-10 bg-red-500/80 hover:bg-red-500"
                            onClick={(e) => {
                              e.stopPropagation();
                              removeFile(index);
                            }}
                          >
                            <X className="w-5 h-5" />
                          </Button>
                        </div>
                        
                        {/* File info overlay */}
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/95 to-black/50 p-3">
                          <p className="text-xs truncate font-medium">{file.name}</p>
                          <p className="text-xs text-cyan-400/70 mono">
                            {(file.size / 1024).toFixed(1)} KB
                          </p>
                        </div>
                        
                        {/* Index badge */}
                        <div className="absolute top-2 left-2 w-6 h-6 rounded bg-black/70 border border-cyan-500/30 flex items-center justify-center">
                          <span className="text-xs text-cyan-400 mono">{index + 1}</span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </ScrollArea>

            {/* Submit Button */}
            <div className="p-4 border-t border-cyan-500/20 bg-black/30">
              <Button
                onClick={handleSubmit}
                className="w-full neon-button bg-gradient-to-r from-cyan-500/20 to-magenta-500/20 hover:from-cyan-500/30 hover:to-magenta-500/30 text-cyan-400 border border-cyan-500/50 h-14 text-lg tracking-widest"
              >
                <Cpu className="w-5 h-5 mr-3" />
                START ENHANCEMENT
                <Sparkles className="w-5 h-5 ml-3" />
              </Button>
            </div>
          </motion.div>
        )}

        {/* Empty State */}
        {files.length === 0 && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex-1 flex items-center justify-center"
          >
            <div className="text-center">
              <div className="relative w-20 h-20 mx-auto mb-6">
                <div className="absolute inset-0 rounded-full bg-gray-800/50 animate-pulse" />
                <div className="relative w-full h-full rounded-full border border-gray-700 flex items-center justify-center">
                  <ImageIcon className="w-8 h-8 text-gray-600" />
                </div>
              </div>
              <p className="text-gray-500 tracking-wide mb-1">No files selected</p>
              <p className="text-gray-600 text-sm">Upload images to begin AI enhancement</p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
