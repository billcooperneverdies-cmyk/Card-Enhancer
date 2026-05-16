import { useState, useEffect, useCallback } from 'react';
import { 
  Upload, Sparkles, Layers, Image as ImageIcon, Zap, Activity
} from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { Toaster } from '@/components/ui/sonner';

import { HolographicBackground } from './components/HolographicBackground';
import { BatchUploader } from './components/BatchUploader';
import { JobMonitor } from './components/JobMonitor';
import { ImagePreview } from './components/ImagePreview';
import { EnhancementSettings } from './components/EnhancementSettings';
import { apiService, type JobStatus, type EnhancementSettings as SettingsType } from './services/api';

export interface ProcessingJob {
  id: string;
  status: JobStatus;
  progress: number;
  files: File[];
  results?: string[];
  error?: string;
  settings: SettingsType;
}

function App() {
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedJob, setSelectedJob] = useState<ProcessingJob | null>(null);
  const [settings, setSettings] = useState<SettingsType>({
    blemish_removal: true,
    blemish_sensitivity: 0.7,
    sharpening: true,
    sharpening_amount: 0.5,
    color_correction: true,
    color_temperature: 0,
    saturation: 1,
    contrast_enhancement: true,
    contrast_amount: 0.3,
    noise_reduction: true,
    noise_reduction_strength: 0.5,
    upscaling: false,
    upscale_factor: 2,
    preserve_holographic: true,
    output_format: 'png',
    output_quality: 95,
    output_dpi: 300
  });

  // Poll for job status updates
  useEffect(() => {
    const interval = setInterval(async () => {
      for (const job of jobs) {
        if (job.status === 'pending' || job.status === 'processing') {
          try {
            const status = await apiService.getJobStatus(job.id);
            updateJobStatus(job.id, status);
          } catch (error) {
            console.error('Failed to fetch job status:', error);
          }
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobs]);

  const updateJobStatus = (jobId: string, status: any) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        return {
          ...job,
          status: status.status,
          progress: status.progress,
          results: status.images?.filter((img: any) => img.processed_path).map((img: any) => img.processed_path)
        };
      }
      return job;
    }));
  };

  const handleFilesSelected = useCallback(async (files: File[]) => {
    try {
      const response = await apiService.enhanceImages(files, settings);
      
      const newJob: ProcessingJob = {
        id: response.job_id,
        status: 'pending',
        progress: 0,
        files,
        settings: { ...settings }
      };
      
      setJobs(prev => [newJob, ...prev]);
      setActiveTab('monitor');
      toast.success(`Job ${response.job_id.slice(0, 8)} started with ${files.length} images`);
    } catch (error) {
      toast.error('Failed to start enhancement job');
      console.error(error);
    }
  }, [settings]);

  const handleDownload = useCallback(async (job: ProcessingJob) => {
    try {
      const downloadInfo = await apiService.getDownloadUrl(job.id);
      window.open(downloadInfo.download_url, '_blank');
      toast.success('Download started');
    } catch (error) {
      toast.error('Failed to get download URL');
      console.error(error);
    }
  }, []);

  const handleDeleteJob = useCallback((jobId: string) => {
    setJobs(prev => prev.filter(job => job.id !== jobId));
    if (selectedJob?.id === jobId) {
      setSelectedJob(null);
    }
    toast.success('Job removed from list');
  }, [selectedJob]);

  const handlePreview = useCallback((job: ProcessingJob) => {
    setSelectedJob(job);
    setActiveTab('preview');
  }, []);

  const completedJobs = jobs.filter(j => j.status === 'completed');
  const processingJobs = jobs.filter(j => j.status === 'pending' || j.status === 'processing');

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden noise-overlay">
      <Toaster position="top-right" theme="dark" />
      
      {/* Holographic Background */}
      <HolographicBackground />
      
      {/* Main Content */}
      <div className="relative z-10 flex flex-col h-screen">
        {/* Header */}
        <header className="glass-dark border-b border-cyan-500/20 px-6 py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center gap-4">
              {/* Logo */}
              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-magenta-500 rounded-xl blur-lg opacity-50 group-hover:opacity-75 transition-opacity" />
                <div className="relative w-12 h-12 rounded-xl bg-black/80 border border-cyan-500/50 flex items-center justify-center">
                  <Zap className="w-6 h-6 text-cyan-400" />
                </div>
              </div>
              
              <div>
                <h1 className="text-2xl font-bold tracking-wider">
                  <span className="holographic-text">CARD</span>
                  <span className="text-white/90">ENHANCE</span>
                  <span className="holographic-text-subtle ml-2 text-lg">AI</span>
                </h1>
                <p className="text-xs text-cyan-400/70 tracking-widest uppercase">
                  Sports Card Restoration Studio
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-6">
              {/* Stats */}
              <div className="hidden md:flex items-center gap-4 text-sm">
                {completedJobs.length > 0 && (
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-green-500/10 border border-green-500/30">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    <span className="text-green-400 font-medium">{completedJobs.length} Complete</span>
                  </div>
                )}
                {processingJobs.length > 0 && (
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
                    <Activity className="w-3 h-3 text-cyan-400 animate-pulse" />
                    <span className="text-cyan-400 font-medium">{processingJobs.length} Processing</span>
                  </div>
                )}
              </div>
              
              {/* Status indicator */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/50 border border-cyan-500/20">
                <div className="w-2 h-2 rounded-full bg-cyan-400 pulse-glow" />
                <span className="text-xs text-cyan-400/80 mono tracking-wider">SYSTEM ONLINE</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Tabs */}
        <div className="flex-1 overflow-hidden">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
            <div className="glass-dark border-b border-cyan-500/20 px-6">
              <div className="max-w-7xl mx-auto">
                <TabsList className="bg-transparent border-0 h-14 gap-1">
                  <TabsTrigger 
                    value="upload" 
                    className="relative px-6 py-3 data-[state=active]:bg-transparent data-[state=active]:text-cyan-400 text-gray-500 hover:text-gray-300 transition-colors group"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    <span className="tracking-wider">UPLOAD</span>
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500 to-magenta-500 scale-x-0 group-data-[state=active]:scale-x-100 transition-transform" />
                  </TabsTrigger>
                  <TabsTrigger 
                    value="settings"
                    className="relative px-6 py-3 data-[state=active]:bg-transparent data-[state=active]:text-cyan-400 text-gray-500 hover:text-gray-300 transition-colors group"
                  >
                    <Sparkles className="w-4 h-4 mr-2" />
                    <span className="tracking-wider">SETTINGS</span>
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500 to-magenta-500 scale-x-0 group-data-[state=active]:scale-x-100 transition-transform" />
                  </TabsTrigger>
                  <TabsTrigger 
                    value="monitor"
                    className="relative px-6 py-3 data-[state=active]:bg-transparent data-[state=active]:text-cyan-400 text-gray-500 hover:text-gray-300 transition-colors group"
                  >
                    <Layers className="w-4 h-4 mr-2" />
                    <span className="tracking-wider">JOBS</span>
                    {jobs.length > 0 && (
                      <Badge className="ml-2 bg-cyan-500/20 text-cyan-400 border-cyan-500/30 text-xs">
                        {jobs.length}
                      </Badge>
                    )}
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500 to-magenta-500 scale-x-0 group-data-[state=active]:scale-x-100 transition-transform" />
                  </TabsTrigger>
                  {selectedJob && (
                    <TabsTrigger 
                      value="preview"
                      className="relative px-6 py-3 data-[state=active]:bg-transparent data-[state=active]:text-cyan-400 text-gray-500 hover:text-gray-300 transition-colors group"
                    >
                      <ImageIcon className="w-4 h-4 mr-2" />
                      <span className="tracking-wider">PREVIEW</span>
                      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500 to-magenta-500 scale-x-0 group-data-[state=active]:scale-x-100 transition-transform" />
                    </TabsTrigger>
                  )}
                </TabsList>
              </div>
            </div>

            <div className="flex-1 overflow-hidden">
              <TabsContent value="upload" className="h-full m-0">
                <BatchUploader onFilesSelected={handleFilesSelected} />
              </TabsContent>

              <TabsContent value="settings" className="h-full m-0 p-6">
                <div className="h-full overflow-auto">
                  <div className="max-w-2xl mx-auto">
                    <div className="mb-8">
                      <h2 className="text-2xl font-bold tracking-wider mb-2">
                        <span className="holographic-text">ENHANCEMENT</span>
                        <span className="text-white/90 ml-2">SETTINGS</span>
                      </h2>
                      <p className="text-gray-500 text-sm tracking-wide">
                        Configure AI-powered image enhancement parameters
                      </p>
                    </div>
                    <EnhancementSettings 
                      settings={settings} 
                      onSettingsChange={setSettings} 
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="monitor" className="h-full m-0 p-6">
                <JobMonitor 
                  jobs={jobs}
                  onDownload={handleDownload}
                  onPreview={handlePreview}
                  onDelete={handleDeleteJob}
                />
              </TabsContent>

              <TabsContent value="preview" className="h-full m-0">
                {selectedJob && (
                  <ImagePreview job={selectedJob} />
                )}
              </TabsContent>
            </div>
          </Tabs>
        </div>

        {/* Footer */}
        <footer className="glass-dark border-t border-cyan-500/20 px-6 py-3">
          <div className="flex items-center justify-between max-w-7xl mx-auto text-xs">
            <div className="flex items-center gap-6 text-gray-600">
              <span className="flex items-center gap-2">
                <span className="text-cyan-500">▸</span>
                <span>JPG, PNG, TIFF, BMP, WebP, ZIP</span>
              </span>
              <span className="flex items-center gap-2">
                <span className="text-magenta-500">▸</span>
                <span>Max 50MB per file</span>
              </span>
            </div>
            <div className="flex items-center gap-6">
              <span className="flex items-center gap-2 text-cyan-400/60">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                <span className="mono tracking-wider">API CONNECTED</span>
              </span>
              <span className="text-gray-600 mono">v2.0.0</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
