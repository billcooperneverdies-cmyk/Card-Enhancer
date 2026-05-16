import { motion, AnimatePresence } from 'framer-motion';
import { 
  Download, Eye, Trash2, Clock, CheckCircle, 
  AlertCircle, Loader2, Image as ImageIcon, Zap, Cpu, Activity
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { ProcessingJob } from '../App';

interface JobMonitorProps {
  jobs: ProcessingJob[];
  onDownload: (job: ProcessingJob) => void;
  onPreview: (job: ProcessingJob) => void;
  onDelete: (jobId: string) => void;
}

function StatusBadge({ status }: { status: string }) {
  const configs: Record<string, { icon: React.ReactNode; className: string; label: string }> = {
    pending: {
      icon: <Clock className="w-3 h-3" />,
      className: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
      label: 'PENDING'
    },
    processing: {
      icon: <Loader2 className="w-3 h-3 animate-spin" />,
      className: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 energy-pulse',
      label: 'PROCESSING'
    },
    completed: {
      icon: <CheckCircle className="w-3 h-3" />,
      className: 'bg-green-500/20 text-green-400 border-green-500/30',
      label: 'COMPLETE'
    },
    failed: {
      icon: <AlertCircle className="w-3 h-3" />,
      className: 'bg-red-500/20 text-red-400 border-red-500/30',
      label: 'FAILED'
    }
  };

  const config = configs[status] || configs.pending;

  return (
    <Badge variant="outline" className={`${config.className} flex items-center gap-1.5 tracking-wider text-xs font-medium`}>
      {config.icon}
      {config.label}
    </Badge>
  );
}

// Circular progress indicator
function ProgressRing({ progress, size = 48, strokeWidth = 4 }: { progress: number; size?: number; strokeWidth?: number }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="progress-ring" width={size} height={size}>
        {/* Background circle */}
        <circle
          className="stroke-gray-800"
          strokeWidth={strokeWidth}
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        {/* Progress circle */}
        <circle
          className="progress-ring-circle"
          stroke="url(#progressGradient)"
          strokeWidth={strokeWidth}
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset
          }}
        />
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00ffff" />
            <stop offset="100%" stopColor="#ff00ff" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-bold mono text-cyan-400">{Math.round(progress)}%</span>
      </div>
    </div>
  );
}

function JobCard({ 
  job, 
  onDownload, 
  onPreview, 
  onDelete 
}: { 
  job: ProcessingJob; 
  onDownload: (job: ProcessingJob) => void;
  onPreview: (job: ProcessingJob) => void;
  onDelete: (jobId: string) => void;
}) {
  const isComplete = job.status === 'completed';
  const isProcessing = job.status === 'processing';
  const isFailed = job.status === 'failed';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="glass-card rounded-xl p-5 card-hover relative overflow-hidden"
    >
      {/* Top accent line */}
      <div className={`absolute top-0 left-0 right-0 h-0.5 ${
        isComplete ? 'bg-gradient-to-r from-green-500 to-cyan-500' :
        isProcessing ? 'bg-gradient-to-r from-cyan-500 to-magenta-500 shimmer' :
        isFailed ? 'bg-gradient-to-r from-red-500 to-orange-500' :
        'bg-gradient-to-r from-orange-500 to-yellow-500'
      }`} />
      
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-4">
          {/* Progress ring for processing, icon for others */}
          {isProcessing ? (
            <ProgressRing progress={job.progress} />
          ) : (
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center border ${
              isComplete ? 'bg-green-500/10 border-green-500/30' : 
              isFailed ? 'bg-red-500/10 border-red-500/30' : 
              'bg-orange-500/10 border-orange-500/30'
            }`}>
              {isComplete ? (
                <CheckCircle className="w-6 h-6 text-green-400" />
              ) : isFailed ? (
                <AlertCircle className="w-6 h-6 text-red-400" />
              ) : (
                <Clock className="w-6 h-6 text-orange-400" />
              )}
            </div>
          )}
          
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h4 className="font-semibold tracking-wide">JOB</h4>
              <span className="mono text-cyan-400 text-sm">{job.id.slice(0, 8).toUpperCase()}</span>
            </div>
            <p className="text-xs text-gray-500 flex items-center gap-2">
              <ImageIcon className="w-3 h-3" />
              <span>{job.files.length} image(s)</span>
              <span className="text-gray-700">•</span>
              <span className="mono">{new Date().toLocaleTimeString()}</span>
            </p>
          </div>
        </div>
        <StatusBadge status={job.status} />
      </div>

      {/* Progress bar (for non-processing states showing completion) */}
      {!isProcessing && (
        <div className="mb-4">
          <div className="h-1.5 bg-gray-800/50 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${
                isComplete ? 'bg-gradient-to-r from-green-500 to-cyan-500' : 
                isFailed ? 'bg-gradient-to-r from-red-500 to-orange-500' : 
                'bg-gradient-to-r from-orange-500 to-yellow-500'
              }`}
              initial={{ width: 0 }}
              animate={{ width: `${job.progress}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
        </div>
      )}

      {/* Processing animation */}
      {isProcessing && (
        <div className="mb-4">
          <div className="h-1.5 bg-gray-800/50 rounded-full overflow-hidden">
            <motion.div
              className="h-full shimmer rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${job.progress}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
          <div className="flex items-center justify-center mt-3 text-xs text-cyan-400/80">
            <Activity className="w-3 h-3 mr-2 animate-pulse" />
            <span className="tracking-wider">AI ENHANCEMENT IN PROGRESS</span>
          </div>
        </div>
      )}

      {/* Settings Summary */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {job.settings.blemish_removal && (
          <Badge variant="outline" className="text-xs border-cyan-500/30 text-cyan-400 bg-cyan-500/5 tracking-wider">
            <Zap className="w-3 h-3 mr-1" />
            BLEMISH
          </Badge>
        )}
        {job.settings.sharpening && (
          <Badge variant="outline" className="text-xs border-purple-500/30 text-purple-400 bg-purple-500/5 tracking-wider">
            SHARPEN
          </Badge>
        )}
        {job.settings.color_correction && (
          <Badge variant="outline" className="text-xs border-pink-500/30 text-pink-400 bg-pink-500/5 tracking-wider">
            COLOR
          </Badge>
        )}
        {job.settings.upscaling && (
          <Badge variant="outline" className="text-xs border-green-500/30 text-green-400 bg-green-500/5 tracking-wider">
            {job.settings.upscale_factor}× UPSCALE
          </Badge>
        )}
        {job.settings.noise_reduction && (
          <Badge variant="outline" className="text-xs border-blue-500/30 text-blue-400 bg-blue-500/5 tracking-wider">
            DENOISE
          </Badge>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        {isComplete && (
          <>
            <Button
              variant="outline"
              size="sm"
              className="flex-1 border-cyan-500/30 hover:bg-cyan-500/10 hover:border-cyan-500/50 text-cyan-400 tracking-wider"
              onClick={() => onPreview(job)}
            >
              <Eye className="w-4 h-4 mr-2" />
              PREVIEW
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="flex-1 border-green-500/30 hover:bg-green-500/10 hover:border-green-500/50 text-green-400 tracking-wider"
              onClick={() => onDownload(job)}
            >
              <Download className="w-4 h-4 mr-2" />
              DOWNLOAD
            </Button>
          </>
        )}
        <Button
          variant="ghost"
          size="sm"
          className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
          onClick={() => onDelete(job.id)}
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </div>

      {/* Error Message */}
      {isFailed && job.error && (
        <motion.div 
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg"
        >
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-red-400 mono">{job.error}</p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

export function JobMonitor({ jobs, onDownload, onPreview, onDelete }: JobMonitorProps) {
  const processingJobs = jobs.filter(j => j.status === 'processing' || j.status === 'pending');
  const completedJobs = jobs.filter(j => j.status === 'completed');
  const failedJobs = jobs.filter(j => j.status === 'failed');

  if (jobs.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="relative w-24 h-24 mx-auto mb-6">
            <div className="absolute inset-0 rounded-full bg-gray-800/30 animate-pulse" />
            <div className="relative w-full h-full rounded-full border border-gray-700 flex items-center justify-center">
              <Cpu className="w-10 h-10 text-gray-600" />
            </div>
          </div>
          <h3 className="text-xl font-semibold mb-2 tracking-wide text-gray-400">NO ACTIVE JOBS</h3>
          <p className="text-sm text-gray-600 tracking-wide">Upload images to start processing</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold tracking-wider mb-1">
            <span className="holographic-text">JOB</span>
            <span className="text-white/90 ml-2">MONITOR</span>
          </h2>
          <p className="text-sm text-gray-500 tracking-wide">Track your enhancement jobs in real-time</p>
        </div>
        <div className="flex gap-3">
          {processingJobs.length > 0 && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
              <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
              <span className="text-cyan-400 text-sm font-medium tracking-wider">{processingJobs.length} ACTIVE</span>
            </div>
          )}
          {completedJobs.length > 0 && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-green-500/10 border border-green-500/30">
              <CheckCircle className="w-3 h-3 text-green-400" />
              <span className="text-green-400 text-sm font-medium tracking-wider">{completedJobs.length} DONE</span>
            </div>
          )}
        </div>
      </div>

      <ScrollArea className="flex-1 pr-4">
        <div className="space-y-4">
          {/* Processing Jobs */}
          <AnimatePresence>
            {processingJobs.map(job => (
              <JobCard
                key={job.id}
                job={job}
                onDownload={onDownload}
                onPreview={onPreview}
                onDelete={onDelete}
              />
            ))}
          </AnimatePresence>

          {/* Completed Jobs */}
          <AnimatePresence>
            {completedJobs.map(job => (
              <JobCard
                key={job.id}
                job={job}
                onDownload={onDownload}
                onPreview={onPreview}
                onDelete={onDelete}
              />
            ))}
          </AnimatePresence>

          {/* Failed Jobs */}
          <AnimatePresence>
            {failedJobs.map(job => (
              <JobCard
                key={job.id}
                job={job}
                onDownload={onDownload}
                onPreview={onPreview}
                onDelete={onDelete}
              />
            ))}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  );
}
