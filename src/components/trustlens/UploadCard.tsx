import { useState, useCallback, useRef } from "react";
import { CloudUpload, Link2, Image, Film, Lock, Zap, AudioWaveform } from "lucide-react";

interface UploadCardProps {
  onAnalyze: (file?: File, url?: string) => void;
}

export function UploadCard({ onAnalyze }: UploadCardProps) {
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const droppedFile = e.dataTransfer.files[0];
        setFile(droppedFile);
        onAnalyze(droppedFile, undefined);
      } else {
        onAnalyze(undefined, url);
      }
    },
    [onAnalyze, url]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      onAnalyze(selectedFile, undefined);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (file) {
      onAnalyze(file, undefined);
    } else if (url) {
      onAnalyze(undefined, url);
    } else {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="relative w-full max-w-3xl rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-6 sm:p-8 shadow-2xl shadow-black/80 transition-all duration-300">
      {/* Wireframe Corner Markers */}
      <div className="absolute top-2 left-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute top-2 right-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute bottom-2 left-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute bottom-2 right-3 font-mono text-[10px] text-zinc-600 select-none">+</div>

      {/* Header */}
      <div className="mb-6 text-center">
        <div className="inline-flex items-center gap-2 mb-2 rounded-full border border-white/10 bg-zinc-900/80 px-3 py-1 font-mono text-[10px] tracking-widest text-white uppercase">
          <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
          SYS_MEDIA_SCANNER // INPUT_GATEWAY
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white uppercase font-sans">
          Verify Media Authenticity
        </h1>
        <p className="mt-1.5 text-xs leading-relaxed text-zinc-400 max-w-xl mx-auto">
          Upload an image, video, or audio file, or paste a URL to run multi-stage forensic signal detection.
        </p>
      </div>

      {/* Wireframe Divider */}
      <div className="mb-6 h-px w-full bg-gradient-to-r from-transparent via-white/15 to-transparent" />

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*,video/*,audio/*,.mp3,.wav,.m4a,.ogg,.flac"
        className="hidden"
      />

      {/* Dropzone with Sonar Rings */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`group relative flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border border-dashed p-8 sm:p-10 text-center transition-all duration-300 ${
          isDragging
            ? "border-white bg-white/10 scale-[1.01]"
            : "border-white/15 bg-zinc-950/60 hover:border-white/40 hover:bg-zinc-900/60"
        }`}
      >
        {/* Dot Grid Background */}
        <div className="pointer-events-none absolute inset-0 bg-dot-grid opacity-30" />

        {/* Sonar Radar Concentric Rings */}
        <svg className="pointer-events-none absolute h-56 w-56 opacity-20 text-white group-hover:scale-110 transition-transform duration-500" viewBox="0 0 200 200" fill="none">
          <circle cx="100" cy="100" r="30" stroke="currentColor" strokeWidth="0.75" strokeDasharray="2 2" />
          <circle cx="100" cy="100" r="60" stroke="currentColor" strokeWidth="0.75" />
          <circle cx="100" cy="100" r="90" stroke="currentColor" strokeWidth="0.75" strokeDasharray="4 4" />
          <line x1="100" y1="0" x2="100" y2="200" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2 2" />
          <line x1="0" y1="100" x2="200" y2="100" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2 2" />
        </svg>

        {/* Icon Badge */}
        <div className="relative z-10 flex h-16 w-16 items-center justify-center rounded-xl border border-white/30 bg-white/10 text-white shadow-[0_0_20px_rgba(255,255,255,0.15)] transition-all duration-300 group-hover:scale-105 group-hover:border-white">
          <CloudUpload className="h-8 w-8 animate-float transition-transform duration-300" />
        </div>

        <p className="relative z-10 mt-4 text-sm sm:text-base font-medium text-zinc-200 transition-colors group-hover:text-white">
          {file ? file.name : "Drop media file here or click to browse"}
        </p>

        {/* Format pills inside dropzone */}
        <div className="relative z-10 mt-3 flex items-center justify-center gap-1.5 font-mono text-[10px]">
          {file ? (
            <span className="text-zinc-300 font-semibold">
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </span>
          ) : (
            <>
              <span className="rounded border border-white/10 bg-zinc-900/90 px-2 py-0.5 uppercase text-zinc-300">
                JPG / PNG
              </span>
              <span className="rounded border border-white/10 bg-zinc-900/90 px-2 py-0.5 uppercase text-zinc-300">
                MP4 / MOV
              </span>
              <span className="rounded border border-white/10 bg-zinc-900/90 px-2 py-0.5 uppercase text-zinc-300">
                MP3 / WAV
              </span>
              <span className="ml-1 text-zinc-500">MAX 50MB</span>
            </>
          )}
        </div>
      </div>

      {/* Divider */}
      <div className="my-5 flex items-center gap-3">
        <div className="h-px flex-1 bg-white/10" />
        <span className="font-mono text-[10px] tracking-widest text-zinc-500 uppercase">OR PASTE URL</span>
        <div className="h-px flex-1 bg-white/10" />
      </div>

      {/* URL Form & Button */}
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="group relative">
          <Link2 className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500 transition-colors group-focus-within:text-white" />
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/media.mp4"
            className="h-12 w-full rounded-xl border border-white/15 bg-zinc-950/80 pl-11 pr-4 font-mono text-xs text-zinc-200 outline-none transition-all placeholder:text-zinc-600 focus:border-white focus:ring-1 focus:ring-white"
          />
        </div>
        <button
          type="submit"
          className="group relative flex h-12 w-full items-center justify-center overflow-hidden rounded-xl bg-white px-6 font-mono text-xs font-bold uppercase tracking-widest text-black shadow-lg shadow-white/20 transition-all duration-200 hover:bg-zinc-200 hover:shadow-white/30 active:scale-[0.99]"
        >
          <span className="pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/30 to-transparent group-hover:animate-shimmer" />
          <span className="flex items-center gap-2">
            <Zap className="h-4 w-4 fill-black text-black" />
            RUN FORENSIC ANALYSIS
          </span>
        </button>
      </form>

      {/* Security indicators */}
      <div className="mt-4 flex items-center justify-center gap-6 font-mono text-[10px] tracking-wider text-zinc-500 uppercase">
        <span className="inline-flex items-center gap-1.5">
          <Lock className="h-3 w-3 text-emerald-400" />
          AES ENCRYPTED PIPELINE
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Zap className="h-3 w-3 text-white" />
          MULTI-FRAME FORENSIC SCAN
        </span>
      </div>

      {/* Format pills bottom */}
      <div className="mt-5 flex justify-center gap-3">
        <div className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-1.5 font-mono text-[10px] text-zinc-400">
          <Image className="h-3.5 w-3.5 text-blue-400" />
          IMAGE ELA
        </div>
        <div className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-1.5 font-mono text-[10px] text-zinc-400">
          <Film className="h-3.5 w-3.5 text-white" />
          KEYFRAME SCAN
        </div>
        <div className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-1.5 font-mono text-[10px] text-zinc-400">
          <AudioWaveform className="h-3.5 w-3.5 text-purple-400" />
          PROSODY SPECTRUM
        </div>
      </div>
    </div>
  );
}
