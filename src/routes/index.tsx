import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { ShieldCheck, AlertCircle, Shield, Scan, Lock, FileSearch } from "lucide-react";
import { UploadCard } from "@/components/trustlens/UploadCard";
import { LoadingState } from "@/components/trustlens/LoadingState";
import { ResultsView, type AnalysisResponse } from "@/components/trustlens/ResultsView";
import { StateToggle } from "@/components/trustlens/StateToggle";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TrustLens — Technical Media Verification" },
      {
        name: "description",
        content:
          "Verify media authenticity with TrustLens technical forensics. Detect manipulation, spectral prosody, and trace sources.",
      },
      { property: "og:title", content: "TrustLens — Media Verification" },
      {
        property: "og:description",
        content:
          "Verify the authenticity of images, videos, and audio with TrustLens.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Index,
});

type AppState = "upload" | "loading" | "results";

function Index() {
  const [appState, setAppState] = useState<AppState>("upload");
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [pendingResult, setPendingResult] = useState<AnalysisResponse | null>(null);

  const handleAnalyze = async (file?: File, url?: string) => {
    setErrorMessage(null);
    setIsBackendReady(false);
    setPendingResult(null);
    setAppState("loading");

    try {
      const isAudioFile = file && (
        file.type.startsWith("audio/") ||
        /\.(mp3|wav|m4a|ogg|flac)$/i.test(file.name)
      );

      const isAudioUrl = url && /\.(mp3|wav|m4a|ogg|flac)($|\?)/i.test(url);

      const isVideoFile = file && (
        file.type.startsWith("video/") ||
        /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name)
      );

      const isVideoUrl = url && /\.(mp4|avi|mov|mkv|webm)($|\?)/i.test(url);

      const targetEndpoint = (isAudioFile || isAudioUrl)
        ? "http://localhost:8000/analyze-audio"
        : (isVideoFile || isVideoUrl)
        ? "http://localhost:8000/analyze-video"
        : "http://localhost:8000/analyze";

      let response: Response;

      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        response = await fetch(targetEndpoint, {
          method: "POST",
          body: formData,
        });
      } else if (url) {
        response = await fetch(targetEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ url }),
        });
      } else {
        throw new Error("Please select a file or enter a valid URL.");
      }

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error (${response.status})`);
      }

      const result: AnalysisResponse = await response.json();
      setPendingResult(result);
      setIsBackendReady(true);
    } catch (err: any) {
      console.error("Analysis request failed:", err);
      const msg = err.message === "Failed to fetch"
        ? "Backend connection failed. Please ensure the Python FastAPI backend server is running on http://localhost:8000."
        : (err.message || "Failed to complete analysis. Please ensure backend is running.");
      setErrorMessage(msg);
      setAppState("upload");
    }
  };

  const handleLoadingComplete = () => {
    if (pendingResult) {
      setAnalysisData(pendingResult);
    }
    setAppState("results");
  };

  return (
    <div className="relative flex min-h-screen h-screen w-full flex-col items-center justify-between px-6 py-6 overflow-x-hidden overflow-y-auto bg-[#0d0d0d] text-zinc-100 font-sans">
      {/* Technical Editorial Sonar & Dot Grid Background */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden flex items-center justify-center">
        {/* Fine Dot Grid Overlay */}
        <div className="absolute inset-0 bg-dot-grid opacity-60" />
        
        {/* Halftone Vignette */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_30%,#0d0d0d_95%)]" />

        {/* Concentric Radar / Sonar Rings */}
        <svg className="absolute h-[900px] w-[900px] opacity-15 text-white pointer-events-none" viewBox="0 0 800 800" fill="none">
          <circle cx="400" cy="400" r="90" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" />
          <circle cx="400" cy="400" r="180" stroke="currentColor" strokeWidth="1" />
          <circle cx="400" cy="400" r="280" stroke="currentColor" strokeWidth="1" strokeDasharray="6 6" />
          <circle cx="400" cy="400" r="370" stroke="currentColor" strokeWidth="1" />
          {/* Radar axes lines */}
          <line x1="400" y1="0" x2="400" y2="800" stroke="currentColor" strokeWidth="1" strokeDasharray="2 4" />
          <line x1="0" y1="400" x2="800" y2="400" stroke="currentColor" strokeWidth="1" strokeDasharray="2 4" />
          {/* Accent focal dot */}
          <circle cx="400" cy="220" r="4" fill="#FFFFFF" />
          <line x1="400" y1="220" x2="480" y2="220" stroke="#FFFFFF" strokeWidth="1" strokeDasharray="2 2" />
        </svg>

        {/* Technical Corner Labels & Crosshair Markers */}
        <div className="absolute top-6 left-6 flex items-center gap-2 font-mono text-[10px] tracking-widest text-zinc-400 uppercase">
          <span className="h-1.5 w-1.5 rounded-full bg-white" />
          <span>SYS_NODE // FORENSIC_01</span>
        </div>
        <div className="absolute top-6 right-6 font-mono text-[10px] tracking-widest text-zinc-500 uppercase">
          <span>TRUST_METRIC // v3.2</span>
        </div>
        <div className="absolute bottom-6 left-6 font-mono text-[10px] tracking-widest text-zinc-500 uppercase">
          <span>FRAME_CHECK // READY</span>
        </div>
        <div className="absolute bottom-6 right-6 font-mono text-[10px] tracking-widest text-zinc-500 uppercase">
          <span>SECURE_PIPELINE</span>
        </div>
      </div>

      {/* Top Header */}
      <header className="relative z-10 flex items-center gap-3 py-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-md border border-white/40 bg-white/10 text-white shadow-[0_0_16px_rgba(255,255,255,0.15)]">
          <ShieldCheck className="h-5 w-5" />
        </div>
        <div className="flex flex-col">
          <span className="text-2xl font-black tracking-tight text-white uppercase font-sans">
            TrustLens
          </span>
          <span className="font-mono text-[9px] tracking-widest text-zinc-400 uppercase -mt-1">
            Technical Media Verification
          </span>
        </div>
      </header>

      {/* Error Notification Badge */}
      {errorMessage && (
        <div className="relative z-10 mb-4 flex items-center gap-3 rounded-lg border border-red-500/40 bg-red-950/30 px-4 py-3 font-mono text-xs text-red-400 max-w-xl w-full shadow-lg backdrop-blur-md">
          <AlertCircle className="h-4 w-4 shrink-0 text-red-500" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Main Content Area */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center w-full max-w-3xl my-4">
        {appState === "upload" && <UploadCard onAnalyze={handleAnalyze} />}
        {appState === "loading" && (
          <LoadingState
            isBackendReady={isBackendReady}
            onComplete={handleLoadingComplete}
          />
        )}
        {appState === "results" && (
          <ResultsView data={analysisData} onReset={() => setAppState("upload")} />
        )}
      </main>

      {/* Footer / Demo State Toggle */}
      <footer className="relative z-10 w-full flex flex-col items-center justify-center pb-2">
        <StateToggle current={appState} onChange={setAppState} />
      </footer>
    </div>
  );
}
