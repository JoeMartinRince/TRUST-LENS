import { useEffect, useState, useRef } from "react";
import { Loader2, CheckCircle2, ShieldCheck, Cpu } from "lucide-react";

export interface ScanStep {
  label: string;
  duration: number;
}

export const SCAN_STEPS: ScanStep[] = [
  { label: "Reading file metadata...", duration: 800 },
  { label: "Extracting EXIF and embedded data...", duration: 900 },
  { label: "Running error-level compression analysis...", duration: 1000 },
  { label: "Scanning for pixel-level manipulation artifacts...", duration: 1000 },
  { label: "Cross-referencing AI generation signatures...", duration: 1200 },
  { label: "Running deep neural authenticity detection...", duration: 1400 },
  { label: "Searching for source matches online...", duration: 900 },
  { label: "Synthesizing forensic report...", duration: 1000 },
];

interface LoadingStateProps {
  isBackendReady?: boolean;
  onComplete?: () => void;
}

export function LoadingState({ isBackendReady = false, onComplete }: LoadingStateProps) {
  const [completedIndices, setCompletedIndices] = useState<number[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const isBackendReadyRef = useRef(isBackendReady);

  useEffect(() => {
    isBackendReadyRef.current = isBackendReady;
  }, [isBackendReady]);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;

    const runStep = (index: number) => {
      if (index >= SCAN_STEPS.length) return;

      const step = SCAN_STEPS[index];
      timeoutId = setTimeout(() => {
        if (index < SCAN_STEPS.length - 1) {
          setCompletedIndices((prev) => [...prev, index]);
          setCurrentStepIndex(index + 1);
          runStep(index + 1);
        } else {
          // Last step: check if real backend response is ready
          const checkBackendCompletion = () => {
            if (isBackendReadyRef.current) {
              setCompletedIndices((prev) => Array.from(new Set([...prev, index])));
              setTimeout(() => {
                onComplete?.();
              }, 400);
            } else {
              timeoutId = setTimeout(checkBackendCompletion, 200);
            }
          };
          checkBackendCompletion();
        }
      }, step.duration);
    };

    runStep(0);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [onComplete]);

  const totalSteps = SCAN_STEPS.length;
  const completedCount = completedIndices.length;
  const isAllComplete = completedCount === totalSteps;

  const progressPercent = isAllComplete
    ? 100
    : Math.min(96, Math.round(((completedCount + 0.35) / totalSteps) * 100));

  return (
    <div className="relative w-full max-w-3xl rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-6 sm:p-8 shadow-2xl shadow-black/80 transition-all duration-300">
      {/* Wireframe Corner Markers */}
      <div className="absolute top-2 left-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute top-2 right-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute bottom-2 left-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute bottom-2 right-3 font-mono text-[10px] text-zinc-600 select-none">+</div>

      {/* Header section with Sonar Rings */}
      <div className="flex flex-col items-center justify-center text-center mb-6">
        <div className="relative flex h-16 w-16 items-center justify-center rounded-xl border border-white/40 bg-white/10 text-white shadow-[0_0_24px_rgba(255,255,255,0.2)] mb-3 overflow-hidden">
          {/* Concentric Radar Rings inside badge */}
          <svg className="absolute inset-0 h-full w-full opacity-30 text-white animate-radar-sweep" viewBox="0 0 100 100" fill="none">
            <circle cx="50" cy="50" r="20" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" />
            <circle cx="50" cy="50" r="40" stroke="currentColor" strokeWidth="1" />
            <line x1="50" y1="0" x2="50" y2="100" stroke="currentColor" strokeWidth="0.5" />
            <line x1="0" y1="50" x2="100" y2="50" stroke="currentColor" strokeWidth="0.5" />
          </svg>
          <Cpu className="h-8 w-8 animate-pulse text-white relative z-10" />
        </div>
        <div className="inline-flex items-center gap-2 mb-1 font-mono text-[10px] tracking-widest text-white uppercase">
          <span className="h-1.5 w-1.5 rounded-full bg-white animate-ping" />
          FORENSIC_ENGINE // PIPELINE_EXEC
        </div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white uppercase font-sans">
          Deep Media Forensics in Progress
        </h2>
        <p className="mt-1 text-xs text-zinc-400 font-mono">
          Running multi-stage technical inspection and neural authenticity scans...
        </p>
      </div>

      {/* Progress Bar Header */}
      <div className="mb-5 space-y-2">
        <div className="flex items-center justify-between font-mono text-[10px] tracking-wider text-zinc-400 uppercase">
          <span className="flex items-center gap-1.5 text-white">
            <ShieldCheck className="h-3.5 w-3.5" />
            STAGE {Math.min(currentStepIndex + 1, totalSteps)} / {totalSteps}
          </span>
          <span className="text-zinc-200 font-semibold">{progressPercent}% COMPLETE</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-md bg-zinc-950 p-0.5 border border-white/10">
          <div
            className="h-full rounded-sm bg-gradient-to-r from-white via-zinc-200 to-zinc-400 transition-all duration-500 ease-out shadow-[0_0_12px_rgba(255,255,255,0.3)]"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Stacked Scan Checklist */}
      <div className="space-y-2.5">
        {SCAN_STEPS.slice(0, currentStepIndex + 1).map((step, idx) => {
          const isDone = completedIndices.includes(idx);

          return (
            <div
              key={idx}
              className={`flex items-center gap-3 p-3 rounded-xl border transition-all duration-300 ${
                isDone
                  ? "border-emerald-500/30 bg-emerald-950/10"
                  : "border-white/40 bg-white/10 shadow-sm"
              }`}
            >
              <div className="flex items-center justify-center shrink-0">
                {isDone ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 transition-transform duration-300 scale-110" />
                ) : (
                  <Loader2 className="h-4 w-4 animate-spin text-white" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <p
                  className={`text-xs font-mono transition-colors ${
                    isDone
                      ? "text-zinc-300"
                      : "text-white font-semibold animate-pulse"
                  }`}
                >
                  {step.label}
                </p>
              </div>

              <div className="shrink-0 font-mono text-[9px] tracking-wider uppercase">
                {isDone ? (
                  <span className="inline-flex items-center rounded bg-emerald-500/15 px-2 py-0.5 font-semibold text-emerald-400 border border-emerald-500/30">
                    COMPLETE
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 rounded bg-white/20 px-2 py-0.5 font-semibold text-white border border-white/40 animate-pulse">
                    SCANNING...
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
