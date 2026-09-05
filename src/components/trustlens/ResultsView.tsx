import {
  Camera,
  AlertTriangle,
  Search,
  FileWarning,
  ShieldAlert,
  Eye,
  Image,
  RefreshCw,
  ExternalLink,
  Info,
  Volume2,
  Activity,
  AudioWaveform,
  Bot,
  type LucideIcon,
} from "lucide-react";

export interface MetadataRow {
  label: string;
  value: string;
  flag: boolean;
}

export interface RedFlagItem {
  label: string;
  desc: string;
  type?: string;
}

export interface FilenameAnalysis {
  filename: string;
  pattern_type: "ai_generator" | "camera_native" | "generic_download" | "unrecognized";
  note: string;
}

export interface MetadataAnalysis {
  has_exif: boolean;
  camera_make: string | null;
  software_tag: string | null;
  creation_date: string | null;
  suspicious_flags: string[];
}

export interface ElaAnalysis {
  score: number;
  heatmap_url: string | null;
  original_image_url?: string | null;
}

export interface EarliestSource {
  domain: string;
  url: string;
  date_seen?: string | null;
}

export interface SourceTrace {
  found_matches: boolean;
  note?: string;
  earliest_source?: EarliestSource;
  total_matches?: number;
  domain?: string;
  date?: string;
  url?: string;
}

export interface AudioAnalysis {
  spectral_flatness: number;
  pitch_variance: number;
  harmonic_to_noise_ratio: number;
  suspicious_flags: string[];
}

export interface AiDetectorResult {
  ai_generation_confidence: number | null;
  per_frame_confidence?: number[];
  frames_analyzed?: number;
  frames_flagged_ai?: number;
  model_used?: string;
  raw_label?: string;
  note?: string;
}

export interface DeepfakeDetectorResult {
  deepfake_audio_confidence: number | null;
  model_used?: string;
  raw_label?: string;
  note?: string;
}

export interface AnalysisResponse {
  trust_score?: number;
  verdict?: string;
  malicious_percentage?: number;
  explanation: string;
  red_flags: RedFlagItem[];
  limitations: string;
  metadata_analysis: MetadataAnalysis;
  filename_analysis?: FilenameAnalysis;
  ela_analysis: ElaAnalysis;
  source_trace: SourceTrace;
  ai_detector?: AiDetectorResult;
  deepfake_detector?: DeepfakeDetectorResult;
  is_audio?: boolean;
  is_video?: boolean;
  audio_analysis?: AudioAnalysis;
}

interface ResultsViewProps {
  onReset: () => void;
  data?: AnalysisResponse | null;
}

export function ResultsView({ onReset, data }: ResultsViewProps) {
  const isAudio = data?.is_audio || Boolean(data?.audio_analysis);
  const isVideo = data?.is_video || data?.malicious_percentage !== undefined;

  return (
    <div className="w-full max-w-3xl space-y-5 pb-8 font-sans text-zinc-100">
      {isVideo ? (
        <MaliciousLikelihoodCard
          maliciousPercentage={data?.malicious_percentage}
          summary={data?.explanation}
        />
      ) : (
        <TrustScoreCard
          score={data?.trust_score}
          verdict={data?.verdict}
          summary={data?.explanation}
        />
      )}
      <MetadataAnalysisCard
        metadata={data?.metadata_analysis}
        filenameAnalysis={data?.filename_analysis}
        isAudio={isAudio}
      />
      
      {isAudio ? (
        <AudioSpectralStatsCard
          audioAnalysis={data?.audio_analysis}
          deepfakeDetector={data?.deepfake_detector}
        />
      ) : (
        <ManipulationScanCard
          score={data?.ela_analysis?.score}
          originalUrl={data?.ela_analysis?.original_image_url}
          heatmapUrl={data?.ela_analysis?.heatmap_url}
          explanation={data?.explanation}
        />
      )}

      <SourceTraceCard sourceTrace={data?.source_trace} />
      {!isAudio && <AiDetectorCard aiDetector={data?.ai_detector} />}
      <RedFlagsCard flags={data?.red_flags} />
      <LimitationsFooter text={data?.limitations} />

      <button
        onClick={onReset}
        className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-white px-6 font-mono text-xs font-bold uppercase tracking-widest text-black shadow-lg shadow-white/25 transition-all hover:bg-zinc-200 active:scale-[0.99]"
      >
        <RefreshCw className="h-4 w-4" />
        ANALYZE ANOTHER MEDIA FILE
      </button>
    </div>
  );
}

function MaliciousLikelihoodCard({
  maliciousPercentage = 30,
  summary = "Video verification completed based on keyframe classification and technical forensic signals.",
}: {
  maliciousPercentage?: number;
  summary?: string;
}) {
  let badgeClass = "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
  let pillText = "LOW RISK";

  if (maliciousPercentage >= 61) {
    badgeClass = "bg-red-500/15 text-red-400 border border-red-500/30";
    pillText = "HIGH RISK";
  } else if (maliciousPercentage >= 31) {
    badgeClass = "bg-zinc-800 text-zinc-200 border border-zinc-700";
    pillText = "MODERATE RISK";
  }

  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-6 shadow-2xl shadow-black/80">
      {/* Corner Wireframe Markers */}
      <div className="absolute top-2 left-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute top-2 right-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute bottom-2 left-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute bottom-2 right-3 font-mono text-[10px] text-zinc-600 select-none">+</div>

      {/* Sonar Rings behind score */}
      <svg className="pointer-events-none absolute right-[-20px] top-[-20px] h-64 w-64 opacity-15 text-white" viewBox="0 0 200 200" fill="none">
        <circle cx="100" cy="100" r="30" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" />
        <circle cx="100" cy="100" r="60" stroke="currentColor" strokeWidth="1" />
        <circle cx="100" cy="100" r="90" stroke="currentColor" strokeWidth="1" strokeDasharray="4 4" />
      </svg>

      <div className="relative z-10 flex items-start justify-between">
        <div>
          <span className="font-mono text-[10px] tracking-widest text-zinc-500 uppercase">
            MALICIOUS_CONTENT_LIKELIHOOD // RISK_ASSESSMENT
          </span>
          <p className="mt-1 text-5xl font-bold tracking-tight text-white font-mono">
            {maliciousPercentage}<span className="text-2xl font-medium text-zinc-500">%</span>
          </p>
        </div>
        <span className={`rounded-lg px-3 py-1 font-mono text-[10px] font-bold tracking-wider ${badgeClass}`}>
          {pillText}
        </span>
      </div>
      <p className="relative z-10 mt-4 text-xs leading-relaxed text-zinc-300">
        {summary}
      </p>
    </div>
  );
}

function TrustScoreCard({
  score = 75,
  verdict = "Likely Authentic",
  summary = "Media analysis indicates high authenticity signals. Metadata is consistent and low compression noise variance was detected.",
}: {
  score?: number;
  verdict?: string;
  summary?: string;
}) {
  let badgeClass = "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
  if (verdict.toLowerCase().includes("manipulated") || score < 50) {
    badgeClass = "bg-red-500/15 text-red-400 border border-red-500/30";
  } else if (verdict.toLowerCase().includes("suspicious") || score < 75) {
    badgeClass = "bg-zinc-800 text-zinc-200 border border-zinc-700";
  }

  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-6 shadow-2xl shadow-black/80">
      {/* Corner Wireframe Markers */}
      <div className="absolute top-2 left-3 font-mono text-[10px] text-zinc-600 select-none">+</div>
      <div className="absolute top-2 right-3 font-mono text-[10px] text-zinc-600 select-none">+</div>

      {/* Sonar Rings behind score */}
      <svg className="pointer-events-none absolute right-[-20px] top-[-20px] h-64 w-64 opacity-15 text-emerald-400" viewBox="0 0 200 200" fill="none">
        <circle cx="100" cy="100" r="30" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" />
        <circle cx="100" cy="100" r="60" stroke="currentColor" strokeWidth="1" />
        <circle cx="100" cy="100" r="90" stroke="currentColor" strokeWidth="1" strokeDasharray="4 4" />
      </svg>

      <div className="relative z-10 flex items-start justify-between">
        <div>
          <span className="font-mono text-[10px] tracking-widest text-zinc-500 uppercase">
            TRUST_SCORE // AUTHENTICITY_RATING
          </span>
          <p className="mt-1 text-5xl font-bold tracking-tight text-white font-mono">
            {score}<span className="text-2xl font-medium text-zinc-500">/100</span>
          </p>
        </div>
        <span className={`rounded-lg px-3 py-1 font-mono text-[10px] font-bold tracking-wider uppercase ${badgeClass}`}>
          {verdict}
        </span>
      </div>
      <p className="relative z-10 mt-4 text-xs leading-relaxed text-zinc-300">
        {summary}
      </p>
    </div>
  );
}

function MetadataAnalysisCard({
  metadata,
  filenameAnalysis,
  isAudio,
}: {
  metadata?: MetadataAnalysis;
  filenameAnalysis?: FilenameAnalysis;
  isAudio?: boolean;
}) {
  const camera = isAudio ? "Audio Recording" : (metadata?.camera_make || "Not embedded");
  const dateTaken = metadata?.creation_date || "Not embedded";
  const software = metadata?.software_tag || "None detected";
  const dimensions = isAudio ? "PCM Stream" : (metadata?.has_exif ? "3024 × 4032" : "Unknown");
  const location = metadata?.has_exif ? "Embedded (GPS present)" : "Not embedded";

  const rows: MetadataRow[] = [
    { label: isAudio ? "Audio Stream" : "Camera", value: camera, flag: false },
    { label: "Date Created", value: dateTaken, flag: dateTaken === "Not embedded" },
    { label: "Format / Specs", value: dimensions, flag: false },
    { label: "Location", value: location, flag: false },
    { label: "Software", value: software, flag: Boolean(metadata?.software_tag && metadata.software_tag !== "None detected") },
  ];

  const pType = filenameAnalysis?.pattern_type || "unrecognized";
  let pillClass = "bg-zinc-900 text-zinc-400 border border-white/10";
  let pillLabel = "UNRECOGNIZED";

  if (pType === "ai_generator") {
    pillClass = "bg-red-500/15 text-red-400 border border-red-500/30";
    pillLabel = "AI GENERATOR";
  } else if (pType === "camera_native") {
    pillClass = "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
    pillLabel = "CAMERA NATIVE";
  } else if (pType === "generic_download") {
    pillClass = "bg-zinc-900 text-zinc-400 border border-white/10";
    pillLabel = "GENERIC DOWNLOAD";
  }

  const filenameStr = filenameAnalysis?.filename || "upload.jpg";

  return (
    <div className="relative rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-5 shadow-xl shadow-black/80">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/40 bg-white/10 text-white">
          {isAudio ? <Volume2 className="h-4 w-4" /> : <Camera className="h-4 w-4" />}
        </div>
        <div>
          <span className="font-mono text-[9px] tracking-widest text-zinc-500 uppercase block">METADATA_INSPECTION</span>
          <h2 className="text-sm font-bold text-white uppercase font-sans">
            EXIF & Technical Metadata
          </h2>
        </div>
      </div>

      <div className="mt-4 space-y-2 font-mono text-xs">
        {/* Filename Row */}
        <div className="flex items-center justify-between rounded-xl border border-white/5 bg-zinc-950/60 px-3.5 py-2.5">
          <span className="text-[11px] text-zinc-500 uppercase">
            Filename
          </span>
          <div className="flex items-center gap-2 max-w-[65%] overflow-hidden">
            <span className="truncate text-xs font-semibold text-zinc-200" title={filenameStr}>
              {filenameStr}
            </span>
            <span className={`shrink-0 rounded px-2 py-0.5 text-[9px] font-bold tracking-wider ${pillClass}`}>
              {pillLabel}
            </span>
          </div>
        </div>

        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-center justify-between rounded-xl border border-white/5 bg-zinc-950/60 px-3.5 py-2.5"
          >
            <span className="text-[11px] text-zinc-500 uppercase">
              {row.label}
            </span>
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-200">
                {row.value}
              </span>
              {row.flag && (
                <span className="rounded bg-red-500/15 border border-red-500/30 px-2 py-0.5 text-[9px] font-bold text-red-400 uppercase">
                  Flag
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AudioSpectralStatsCard({
  audioAnalysis,
  deepfakeDetector,
}: {
  audioAnalysis?: AudioAnalysis;
  deepfakeDetector?: DeepfakeDetectorResult;
}) {
  const flatness = audioAnalysis?.spectral_flatness ?? 0.042;
  const pitchVar = audioAnalysis?.pitch_variance ?? 185.0;
  const hnr = audioAnalysis?.harmonic_to_noise_ratio ?? 14.2;
  const dfConf = deepfakeDetector?.deepfake_audio_confidence ?? null;
  const dfAvailable = dfConf !== null;
  const dfModelUsed = deepfakeDetector?.model_used || null;
  const dfNote = deepfakeDetector?.note || null;

  let dfPillClass = "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
  let dfPillLabel = "LIKELY REAL";
  let dfBarColor = "bg-emerald-400";

  if (!dfAvailable) {
    dfPillClass = "bg-zinc-900 text-zinc-500 border border-white/10";
    dfPillLabel = "UNAVAILABLE";
    dfBarColor = "bg-zinc-700";
  } else if (dfConf > 90) {
    dfPillClass = "bg-red-500/15 text-red-400 border border-red-500/30";
    dfPillLabel = "VERY LIKELY DEEPFAKE";
    dfBarColor = "bg-red-500";
  } else if (dfConf > 70) {
    dfPillClass = "bg-red-500/15 text-red-400 border border-red-500/30";
    dfPillLabel = "LIKELY SYNTHETIC";
    dfBarColor = "bg-red-500";
  } else if (dfConf > 45) {
    dfPillClass = "bg-zinc-800 text-zinc-200 border border-zinc-700";
    dfPillLabel = "UNCERTAIN";
    dfBarColor = "bg-zinc-300";
  }

  const stats = [
    { label: "Spectral Flatness", value: flatness.toFixed(4), flag: flatness > 0.07 },
    { label: "Pitch Variance", value: pitchVar.toFixed(1), flag: pitchVar < 140.0 },
    { label: "Harmonic-to-Noise", value: `${hnr.toFixed(1)} dB`, flag: hnr > 24.0 },
  ];

  return (
    <div className="relative rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-5 shadow-xl shadow-black/80">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-purple-500/40 bg-purple-500/10 text-purple-400">
            <AudioWaveform className="h-4 w-4" />
          </div>
          <div>
            <span className="font-mono text-[9px] tracking-widest text-zinc-500 uppercase block">AUDIO_PROSODY</span>
            <h2 className="text-sm font-bold text-white uppercase font-sans">
              Spectral & Acoustic Analysis
            </h2>
          </div>
        </div>
        <span className="rounded px-2.5 py-1 font-mono text-[10px] font-bold text-purple-400 bg-purple-500/10 border border-purple-500/30 uppercase">
          PROSODY SCAN
        </span>
      </div>

      {/* Deepfake detector confidence bar */}
      <div className="mt-4 rounded-xl border border-white/5 bg-zinc-950/60 p-3.5 font-mono">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Bot className="h-3.5 w-3.5 text-zinc-400" />
            <span className="text-xs font-semibold text-zinc-300">Deepfake Audio Classifier</span>
          </div>
          <span className={`rounded px-2 py-0.5 text-[9px] font-bold tracking-wider ${dfPillClass}`}>
            {dfPillLabel}
          </span>
        </div>
        {dfAvailable ? (
          <>
            <div className="mt-2.5 flex items-center justify-between text-[10px] text-zinc-400">
              <span>Synthetic Voice Confidence</span>
              <span className="font-bold text-white">{dfConf!.toFixed(1)}%</span>
            </div>
            <div className="mt-1.5 h-2 w-full overflow-hidden rounded-md bg-zinc-900 border border-white/5">
              <div
                className={`h-full rounded-sm transition-all duration-500 ${dfBarColor}`}
                style={{ width: `${dfConf}%` }}
              />
            </div>
            {dfModelUsed && (
              <p className="mt-1.5 text-[9px] text-zinc-600 font-mono">{dfModelUsed}</p>
            )}
          </>
        ) : (
          <p className="mt-1.5 text-[10px] text-zinc-500">
            {dfNote || "Add HF_API_TOKEN to .env to enable the deepfake audio detector."}
          </p>
        )}
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2 font-mono">
        {stats.map((s) => (
          <div key={s.label} className="rounded-xl border border-white/5 bg-zinc-950/60 p-3 text-center">
            <p className="text-[9px] text-zinc-500 uppercase">{s.label}</p>
            <p className="mt-1 text-xs font-bold text-white">{s.value}</p>
            {s.flag && (
              <span className="mt-1 inline-block rounded bg-red-500/15 border border-red-500/30 px-1.5 py-0.5 text-[8px] font-bold text-red-400 uppercase">
                SUSPICIOUS
              </span>
            )}
          </div>
        ))}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-zinc-400 font-mono">
        Acoustic prosody analysis measures fundamental frequency variations and spectral noise balance to evaluate potential neural text-to-speech or voice cloning artifacts.
      </p>
    </div>
  );
}

function ManipulationScanCard({
  score = 0,
  originalUrl,
  heatmapUrl,
  explanation = "Error-level analysis shows elevated noise around the subject edges, suggesting splicing or inpainting.",
}: {
  score?: number;
  originalUrl?: string | null;
  heatmapUrl?: string | null;
  explanation?: string;
}) {
  return (
    <div className="relative rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-5 shadow-xl shadow-black/80">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-purple-500/40 bg-purple-500/10 text-purple-400">
            <Search className="h-4 w-4" />
          </div>
          <div>
            <span className="font-mono text-[9px] tracking-widest text-zinc-500 uppercase block">ERROR_LEVEL_ANALYSIS</span>
            <h2 className="text-sm font-bold text-white uppercase font-sans">
              Compression & Noise Scan
            </h2>
          </div>
        </div>
        <span className="rounded px-2.5 py-1 font-mono text-[10px] font-bold text-white bg-white/10 border border-white/30 uppercase">
          {score}% SUSPICIOUS
        </span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 font-mono">
        <div className="overflow-hidden rounded-xl border border-white/10 bg-zinc-950">
          <div className="flex aspect-square items-center justify-center bg-zinc-900 overflow-hidden">
            {originalUrl ? (
              <img src={originalUrl} alt="Original media" className="h-full w-full object-cover" />
            ) : (
              <Image className="h-8 w-8 text-zinc-600" />
            )}
          </div>
          <p className="px-3 py-2 text-center text-[10px] text-zinc-400 uppercase border-t border-white/5">
            ORIGINAL_FRAME
          </p>
        </div>
        <div className="overflow-hidden rounded-xl border border-white/10 bg-zinc-950">
          <div className="flex aspect-square items-center justify-center bg-zinc-900 overflow-hidden">
            {heatmapUrl ? (
              <img src={heatmapUrl} alt="ELA Heatmap" className="h-full w-full object-cover" />
            ) : (
              <Eye className="h-8 w-8 text-white/60" />
            )}
          </div>
          <p className="px-3 py-2 text-center text-[10px] text-zinc-400 uppercase border-t border-white/5">
            ELA_HEATMAP
          </p>
        </div>
      </div>

      <p className="mt-3 text-xs leading-relaxed text-zinc-300">
        {explanation}
      </p>
    </div>
  );
}

function SourceTraceCard({ sourceTrace }: { sourceTrace?: SourceTrace }) {
  const found = sourceTrace?.found_matches ?? false;
  const domain = sourceTrace?.earliest_source?.domain || sourceTrace?.domain || "socialfeed.example.com";
  const date = sourceTrace?.earliest_source?.date_seen || sourceTrace?.date || "2023-09-01";
  const note = sourceTrace?.note || "reverse search unavailable";
  const url = sourceTrace?.earliest_source?.url || sourceTrace?.url || "#";

  return (
    <div className="relative rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-5 shadow-xl shadow-black/80">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-emerald-500/40 bg-emerald-500/10 text-emerald-400">
          <ExternalLink className="h-4 w-4" />
        </div>
        <div>
          <span className="font-mono text-[9px] tracking-widest text-zinc-500 uppercase block">REVERSE_INDEX</span>
          <h2 className="text-sm font-bold text-white uppercase font-sans">
            Source Trace & Index Matching
          </h2>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-white/5 bg-zinc-950/60 p-4 font-mono">
        {found ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded border border-white/10 bg-zinc-900 px-2.5 py-1 text-xs font-semibold text-white">
                {domain}
              </span>
              <span className="rounded border border-white/30 bg-white/15 px-2.5 py-1 text-xs font-semibold text-white">
                {date}
              </span>
              {sourceTrace?.total_matches ? (
                <span className="rounded border border-blue-500/30 bg-blue-500/15 px-2 py-0.5 text-[10px] font-semibold text-blue-400">
                  {sourceTrace.total_matches} MATCHES
                </span>
              ) : null}
            </div>
            <p className="mt-3 text-xs text-zinc-400">
              Earliest indexed web publication match located.
            </p>
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-white px-4 py-2 text-xs font-bold text-black uppercase tracking-wider transition-colors hover:bg-zinc-200"
            >
              VIEW ORIGINAL SOURCE
              <ExternalLink className="h-3 w-3" />
            </a>
          </>
        ) : (
          <p className="text-xs text-zinc-400 leading-relaxed font-sans">
            {note}
          </p>
        )}
      </div>
    </div>
  );
}

interface FlagItem {
  icon: LucideIcon;
  badgeClass: string;
  dotClass: string;
  label: string;
  desc: string;
}

function RedFlagsCard({ flags }: { flags?: RedFlagItem[] }) {
  const defaultFlags: FlagItem[] = [
    {
      icon: FileWarning,
      badgeClass: "bg-red-500/15 text-red-400 border border-red-500/30",
      dotClass: "bg-red-500",
      label: "Edited with Photoshop",
      desc: "Software tag detected in metadata",
    },
    {
      icon: ShieldAlert,
      badgeClass: "bg-white/15 text-white border border-white/30",
      dotClass: "bg-white",
      label: "Inconsistent timeline",
      desc: "Create date differs from upload date",
    },
    {
      icon: AlertTriangle,
      badgeClass: "bg-purple-500/15 text-purple-400 border border-purple-500/30",
      dotClass: "bg-purple-500",
      label: "No verified source",
      desc: "Could not trace to original publisher",
    },
  ];

  let displayFlags: FlagItem[] = defaultFlags;

  if (flags && flags.length > 0) {
    displayFlags = flags.map((item, i) => {
      let icon = FileWarning;
      let badgeClass = "bg-red-500/15 text-red-400 border border-red-500/30";
      let dotClass = "bg-red-500";

      if (item.type === "timeline" || i % 3 === 1) {
        icon = ShieldAlert;
        badgeClass = "bg-white/15 text-white border border-white/30";
        dotClass = "bg-white";
      } else if (item.type === "source" || i % 3 === 2) {
        icon = AlertTriangle;
        badgeClass = "bg-purple-500/15 text-purple-400 border border-purple-500/30";
        dotClass = "bg-purple-500";
      }

      return {
        icon,
        badgeClass,
        dotClass,
        label: item.label,
        desc: item.desc,
      };
    });
  }

  return (
    <div className="relative rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-5 shadow-xl shadow-black/80">
      <span className="font-mono text-[9px] tracking-widest text-zinc-500 uppercase block">DETECTION_BULLETS</span>
      <h2 className="text-sm font-bold text-white uppercase font-sans">
        Identified Red Flags
      </h2>

      <div className="mt-4 space-y-3">
        {displayFlags.map((flag) => {
          const Icon = flag.icon;
          return (
            <div key={flag.label} className="flex items-start gap-3 rounded-xl border border-white/5 bg-zinc-950/60 p-3">
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${flag.badgeClass}`}
              >
                <Icon className="h-4 w-4" />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-white">
                    {flag.label}
                  </span>
                  <span className={`h-1.5 w-1.5 rounded-full ${flag.dotClass}`} />
                </div>
                <p className="text-[11px] text-zinc-400 font-mono mt-0.5">{flag.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function LimitationsFooter({ text }: { text?: string }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-white/10 bg-zinc-950/80 p-4 font-mono">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-white" />
      <p className="text-[11px] leading-relaxed text-zinc-400">
        {text || "TrustLens uses AI-assisted signals and public metadata. Results are not legal evidence and should be cross-checked with primary sources."}
      </p>
    </div>
  );
}

function AiDetectorCard({ aiDetector }: { aiDetector?: AiDetectorResult }) {
  const confidence = aiDetector?.ai_generation_confidence ?? null;
  const available = confidence !== null;
  const modelUsed = aiDetector?.model_used || null;
  const note = aiDetector?.note || null;
  const perFrame = aiDetector?.per_frame_confidence ?? [];
  const framesAnalyzed = aiDetector?.frames_analyzed ?? 0;
  const framesFlagged = aiDetector?.frames_flagged_ai ?? 0;
  const isVideo = perFrame.length > 0;

  let pillClass = "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
  let pillLabel = isVideo ? "VIDEO: LIKELY REAL" : "LIKELY REAL";
  let barColor = "bg-emerald-400";

  if (!available) {
    pillClass = "bg-zinc-900 text-zinc-500 border border-white/10";
    pillLabel = "UNAVAILABLE";
    barColor = "bg-zinc-700";
  } else if (confidence > 90) {
    pillClass = "bg-red-500/15 text-red-400 border border-red-500/30";
    pillLabel = isVideo ? "VIDEO: VERY LIKELY AI" : "VERY LIKELY AI";
    barColor = "bg-red-500";
  } else if (confidence > 70) {
    pillClass = "bg-red-500/15 text-red-400 border border-red-500/30";
    pillLabel = isVideo ? "VIDEO: LIKELY AI" : "LIKELY AI";
    barColor = "bg-red-500";
  } else if (confidence > 45) {
    pillClass = "bg-zinc-800 text-zinc-200 border border-zinc-700";
    pillLabel = "UNCERTAIN";
    barColor = "bg-zinc-300";
  }

  return (
    <div className="relative rounded-2xl border border-white/10 bg-[#121212]/90 backdrop-blur-md p-5 shadow-xl shadow-black/80 font-mono">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-blue-500/40 bg-blue-500/10 text-blue-400">
            <Bot className="h-4 w-4" />
          </div>
          <div>
            <span className="font-mono text-[9px] tracking-widest text-zinc-500 uppercase block">NEURAL_CLASSIFIER</span>
            <h2 className="text-sm font-bold text-white uppercase font-sans">
              {isVideo ? "Video AI Keyframe Detector" : "AI Image Content Detector"}
            </h2>
          </div>
        </div>
        <span className={`rounded px-2.5 py-1 text-[9px] font-bold tracking-wider uppercase ${pillClass}`}>
          {pillLabel}
        </span>
      </div>

      <div className="mt-4">
        {available ? (
          <>
            {/* Max confidence bar */}
            <div className="mb-2 flex items-center justify-between text-[11px] text-zinc-400">
              <span>{isVideo ? "Max Frame AI Confidence" : "AI Generation Confidence"}</span>
              <span className="font-bold text-white">{confidence.toFixed(1)}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-md bg-zinc-950 border border-white/5">
              <div
                className={`h-full rounded-sm transition-all duration-500 ${barColor}`}
                style={{ width: `${confidence}%` }}
              />
            </div>

            {/* Per-frame breakdown for video */}
            {isVideo && (
              <div className="mt-3.5">
                <div className="mb-1.5 flex items-center justify-between text-[10px] text-zinc-400">
                  <span>Per-Frame breakdown ({perFrame.length}/{framesAnalyzed} keyframes)</span>
                  <span className={framesFlagged > 0 ? "font-bold text-red-400" : ""}>
                    {framesFlagged} frame{framesFlagged !== 1 ? "s" : ""} flagged
                  </span>
                </div>
                <div className="flex items-end gap-1.5 h-9 bg-zinc-950 p-1.5 rounded-lg border border-white/5">
                  {perFrame.map((fc, i) => {
                    const barBg =
                      fc > 90 ? "bg-red-500" :
                      fc > 70 ? "bg-red-400" :
                      fc > 45 ? "bg-zinc-300" : "bg-emerald-400";
                    const heightPct = Math.max(12, Math.round(fc * 0.28));
                    return (
                      <div key={i} className="flex flex-1 flex-col items-center gap-0.5">
                        <div
                          className={`w-full rounded-xs ${barBg} transition-all duration-500`}
                          style={{ height: `${heightPct}px` }}
                          title={`Frame ${i + 1}: ${fc.toFixed(1)}%`}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <p className="mt-3 text-xs leading-relaxed text-zinc-300 font-sans">
              {isVideo
                ? confidence > 70
                  ? `${framesFlagged} of ${perFrame.length} sampled keyframes exceeded the AI-generation threshold. Highest-confidence frame scored ${confidence.toFixed(1)}% — strong evidence of AI generation.`
                  : `All ${perFrame.length} sampled keyframes scored below the 70% threshold. No strong AI generation signal detected in video content.`
                : confidence > 70
                ? "This dedicated image classifier detected strong signals consistent with AI-generated imagery."
                : confidence > 45
                ? "The AI detector found some signals of possible AI generation, but the result is inconclusive."
                : "The AI detector found no strong signals of AI generation in the image content."}
            </p>

            {modelUsed && (
              <p className="mt-2 text-[9px] text-zinc-600 font-mono">
                MODEL: {modelUsed}
              </p>
            )}
          </>
        ) : (
          <div className="rounded-xl border border-white/5 bg-zinc-950/60 p-3">
            <p className="text-xs text-zinc-400 leading-relaxed font-sans">
              {note || "AI detector was unavailable for this request. Add HF_API_TOKEN to .env to enable this signal."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
