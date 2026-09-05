interface StateToggleProps {
  current: "upload" | "loading" | "results";
  onChange: (state: "upload" | "loading" | "results") => void;
}

export function StateToggle({ current, onChange }: StateToggleProps) {
  const states: { value: "upload" | "loading" | "results"; label: string }[] = [
    { value: "upload", label: "01 // UPLOAD" },
    { value: "loading", label: "02 // LOADING" },
    { value: "results", label: "03 // RESULTS" },
  ];

  return (
    <div className="relative flex items-center gap-1 rounded-xl border border-white/10 bg-zinc-950/90 p-1.5 shadow-xl shadow-black/80 backdrop-blur-md font-mono text-xs">
      {states.map((s) => {
        const active = current === s.value;
        return (
          <button
            key={s.value}
            onClick={() => onChange(s.value)}
            className={`relative z-10 rounded-lg px-3.5 py-1.5 font-mono text-[11px] font-bold uppercase tracking-wider transition-all duration-200 ${
              active
                ? "bg-white text-black shadow-md shadow-white/25 scale-[1.02]"
                : "text-zinc-500 hover:text-zinc-200 hover:bg-white/5"
            }`}
          >
            {s.label}
          </button>
        );
      })}
    </div>
  );
}
