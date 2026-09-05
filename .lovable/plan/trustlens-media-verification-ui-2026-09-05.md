# TrustLens — Media Verification UI

## What we're building
A single-page React app called **TrustLens** that verifies media (images/video) and reports trust, metadata, manipulation, source, and red flags. Three states render conditionally in one route: **Upload**, **Loading**, and **Results**. Dummy data is used throughout; no backend.

## Design direction
- Soft off-white background (`#FAFAF8` / warm gray) with a large, blurred mint gradient blob behind the main card.
- White cards with `rounded-2xl`, soft shadow (`shadow-lg` / `shadow-xl` with low opacity), and generous padding.
- Pastel circular icon badges: blue, purple, orange, red, green tints.
- Rounded-full pill tags and buttons.
- Clean sans-serif typography using the project's default font stack (no custom web font unless requested).
- Layout is mobile-first, centered, max-width ~480px, suitable for the phone-like density shown in the reference.

## State machine
A single `appState` controls what is rendered:

1. **upload** — default on first load.
2. **loading** — triggered by clicking Analyze or dropping a file.
3. **results** — triggered after a short simulated delay.

A small developer/debug toggle (or buttons) lets the user jump between states to preview all three without routing.

## Components to create

### `src/routes/index.tsx`
The single page. Holds `appState` and renders the active state component.

### `src/components/trustlens/UploadCard.tsx`
- Centered white card.
- Drag-drop zone with dashed border and upload/cloud icon.
- URL input field with rounded-full styling.
- Rounded-full "Analyze" button (primary).
- On click/drop, transition to `loading`.

### `src/components/trustlens/LoadingState.tsx`
- Skeleton cards (score card, metadata, manipulation, source, red flags).
- Cycling status text: "Scanning metadata...", "Checking manipulation signals...", "Tracing source...", "Compiling report...".
- Auto-advances to `results` after ~3 seconds.

### `src/components/trustlens/ResultsView.tsx`
Stack of cards:

1. **TrustScoreCard** — large trust score number (e.g., 42/100), colored verdict pill ("Likely Manipulated" in red/orange), short explanation text.
2. **MetadataAnalysisCard** — camera icon badge, key-value rows (camera model, date taken, location, dimensions), red flag pills for suspicious fields.
3. **ManipulationScanCard** — two thumbnails side by side (original vs heatmap/error-level), score pill.
4. **SourceTraceCard** — domain/date pill, "View original" button, or empty-state text if none.
5. **RedFlagsCard** — icon rows with warning dots and short labels.
6. **LimitationsFooter** — small muted text about AI-assisted analysis.
7. **"Analyze another"** rounded-full button resets to `upload`.

### `src/components/trustlens/StateToggle.tsx`
Small segmented control or button row to switch between upload/loading/results for demo purposes.

## Styling updates
- Update `src/styles.css` to add any needed semantic tokens (mint blob color, trust-score colors: red/orange/green/blue).
- Keep all colors in `oklch` per project convention.
- No hardcoded hex utilities in components; use CSS variables / Tailwind tokens.

## Assets
- Use Lucide icons for all iconography (cloud upload, camera, warning, search, file, etc.).
- Placeholder thumbnails: generate one or two abstract/image-like placeholders under `src/assets/` for the manipulation scan card, or use colored placeholder divs if generation is not needed.

## Acceptance criteria
- `/` renders the TrustLens UI, not the placeholder.
- All three states are visible via the demo toggle.
- Results state shows all six cards with dummy data.
- No build errors; responsive and visually consistent with the reference.
- No routing or backend code added.
