import { describe, test, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UploadCard } from "../UploadCard";
import { LoadingState } from "../LoadingState";
import { ResultsView, type AnalysisResponse } from "../ResultsView";

describe("TrustLens Frontend Component Suite", () => {
  test("1. Upload card button click triggers onAnalyze callback with input payload", async () => {
    const mockOnAnalyze = vi.fn();
    render(<UploadCard onAnalyze={mockOnAnalyze} />);

    const urlInput = screen.getByPlaceholderText("https://example.com/media.mp4");
    const submitButton = screen.getByRole("button", { name: /RUN FORENSIC ANALYSIS/i });

    // Type sample URL into input
    await userEvent.type(urlInput, "https://example.com/test_sample.mp4");
    fireEvent.click(submitButton);

    // Verify callback was triggered with the URL
    expect(mockOnAnalyze).toHaveBeenCalledTimes(1);
    expect(mockOnAnalyze).toHaveBeenCalledWith(undefined, "https://example.com/test_sample.mp4");
  });

  test("2. Loading state renders progress title and forensic scanning checklist", () => {
    render(<LoadingState isBackendReady={false} />);

    // Verify multi-stage scanner title & engine header render
    expect(screen.getByText(/Deep Media Forensics in Progress/i)).toBeInTheDocument();
    expect(screen.getByText(/FORENSIC_ENGINE \/\/ PIPELINE_EXEC/i)).toBeInTheDocument();
    expect(screen.getByText(/Reading file metadata\.\.\./i)).toBeInTheDocument();
  });

  test("3. Mocked successful response renders results cards and trust score", () => {
    const mockData: AnalysisResponse = {
      trust_score: 85,
      verdict: "Likely Authentic",
      explanation: "Media verification completed cleanly with consistent EXIF headers.",
      red_flags: [
        { label: "Edited with Photoshop", desc: "Software tag in metadata", type: "editing" },
      ],
      limitations: "TrustLens uses AI-assisted signals and public metadata.",
      metadata_analysis: {
        has_exif: true,
        camera_make: "Canon EOS R5",
        software_tag: "Adobe Photoshop",
        creation_date: "2026-09-05",
        suspicious_flags: [],
      },
      filename_analysis: {
        filename: "sample_photo.jpg",
        pattern_type: "camera_native",
        note: "Matches camera hardware convention",
      },
      ela_analysis: {
        score: 12,
        heatmap_url: "http://localhost:8000/static/heatmaps/ela_test.png",
        original_image_url: "http://localhost:8000/static/heatmaps/orig_test.jpg",
      },
      source_trace: {
        found_matches: false,
        note: "no web index matches",
      },
      ai_detector: {
        ai_generation_confidence: 12.5,
        model_used: "organika/sdxl-detector",
      },
    };

    const mockReset = vi.fn();
    render(<ResultsView data={mockData} onReset={mockReset} />);

    // Verify Trust Score card number & verdict
    expect(screen.getByText("85")).toBeInTheDocument();
    expect(screen.getByText("/100")).toBeInTheDocument();
    expect(screen.getByText("Likely Authentic")).toBeInTheDocument();

    // Verify Metadata and Compression card section headers
    expect(screen.getByText("EXIF & Technical Metadata")).toBeInTheDocument();
    expect(screen.getByText("Compression & Noise Scan")).toBeInTheDocument();

    // Verify Red flags section
    expect(screen.getByText("Identified Red Flags")).toBeInTheDocument();
    expect(screen.getByText("Edited with Photoshop")).toBeInTheDocument();

    // Verify CTA button triggers reset
    const analyzeAnotherButton = screen.getByRole("button", { name: /ANALYZE ANOTHER MEDIA FILE/i });
    fireEvent.click(analyzeAnotherButton);
    expect(mockReset).toHaveBeenCalledTimes(1);
  });
});
