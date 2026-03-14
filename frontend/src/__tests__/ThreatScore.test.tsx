import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import ThreatScore from "@/components/ThreatScore";

// framer-motion renders fine in jsdom, no special mock needed

describe("ThreatScore", () => {
  it("renders the score value", () => {
    render(<ThreatScore score={85} verdict="VERIFIED" />);
    expect(screen.getByText("85")).toBeInTheDocument();
  });

  it("clamps score to 0–100 range", () => {
    render(<ThreatScore score={150} verdict="VERIFIED" />);
    // The axis label also shows "100", so use getAllByText
    const matches = screen.getAllByText("100");
    expect(matches.length).toBeGreaterThanOrEqual(2); // score text + axis label
  });

  it("shows VERIFIED verdict badge", () => {
    render(<ThreatScore score={90} verdict="VERIFIED" />);
    expect(screen.getByText("✓ VERIFIED")).toBeInTheDocument();
  });

  it("shows DEEPFAKE verdict badge", () => {
    render(<ThreatScore score={15} verdict="DEEPFAKE" />);
    expect(screen.getByText("✗ DEEPFAKE")).toBeInTheDocument();
  });

  it("shows SUSPICIOUS verdict badge", () => {
    render(<ThreatScore score={55} verdict="SUSPICIOUS" />);
    expect(screen.getByText("⚠ SUSPICIOUS")).toBeInTheDocument();
  });

  it("shows IDLE verdict for unknown state", () => {
    render(<ThreatScore score={0} verdict="IDLE" />);
    expect(screen.getByText("– IDLE")).toBeInTheDocument();
  });
});
