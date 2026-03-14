import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import StatusBar from "@/components/StatusBar";

describe("StatusBar", () => {
  it("shows STANDBY when not streaming", () => {
    render(<StatusBar isStreaming={false} audioLevel={0} />);
    expect(screen.getByText("STANDBY")).toBeInTheDocument();
  });

  it("shows LIVE when streaming", () => {
    render(<StatusBar isStreaming={true} audioLevel={0.5} />);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
  });

  it("renders layer indicators L1, L2, L3", () => {
    render(<StatusBar isStreaming={false} audioLevel={0} />);
    expect(screen.getByText("L1")).toBeInTheDocument();
    expect(screen.getByText("L2")).toBeInTheDocument();
    expect(screen.getByText("L3")).toBeInTheDocument();
  });
});
