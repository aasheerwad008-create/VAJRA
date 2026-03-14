import { describe, it, expect, vi, beforeEach } from "vitest";
import { apiClient } from "@/lib/api";

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("apiClient", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("health() calls GET /health", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ status: "ok" }),
    });

    const result = await apiClient.health();
    expect(result).toEqual({ status: "ok" });
    expect(mockFetch).toHaveBeenCalledWith("http://localhost:8080/health");
  });

  it("verify() calls POST /api/verify with body", async () => {
    const response = {
      session_id: "abc",
      trust_score: 85,
      verdict: "VERIFIED",
      proof_hash: "0x123",
      timestamp: "2025-01-01T00:00:00Z",
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(response),
    });

    const result = await apiClient.verify({ user_id: "user1" });
    expect(result).toEqual(response);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/api/verify",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "user1" }),
      }),
    );
  });

  it("throws on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: () => Promise.resolve("Internal Server Error"),
    });

    await expect(apiClient.health()).rejects.toThrow("API /health → 500");
  });

  it("getSession() calls correct endpoint", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          session_id: "s1",
          status: "completed",
          created_at: "2025-01-01",
          updated_at: "2025-01-01",
        }),
    });

    const result = await apiClient.getSession("s1");
    expect(result.session_id).toBe("s1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/api/sessions/s1",
    );
  });
});
