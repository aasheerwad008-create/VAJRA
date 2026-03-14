const { expect } = require("chai");
const { ethers } = require("hardhat");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");

describe("KavachaTrustRegistry", function () {
  let registry;
  let owner;
  let verifier;
  let other;

  beforeEach(async function () {
    [owner, verifier, other] = await ethers.getSigners();
    const KavachaTrustRegistry = await ethers.getContractFactory("KavachaTrustRegistry");
    registry = await KavachaTrustRegistry.deploy();
    await registry.waitForDeployment();
  });

  // ── Deployment ─────────────────────────────────────────────────────────

  it("deploys and sets owner as verifier", async function () {
    expect(await registry.verifiers(owner.address)).to.equal(true);
  });

  it("exposes the correct VERSION string", async function () {
    expect(await registry.VERSION()).to.equal("1.0.0");
  });

  // ── Verifier management ────────────────────────────────────────────────

  it("allows owner to add verifiers", async function () {
    await registry.addVerifier(verifier.address);
    expect(await registry.verifiers(verifier.address)).to.equal(true);
  });

  it("allows owner to remove verifiers", async function () {
    await registry.addVerifier(verifier.address);
    await registry.removeVerifier(verifier.address);
    expect(await registry.verifiers(verifier.address)).to.equal(false);
  });

  it("emits VerifierAdded when a verifier is added", async function () {
    await expect(registry.addVerifier(verifier.address))
      .to.emit(registry, "VerifierAdded")
      .withArgs(verifier.address);
  });

  it("emits VerifierRemoved when a verifier is removed", async function () {
    await registry.addVerifier(verifier.address);
    await expect(registry.removeVerifier(verifier.address))
      .to.emit(registry, "VerifierRemoved")
      .withArgs(verifier.address);
  });

  // ── Anchoring (success path) ───────────────────────────────────────────

  it("anchors a verified identity and emits IdentityVerified", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("user-123:nonce-abc"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("proof-xyz"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-ref"));

    await expect(
      registry.anchorVerification(commitment, proofHash, txRef, true, "VERIFIED")
    )
      .to.emit(registry, "IdentityVerified")
      .withArgs(commitment, proofHash, anyValue);

    const record = await registry.getLatestRecord(commitment);
    expect(record.verified).to.equal(true);
    expect(record.verdict).to.equal("VERIFIED");
  });

  it("increments totalVerifications on successful anchor", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("user-1:nonce"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("proof-1"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-1"));

    await registry.anchorVerification(commitment, proofHash, txRef, true, "VERIFIED");
    expect(await registry.totalVerifications()).to.equal(1n);
  });

  // ── Anchoring (fraud path) ─────────────────────────────────────────────

  it("anchors a deepfake and emits FraudAttemptDetected", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("bad-actor:nonce"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("fake-proof"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-ref-2"));

    await expect(
      registry.anchorVerification(commitment, proofHash, txRef, false, "DEEPFAKE")
    ).to.emit(registry, "FraudAttemptDetected");

    expect(await registry.isVerified(commitment)).to.equal(false);
    expect(await registry.totalFraudAttempts()).to.equal(1n);
  });

  it("anchors a suspicious verdict and increments totalFraudAttempts", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("suspicious-user:nonce"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("proof-susp"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-susp"));

    await registry.anchorVerification(commitment, proofHash, txRef, false, "SUSPICIOUS");
    expect(await registry.totalFraudAttempts()).to.equal(1n);

    const record = await registry.getLatestRecord(commitment);
    expect(record.verdict).to.equal("SUSPICIOUS");
  });

  // ── Record retrieval ───────────────────────────────────────────────────

  it("accumulates multiple records for the same identity", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("multi-user:nonce"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-multi"));

    await registry.anchorVerification(
      commitment,
      ethers.keccak256(ethers.toUtf8Bytes("proof-1")),
      txRef, true, "VERIFIED"
    );
    await registry.anchorVerification(
      commitment,
      ethers.keccak256(ethers.toUtf8Bytes("proof-2")),
      txRef, false, "SUSPICIOUS"
    );

    const all = await registry.getAllRecords(commitment);
    expect(all.length).to.equal(2);
    expect(await registry.recordCount(commitment)).to.equal(2n);
  });

  it("getLatestRecord returns the most recent entry", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("seq-user:nonce"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-seq"));

    await registry.anchorVerification(
      commitment,
      ethers.keccak256(ethers.toUtf8Bytes("proof-old")),
      txRef, true, "VERIFIED"
    );
    await registry.anchorVerification(
      commitment,
      ethers.keccak256(ethers.toUtf8Bytes("proof-new")),
      txRef, false, "DEEPFAKE"
    );

    const latest = await registry.getLatestRecord(commitment);
    expect(latest.verdict).to.equal("DEEPFAKE");
    expect(latest.verified).to.equal(false);
  });

  // ── Access control ─────────────────────────────────────────────────────

  it("reverts for non-verifier callers", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("user-999:nonce"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("proof"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx"));

    await expect(
      registry
        .connect(other)
        .anchorVerification(commitment, proofHash, txRef, true, "VERIFIED")
    ).to.be.revertedWith("KavachaTrustRegistry: not authorized");
  });

  it("allows a newly added verifier to anchor", async function () {
    await registry.addVerifier(verifier.address);

    const commitment = ethers.keccak256(ethers.toUtf8Bytes("user-v2:nonce"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("proof-v2"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-v2"));

    await expect(
      registry
        .connect(verifier)
        .anchorVerification(commitment, proofHash, txRef, true, "VERIFIED")
    ).to.emit(registry, "IdentityVerified");
  });

  // ── Pause ──────────────────────────────────────────────────────────────

  it("reverts when paused", async function () {
    await registry.pause();

    const commitment = ethers.keccak256(ethers.toUtf8Bytes("paused-user:nonce"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("proof-paused"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-paused"));

    await expect(
      registry.anchorVerification(commitment, proofHash, txRef, true, "VERIFIED")
    ).to.be.revertedWithCustomError(registry, "EnforcedPause");
  });

  it("resumes after unpause", async function () {
    await registry.pause();
    await registry.unpause();

    const commitment = ethers.keccak256(ethers.toUtf8Bytes("resumed-user:nonce"));
    const proofHash  = ethers.keccak256(ethers.toUtf8Bytes("proof-r"));
    const txRef      = ethers.keccak256(ethers.toUtf8Bytes("tx-r"));

    await expect(
      registry.anchorVerification(commitment, proofHash, txRef, true, "VERIFIED")
    ).to.emit(registry, "IdentityVerified");
  });

  // ── Edge cases ─────────────────────────────────────────────────────────

  it("reverts on getLatestRecord when no records exist", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("unknown-user"));
    await expect(registry.getLatestRecord(commitment)).to.be.revertedWith(
      "KavachaTrustRegistry: no records found"
    );
  });

  it("isVerified returns false when no records exist", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("empty-user"));
    expect(await registry.isVerified(commitment)).to.equal(false);
  });

  it("recordCount returns 0 for unknown identity", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("ghost-user"));
    expect(await registry.recordCount(commitment)).to.equal(0n);
  });
});
