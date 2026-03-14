const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("VajraTrustRegistry", function () {
  let registry;
  let owner;
  let verifier;
  let other;

  beforeEach(async function () {
    [owner, verifier, other] = await ethers.getSigners();
    const VajraTrustRegistry = await ethers.getContractFactory("VajraTrustRegistry");
    registry = await VajraTrustRegistry.deploy();
    await registry.waitForDeployment();
  });

  it("deploys and sets owner as verifier", async function () {
    expect(await registry.verifiers(owner.address)).to.equal(true);
  });

  it("allows owner to add verifiers", async function () {
    await registry.addVerifier(verifier.address);
    expect(await registry.verifiers(verifier.address)).to.equal(true);
  });

  it("anchors a verified identity and emits IdentityVerified", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("user-123:nonce-abc"));
    const proofHash = ethers.keccak256(ethers.toUtf8Bytes("proof-xyz"));
    const txRef = ethers.keccak256(ethers.toUtf8Bytes("tx-ref"));

    await expect(
      registry.anchorVerification(commitment, proofHash, txRef, true, "VERIFIED")
    )
      .to.emit(registry, "IdentityVerified")
      .withArgs(commitment, proofHash, await ethers.provider.getBlock("latest").then((b) => b.timestamp + 1));

    const record = await registry.getLatestRecord(commitment);
    expect(record.verified).to.equal(true);
    expect(record.verdict).to.equal("VERIFIED");
  });

  it("anchors a deepfake and emits FraudAttemptDetected", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("bad-actor:nonce"));
    const proofHash = ethers.keccak256(ethers.toUtf8Bytes("fake-proof"));
    const txRef = ethers.keccak256(ethers.toUtf8Bytes("tx-ref-2"));

    await expect(
      registry.anchorVerification(commitment, proofHash, txRef, false, "DEEPFAKE")
    ).to.emit(registry, "FraudAttemptDetected");

    expect(await registry.isVerified(commitment)).to.equal(false);
    expect(await registry.totalFraudAttempts()).to.equal(1n);
  });

  it("reverts for non-verifier callers", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("user-999:nonce"));
    const proofHash = ethers.keccak256(ethers.toUtf8Bytes("proof"));
    const txRef = ethers.keccak256(ethers.toUtf8Bytes("tx"));

    await expect(
      registry
        .connect(other)
        .anchorVerification(commitment, proofHash, txRef, true, "VERIFIED")
    ).to.be.revertedWith("VajraTrustRegistry: not authorized");
  });

  it("reverts on getLatestRecord when no records exist", async function () {
    const commitment = ethers.keccak256(ethers.toUtf8Bytes("unknown-user"));
    await expect(registry.getLatestRecord(commitment)).to.be.revertedWith(
      "VajraTrustRegistry: no records found"
    );
  });
});
