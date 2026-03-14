const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying contracts with account:", deployer.address);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", ethers.formatEther(balance), "MATIC");

  // ── Deploy KavachaTrustRegistry (primary) ────────────────────────────────
  const KavachaTrustRegistry = await ethers.getContractFactory("KavachaTrustRegistry");
  const kavachaRegistry = await KavachaTrustRegistry.deploy();
  await kavachaRegistry.waitForDeployment();

  const kavachaAddress = await kavachaRegistry.getAddress();
  console.log("KavachaTrustRegistry deployed to:", kavachaAddress);
  console.log("Set TRUST_REGISTRY_ADDRESS=" + kavachaAddress + " in your .env");

  // ── Deploy VajraTrustRegistry (legacy / reference) ───────────────────────
  const VajraTrustRegistry = await ethers.getContractFactory("VajraTrustRegistry");
  const vajraRegistry = await VajraTrustRegistry.deploy();
  await vajraRegistry.waitForDeployment();

  const vajraAddress = await vajraRegistry.getAddress();
  console.log("VajraTrustRegistry deployed to:", vajraAddress);

  console.log("\n=== Deployment Summary ===");
  console.log("KavachaTrustRegistry:", kavachaAddress);
  console.log("VajraTrustRegistry:  ", vajraAddress);
  console.log("Network:              Polygon Amoy (chain ID 80002)");
  console.log("Deployer:            ", deployer.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
