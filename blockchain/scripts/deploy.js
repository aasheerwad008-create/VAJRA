const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying VajraTrustRegistry with account:", deployer.address);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", ethers.formatEther(balance), "MATIC");

  const VajraTrustRegistry = await ethers.getContractFactory("VajraTrustRegistry");
  const registry = await VajraTrustRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log("VajraTrustRegistry deployed to:", address);
  console.log("Set TRUST_REGISTRY_ADDRESS=" + address + " in your .env");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
