/**
 * VAJRA Blockchain — Contract verification script for Polygon Amoy.
 *
 * Verifies both KavachaTrustRegistry and VajraTrustRegistry on Polygonscan
 * after deployment.  Requires:
 *   - POLYGONSCAN_API_KEY in environment / .env
 *   - KAVACH_REGISTRY_ADDRESS and VAJRA_REGISTRY_ADDRESS to be set
 *     (populated automatically when running after deploy.js in CI)
 *
 * Usage:
 *   npx hardhat run scripts/verify.js --network amoy
 *
 *   Or with explicit addresses:
 *   KAVACH_REGISTRY_ADDRESS=0x... npx hardhat run scripts/verify.js --network amoy
 */

const { run, network } = require("hardhat");

async function verifyContract(contractName, address, constructorArgs = []) {
  console.log(`\nVerifying ${contractName} at ${address} on ${network.name}...`);

  if (!address || address === "0x0000000000000000000000000000000000000000") {
    console.warn(`  ⚠  Skipping ${contractName}: no address provided.`);
    return false;
  }

  try {
    await run("verify:verify", {
      address,
      constructorArguments: constructorArgs,
    });
    console.log(`  ✅ ${contractName} verified on Polygonscan.`);
    return true;
  } catch (err) {
    if (err.message.includes("Already Verified")) {
      console.log(`  ℹ  ${contractName} is already verified.`);
      return true;
    }
    console.error(`  ❌ Verification failed for ${contractName}:`, err.message);
    return false;
  }
}

async function main() {
  const kavachaAddress = process.env.KAVACH_REGISTRY_ADDRESS || "";
  const vajraAddress   = process.env.VAJRA_REGISTRY_ADDRESS  || "";

  if (!kavachaAddress && !vajraAddress) {
    console.error(
      "Error: Set KAVACH_REGISTRY_ADDRESS and/or VAJRA_REGISTRY_ADDRESS before running."
    );
    process.exitCode = 1;
    return;
  }

  console.log("=== VAJRA Contract Verification ===");
  console.log("Network:", network.name);

  const results = await Promise.all([
    verifyContract("KavachaTrustRegistry", kavachaAddress),
    verifyContract("VajraTrustRegistry",   vajraAddress),
  ]);

  const allPassed = results.every(Boolean);
  console.log("\n=== Verification Summary ===");
  console.log("KavachaTrustRegistry:", kavachaAddress || "skipped");
  console.log("VajraTrustRegistry:  ", vajraAddress   || "skipped");
  console.log("Status:", allPassed ? "✅ All verified" : "⚠  Some failed — check output above");

  if (!allPassed) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
