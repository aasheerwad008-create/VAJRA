// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title KavachaTrustRegistry
 * @notice On-chain immutable audit registry for VAJRA identity verification proofs.
 *         "Kavach" (कवच) means armour in Sanskrit — the protective layer of the
 *         VAJRA zero-trust identity defense platform.
 *
 * Layer 3 — Blockchain Trust Registry
 * ─────────────────────────────────────
 * Stores ZK proof hashes anchored to verified identities on Polygon Amoy.
 * Each verification record is append-only and timestamped.
 *
 * Events:
 *   IdentityVerified     — emitted on successful verification
 *   FraudAttemptDetected — emitted when a suspicious/deepfake verdict is anchored
 */
contract KavachaTrustRegistry is Ownable, Pausable {

    // ── Structs ────────────────────────────────────────────────────────────

    struct VerificationRecord {
        bytes32 proofHash;      // SHA-256 ZK proof commitment
        bytes32 txHash;         // Reference TX hash (off-chain context)
        uint256 timestamp;
        bool    verified;
        string  verdict;        // VERIFIED | SUSPICIOUS | DEEPFAKE
    }

    // ── State ──────────────────────────────────────────────────────────────

    /// @dev Maps identity commitment (SHA-256 of user_id:nullifier) → records
    mapping(bytes32 => VerificationRecord[]) private _records;

    /// @dev Authorized verifier addresses (VAJRA backend)
    mapping(address => bool) public verifiers;

    uint256 public totalVerifications;
    uint256 public totalFraudAttempts;

    /// @dev Registry version for upgrade tracking
    string public constant VERSION = "1.0.0";

    // ── Events ─────────────────────────────────────────────────────────────

    event IdentityVerified(
        bytes32 indexed identityCommitment,
        bytes32 proofHash,
        uint256 timestamp
    );

    event FraudAttemptDetected(
        bytes32 indexed identityCommitment,
        bytes32 proofHash,
        string  verdict,
        uint256 timestamp
    );

    event VerifierAdded(address indexed verifier);
    event VerifierRemoved(address indexed verifier);

    // ── Modifiers ──────────────────────────────────────────────────────────

    modifier onlyVerifier() {
        require(
            verifiers[msg.sender] || msg.sender == owner(),
            "KavachaTrustRegistry: not authorized"
        );
        _;
    }

    // ── Constructor ────────────────────────────────────────────────────────

    constructor() Ownable(msg.sender) {
        verifiers[msg.sender] = true;
    }

    // ── Admin ──────────────────────────────────────────────────────────────

    function addVerifier(address verifier) external onlyOwner {
        verifiers[verifier] = true;
        emit VerifierAdded(verifier);
    }

    function removeVerifier(address verifier) external onlyOwner {
        verifiers[verifier] = false;
        emit VerifierRemoved(verifier);
    }

    function pause() external onlyOwner { _pause(); }
    function unpause() external onlyOwner { _unpause(); }

    // ── Core ───────────────────────────────────────────────────────────────

    /**
     * @notice Anchor a verification result to the Kavach trust registry.
     * @param identityCommitment SHA-256(user_id:nullifier) — no raw PII on-chain
     * @param proofHash          SHA-256 ZK proof commitment from zk-proof-system
     * @param txRefHash          Reference hash for off-chain lookup
     * @param verified           True if identity passed all checks
     * @param verdict            "VERIFIED" | "SUSPICIOUS" | "DEEPFAKE"
     */
    function anchorVerification(
        bytes32 identityCommitment,
        bytes32 proofHash,
        bytes32 txRefHash,
        bool    verified,
        string  calldata verdict
    ) external onlyVerifier whenNotPaused {
        VerificationRecord memory record = VerificationRecord({
            proofHash:  proofHash,
            txHash:     txRefHash,
            timestamp:  block.timestamp,
            verified:   verified,
            verdict:    verdict
        });

        _records[identityCommitment].push(record);

        if (verified) {
            totalVerifications++;
            emit IdentityVerified(identityCommitment, proofHash, block.timestamp);
        } else {
            totalFraudAttempts++;
            emit FraudAttemptDetected(
                identityCommitment,
                proofHash,
                verdict,
                block.timestamp
            );
        }
    }

    // ── Views ──────────────────────────────────────────────────────────────

    /**
     * @notice Get the latest verification record for an identity.
     */
    function getLatestRecord(bytes32 identityCommitment)
        external
        view
        returns (VerificationRecord memory)
    {
        VerificationRecord[] storage records = _records[identityCommitment];
        require(records.length > 0, "KavachaTrustRegistry: no records found");
        return records[records.length - 1];
    }

    /**
     * @notice Get all records for an identity.
     */
    function getAllRecords(bytes32 identityCommitment)
        external
        view
        returns (VerificationRecord[] memory)
    {
        return _records[identityCommitment];
    }

    /**
     * @notice Check if the latest verification was successful.
     */
    function isVerified(bytes32 identityCommitment) external view returns (bool) {
        VerificationRecord[] storage records = _records[identityCommitment];
        if (records.length == 0) return false;
        return records[records.length - 1].verified;
    }

    /**
     * @notice Return the total number of records for an identity.
     */
    function recordCount(bytes32 identityCommitment) external view returns (uint256) {
        return _records[identityCommitment].length;
    }
}
