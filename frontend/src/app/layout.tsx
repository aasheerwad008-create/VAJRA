import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VAJRA — Zero-Trust AI Identity Defense",
  description:
    "Real-time AI voice cloning & deepfake defense platform with ZK proofs and blockchain attestation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-vajra-900 text-white font-mono antialiased">
        {children}
      </body>
    </html>
  );
}
