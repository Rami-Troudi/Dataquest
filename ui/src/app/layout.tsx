import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DataQuest Insurance Recommender",
  description: "Phase II Inference UI",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
