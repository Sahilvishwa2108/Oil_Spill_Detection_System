import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://oil-spill-detection.vercel.app'),
  title: "Oil Spill Detection System | AI-Powered Environmental Protection",
  description: "Revolutionary deep learning system combining U-Net and DeepLabV3+ models for real-time satellite image analysis and environmental threat detection.",
  keywords: ["oil spill detection", "AI", "machine learning", "environmental protection", "satellite imagery", "deep learning"],
  authors: [{ name: "Sahil Vishwakarma" }],
  creator: "Sahil Vishwakarma",
  openGraph: {
    title: "Oil Spill Detection System",
    description: "AI-Powered Environmental Protection with Deep Learning",
    url: "https://oil-spill-detection.vercel.app",
    siteName: "Oil Spill Detection System",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Oil Spill Detection System",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Oil Spill Detection System",
    description: "AI-Powered Environmental Protection with Deep Learning",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' }
  ]
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange={false}
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
