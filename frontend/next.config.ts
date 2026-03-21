import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
      {
        source: "/images/:path*",
        destination: "http://127.0.0.1:8000/images/:path*",
      },
    ];
  },
};

export default nextConfig;
