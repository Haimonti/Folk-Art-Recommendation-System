import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://gemi-backend-eftn.onrender.com/api/:path*",
      },
      {
        source: "/images/:path*",
        destination: "https://gemi-backend-eftn.onrender.com/images/:path*",
      },
    ];
  },
};

export default nextConfig;
