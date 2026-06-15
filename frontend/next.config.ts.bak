import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://gemi-backend-folkart-healthcare-frontend.apps.buffalo.edu/api/:path*",
      },
      {
        source: "/images/:path*",
        destination: "https://gemi-backend-folkart-healthcare-frontend.apps.buffalo.edu/images/:path*",
      },
    ];
  },
};

export default nextConfig;
