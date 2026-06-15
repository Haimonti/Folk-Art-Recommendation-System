import type { NextConfig } from "next";

const BACKEND =
  process.env.BACKEND_URL ||
  "https://gemi-backend-folkart-healthcare-frontend.apps.buffalo.edu";

const nextConfig: NextConfig = {
  devIndicators: false,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/images/:path*", destination: `${BACKEND}/images/:path*` },
    ];
  },
};
export default nextConfig;
