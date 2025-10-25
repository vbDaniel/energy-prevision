import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  compiler: {
    styledComponents: {
      displayName: true,
      ssr: true,
      fileName: true,
      topLevelImportPaths: [],
      meaninglessFileNames: ["index", "styles"],
      cssProp: true,
      minify: false,
      transpileTemplateLiterals: false,
    },
  },
};

export default nextConfig;
