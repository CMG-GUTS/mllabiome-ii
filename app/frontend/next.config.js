module.exports = {
  reactStrictMode: true,
  swcMinify: true,
  distDir: process.env.BUILD_DIR || '.next',
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    CLIENT_ID: process.env.CLIENT_ID,
    NEXT_PUBLIC_SERVER_URL: process.env.SERVER_URL || 'http://localhost:8000',
  },
}
