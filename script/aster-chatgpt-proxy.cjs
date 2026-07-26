#!/usr/bin/env node
/**
 * aster-chatgpt-proxy.js
 *
 * HTTPS reverse proxy that wraps openai-oauth (http://127.0.0.1:10531)
 * and exposes it as https://127.0.0.1:10532
 *
 * Uses a self-signed certificate for localhost. Aster registers this
 * HTTPS URL as a custom endpoint so Warp's server accepts it.
 *
 * Usage: node aster-chatgpt-proxy.js [--generate-cert]
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const os = require('os');

const UPSTREAM_HOST = '127.0.0.1';
const UPSTREAM_PORT = 10531;
const PROXY_PORT = 10532;
const CERT_DIR = path.join(os.homedir(), '.aster');
const CERT_PATH = path.join(CERT_DIR, 'localhost.crt');
const KEY_PATH = path.join(CERT_DIR, 'localhost.key');

function ensureCert() {
  if (fs.existsSync(CERT_PATH) && fs.existsSync(KEY_PATH)) {
    return;
  }
  fs.mkdirSync(CERT_DIR, { recursive: true });
  console.log('Generating self-signed certificate for localhost...');
  execSync(
    `openssl req -x509 -newkey rsa:2048 -keyout "${KEY_PATH}" -out "${CERT_PATH}" ` +
    `-days 3650 -nodes -subj "/CN=localhost" ` +
    `-addext "subjectAltName=IP:127.0.0.1,DNS:localhost"`,
    { stdio: 'pipe' }
  );
  console.log(`Certificate written to ${CERT_PATH}`);

  // Trust the cert in macOS login keychain so apps accept it.
  try {
    execSync(
      `security add-trusted-cert -d -r trustRoot -k ~/Library/Keychains/login.keychain-db "${CERT_PATH}"`,
      { stdio: 'pipe' }
    );
    console.log('Certificate trusted in login keychain.');
  } catch (e) {
    console.warn('Could not auto-trust certificate. You may need to trust it manually:', CERT_PATH);
  }
}

function startProxy() {
  ensureCert();

  const options = {
    key: fs.readFileSync(KEY_PATH),
    cert: fs.readFileSync(CERT_PATH),
  };

  const server = https.createServer(options, (req, res) => {
    const proxyOptions = {
      hostname: UPSTREAM_HOST,
      port: UPSTREAM_PORT,
      path: req.url,
      method: req.method,
      headers: {
        ...req.headers,
        host: `${UPSTREAM_HOST}:${UPSTREAM_PORT}`,
      },
    };

    const proxyReq = http.request(proxyOptions, (proxyRes) => {
      // Add CORS headers.
      res.writeHead(proxyRes.statusCode, {
        ...proxyRes.headers,
        'access-control-allow-origin': '*',
      });
      proxyRes.pipe(res);
    });

    proxyReq.on('error', (err) => {
      res.writeHead(502);
      res.end(JSON.stringify({ error: `Upstream error: ${err.message}. Is openai-oauth running? Run: npx openai-oauth` }));
    });

    req.pipe(proxyReq);
  });

  server.listen(PROXY_PORT, '127.0.0.1', () => {
    console.log(`Aster ChatGPT HTTPS proxy ready at https://127.0.0.1:${PROXY_PORT}/v1`);
    console.log(`Upstream: http://${UPSTREAM_HOST}:${UPSTREAM_PORT}`);
    console.log(`Certificate: ${CERT_PATH}`);
  });

  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.log(`Port ${PROXY_PORT} already in use — proxy may already be running.`);
      process.exit(0);
    }
    console.error('Proxy error:', err);
    process.exit(1);
  });
}

// Print cert path for Aster to add to trusted certs.
if (process.argv.includes('--cert-path')) {
  console.log(CERT_PATH);
  process.exit(0);
}

startProxy();
