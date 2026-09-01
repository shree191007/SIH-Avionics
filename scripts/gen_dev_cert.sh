#!/bin/bash
# Generates a self-signed TLS cert/key for local HTTPS dev (certs/dev_cert.pem,
# certs/dev_key.pem) if they don't already exist. Not for production -- browsers
# will warn on the self-signed cert; that's expected for local dev.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="$ROOT/certs"
mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/dev_cert.pem" ] && [ -f "$CERT_DIR/dev_key.pem" ]; then
    echo "== certs/dev_cert.pem and dev_key.pem already exist, skipping =="
    exit 0
fi

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl not found on PATH -- install it (macOS: comes with Xcode CLT / Homebrew) to generate a dev TLS cert."
    exit 1
fi

echo "== generating self-signed dev TLS cert (certs/dev_cert.pem, certs/dev_key.pem) =="
openssl req -x509 -newkey rsa:2048 -keyout "$CERT_DIR/dev_key.pem" -out "$CERT_DIR/dev_cert.pem" \
    -days 365 -nodes -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
echo "Done. This is a self-signed cert for LOCAL DEV ONLY -- browsers/curl will warn/reject it as untrusted, which is expected."
