@echo off
REM Generates a self-signed TLS cert/key for local HTTPS dev (certs\dev_cert.pem,
REM certs\dev_key.pem) if they don't already exist. Not for production --
REM browsers will warn on the self-signed cert; that's expected for local dev.
setlocal
cd /d "%~dp0.."

if exist certs\dev_cert.pem if exist certs\dev_key.pem (
    echo == certs\dev_cert.pem and dev_key.pem already exist, skipping ==
    exit /b 0
)

where openssl >nul 2>nul
if errorlevel 1 (
    echo openssl not found on PATH -- install it ^(ships with Git for Windows' Git Bash^)
    echo or run scripts\gen_dev_cert.sh from Git Bash / WSL instead.
    exit /b 1
)

if not exist certs mkdir certs

echo == generating self-signed dev TLS cert ^(certs\dev_cert.pem, certs\dev_key.pem^) ==
openssl req -x509 -newkey rsa:2048 -keyout certs\dev_key.pem -out certs\dev_cert.pem ^
    -days 365 -nodes -subj "/CN=localhost" ^
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
echo Done. This is a self-signed cert for LOCAL DEV ONLY -- browsers/curl will warn/reject it as untrusted, which is expected.
