#!/bin/bash
# EC2 one-shot deploy script for live-idea-bench
# Usage: bash ec2_deploy.sh
# Run on a fresh Ubuntu 22.04 EC2 instance.

set -euo pipefail

REPO_URL="https://github.com/ulab-uiuc/live-idea-bench.git"
APP_DIR="$HOME/live-idea-bench"

echo "=== [1/5] Installing Docker ==="
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi

echo "=== [2/5] Installing Docker Compose plugin ==="
if ! sudo docker compose version &>/dev/null; then
  sudo apt-get install -y docker-compose-plugin
fi

echo "=== [3/5] Cloning / updating repo ==="
if [ -d "$APP_DIR" ]; then
  git -C "$APP_DIR" pull
else
  git clone "$REPO_URL" "$APP_DIR"
fi

echo "=== [4/5] Setting up .env ==="
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo ""
  echo "IMPORTANT: Edit $APP_DIR/.env and fill in your API keys, then re-run this script."
  echo "  nano $APP_DIR/.env"
  exit 1
fi

echo "=== [5/5] Building and starting services ==="
cd "$APP_DIR"
sudo docker compose up -d --build

PUBLIC_IP=$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || curl -sf https://ifconfig.me 2>/dev/null || echo "<your-ec2-ip>")

echo ""
echo "=== Done ==="
echo "Frontend: http://$PUBLIC_IP"
echo "Health:   http://$PUBLIC_IP/healthz"
echo ""
echo "Logs: sudo docker compose -f $APP_DIR/docker-compose.yml logs -f"
