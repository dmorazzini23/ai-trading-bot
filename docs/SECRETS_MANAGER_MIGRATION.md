# Secrets Manager Migration (AWS Secrets Manager)

This moves runtime secrets out of local `.env` values and into AWS Secrets Manager.

## Prerequisites

- AWS CLI v2 installed (`aws --version`)
- IAM credentials configured on host (`aws configure` or instance role)
- Permissions:
  - `secretsmanager:GetSecretValue`
  - `secretsmanager:CreateSecret`
  - `secretsmanager:PutSecretValue`
  - `secretsmanager:DescribeSecret`

## What was added

- `scripts/runtime_env_sync.py`
  - used by `scripts/sync_env_runtime.sh`
  - supports `AI_TRADING_SECRETS_BACKEND=aws-secrets-manager`
  - pulls secret payload from AWS and overlays managed keys into `.env.runtime`
- `scripts/migrate_secrets_to_aws_sm.py`
  - uploads current local secret values from `.env` to AWS Secrets Manager
  - optional flags to strip local secrets and write backend config keys

## 1) Create or update AWS secret from your current `.env`

```bash
cd /home/aiuser/ai-trading-bot
source venv/bin/activate

python scripts/migrate_secrets_to_aws_sm.py \
  --secret-id ai-trading-bot/prod \
  --region us-west-2 \
  --merge-existing \
  --strip-local \
  --write-backend-config \
  --json | jq .
```

Notes:

- `--strip-local` blanks managed secrets in `.env` after successful upload.
- a backup `.env.bak.<timestamp>` is written before modifying `.env`.

## 2) Render `.env.runtime` from manager-backed secrets

```bash
./scripts/sync_env_runtime.sh
```

This now runs `scripts/runtime_env_sync.py` and writes `.env.runtime` with mode `0600`.

## 3) Enforce manager-only secrets (optional but recommended)

Set in `.env`:

```dotenv
AI_TRADING_REQUIRE_MANAGED_SECRETS=1
```

With this enabled, sync fails fast if a managed key is missing from the AWS secret payload.

## 4) Verify bot can read managed secrets

```bash
grep -n '^AI_TRADING_SECRETS_BACKEND=' .env .env.runtime
sudo systemctl restart ai-trading
curl -fsS http://127.0.0.1:8081/healthz | jq '{ok,status,reason,broker_status:.broker.status}'
```

## Secret payload format in AWS

Use a JSON object with env keys:

```json
{
  "ALPACA_API_KEY": "...",
  "ALPACA_SECRET_KEY": "...",
  "AI_TRADING_SLACK_WEBHOOK_URL": "...",
  "AI_TRADING_LINEAR_API_KEY": "...",
  "AI_TRADING_PROM_REMOTE_WRITE_PASSWORD": "..."
}
```

Dotenv-style payload text is also supported.
