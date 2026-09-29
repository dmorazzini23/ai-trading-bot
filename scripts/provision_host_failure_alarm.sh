#!/usr/bin/env bash
# Run under an AWS administrator identity after reviewing the SNS recipient.
set -euo pipefail

recipient="${1:?usage: provision_host_failure_alarm.sh EMAIL_ADDRESS}"
region=us-east-2
account_id="$(aws sts get-caller-identity --query Account --output text)"
if [[ "$account_id" != 399705375437 ]]; then
  echo "Refusing to configure monitoring in unexpected AWS account" >&2
  exit 1
fi

topic_arn="$(aws sns create-topic --region "$region" --name ai-trading-host-failure --query TopicArn --output text)"
aws sns subscribe \
  --region "$region" \
  --topic-arn "$topic_arn" \
  --protocol email \
  --notification-endpoint "$recipient" \
  --output json

aws cloudwatch put-metric-alarm \
  --region "$region" \
  --alarm-name ai-trading-primary-host-failure \
  --alarm-description 'Paper host or canonical local health route stopped responding; a degraded trading gate still counts as host liveness.' \
  --namespace AITrading/Host \
  --metric-name PaperRuntimeResponding \
  --dimensions Name=Host,Value=ai-trading-primary \
  --statistic Minimum \
  --period 180 \
  --evaluation-periods 2 \
  --datapoints-to-alarm 2 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data breaching \
  --alarm-actions "$topic_arn" \
  --ok-actions "$topic_arn"

echo "Alarm configured. Confirm the SNS email subscription before testing delivery."
