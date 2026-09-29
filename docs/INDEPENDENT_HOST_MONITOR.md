# Independent paper-host failure alert

The packaged `ai-trading-host-heartbeat.timer` starts a separate oneshot every
two minutes. It publishes a standard CloudWatch metric in `us-east-2` only
after checking the local systemd service and canonical `/healthz` response.
HTTP 503 with a structured `degraded` response counts as **host liveness**, not
permission to trade. Failure to reach the local service publishes zero. Host,
network, credential or heartbeat-process failure stops the metric; the remote
alarm treats missing data as breaching. The alarm sends both failure and
recovery transitions to a confirmed SNS email subscription. It does not
replace broker, model, replay, cost, freshness or promotion gates.

## Administrator setup

Review the recurring CloudWatch metric/alarm and SNS usage for this account.
The host's IAM identity `ai-trading-secrets-bot` currently receives
`AccessDenied` for SNS topic listing and CloudWatch alarm description. An AWS
administrator should grant this narrowly scoped publisher permission to that
identity:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublishPaperHostLiveness",
    "Effect": "Allow",
    "Action": "cloudwatch:PutMetricData",
    "Resource": "*",
    "Condition": {"StringEquals": {"cloudwatch:namespace": "AITrading/Host"}}
  }]
}
```

After the exact-tip release has passed CI and the host exposure review, install
`packaging/systemd/ai-trading-host-heartbeat.{service,timer}` as root-owned
0644 files in `/etc/systemd/system`, run `sudo systemctl daemon-reload`, then
`sudo systemctl enable --now ai-trading-host-heartbeat.timer`. This separate
timer does not restart the trading service. Verify the service publishes value
1 from the same host. The known model and replay blockers should still appear
on `/healthz` as degraded readiness.

```bash
sudo install -o root -g root -m 0644 packaging/systemd/ai-trading-host-heartbeat.service /etc/systemd/system/ai-trading-host-heartbeat.service
sudo install -o root -g root -m 0644 packaging/systemd/ai-trading-host-heartbeat.timer /etc/systemd/system/ai-trading-host-heartbeat.timer
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-host-heartbeat.timer
systemctl status ai-trading-host-heartbeat.timer --no-pager
```

Under an administrator identity in account `399705375437`, run the reviewed
`bash scripts/provision_host_failure_alarm.sh dmorazzini23@gmail.com` after
the first metric is visible. The script creates one SNS topic, one email
subscription and one CloudWatch alarm for the exact metric/dimension emitted
by the service. The owner must click the `Confirm subscription` link in the
AWS email; `PendingConfirmation` is not delivery evidence. Do not forward the
confirmation link to anyone. Confirm the alarm becomes `OK`.

For the actual notification drill, record the alarm state and subscription
ARN, use CloudWatch `set-alarm-state` to enter `ALARM` with the reason
`Authorized host-failure notification drill`, verify email delivery and the
owner's acknowledgement, then observe normal metric evaluation return the
alarm to `OK`. A temporary alarm state is test evidence only; it does not
prove a real host-outage response until a separate controlled host-failure
drill is completed. The recipient must not treat a test alert as an actual
trading incident. Record all timestamps, metric values, alarm history and
confirmation/acknowledgement in the handoff.

Until IAM, subscription confirmation, host installation, alarm history and
delivered/acknowledged test email are verified, independent alerting remains
**unproven**. Existing same-host health checks cannot detect total host loss.
