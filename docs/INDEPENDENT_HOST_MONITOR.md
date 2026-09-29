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
The host IAM identity `ai-trading-secrets-bot` has the following narrowly
scoped publisher permission. A manual publish succeeded on September 29.
The host still lacks SNS topic listing and CloudWatch alarm description, so
its IAM identity cannot independently verify the owner's report that the
Ohio topic `ai-trading-host-failure` and email subscription are confirmed:

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

After recurring value 1 appears for custom metric `AITrading/Host`,
`PaperRuntimeResponding`, `Host=ai-trading-primary`, an AWS administrator in
account `399705375437` should create the alarm against the **existing**
confirmed SNS topic. Do not run the full provisioning script for this setup:
it also calls `sns subscribe`, which is unnecessary for the confirmed email.
The CloudWatch alarm settings are: Minimum, 180-second period, 2 of 2
datapoints, less than 1, missing data breaching, and both alarm and OK actions
to `arn:aws:sns:us-east-2:399705375437:ai-trading-host-failure`. Confirm the
alarm becomes `OK` after recurring metrics arrive.

For the actual notification drill, record the alarm state and subscription
ARN, use CloudWatch `set-alarm-state` to enter `ALARM` with the reason
`Authorized host-failure notification drill`, verify email delivery and the
owner's acknowledgement, then observe normal metric evaluation return the
alarm to `OK`. A temporary alarm state is test evidence only; it does not
prove a real host-outage response until a separate controlled host-failure
drill is completed. The recipient must not treat a test alert as an actual
trading incident. Record all timestamps, metric values, alarm history and
confirmation/acknowledgement in the handoff.

Until host installation, recurring metric, alarm history and a delivered and
acknowledged test email are verified, independent alerting remains **unproven**.
Existing same-host health checks cannot detect total host loss.
