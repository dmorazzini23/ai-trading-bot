import pytest
from server import WebhookPayload


def test_payload_valid():
    data = {'symbol': 'AAPL', 'action': 'buy'}
    payload = WebhookPayload.model_validate(data)
    assert payload.symbol == 'AAPL'
    assert payload.action == 'buy'


@pytest.mark.parametrize('data', [
    {},
    {'symbol': 'AAPL'},
    {'action': 'sell'},
])
def test_payload_invalid(data):
    with pytest.raises(Exception):
        WebhookPayload.model_validate(data)
