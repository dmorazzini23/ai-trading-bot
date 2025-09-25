# AI-AGENT-REF: shared constants for tests
from binascii import unhexlify


LEGACY_ENV_PREFIXES = (unhexlify("415043415f").decode("ascii"),)
LEGACY_ENV_WHITELIST: set[str] = set()
