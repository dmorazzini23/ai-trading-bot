        # Drop bars from the current (incomplete) minute and filter invalid volume rows
                provider_tokens: set[str] = set()
                if provider_hint:
                    provider_tokens.add(provider_hint)
                    if "_" in provider_hint:
                        provider_tokens.add(provider_hint.split("_", 1)[0])
                allow_zero_volume_providers = {
                provider_is_alpaca = any(
                    token.startswith("alpaca") for token in provider_tokens
                )
                provider_allows_zero = any(
                    token in allow_zero_volume_providers for token in provider_tokens
                )
                asset_class_attr = str(
                    raw_attrs.get("asset_class")
                    or raw_attrs.get("assetType")
                    or ""
                ).strip().lower()
                asset_is_equity = asset_class_attr == "equity"
                if not asset_is_equity:
                    try:
                        asset_is_equity = asset_class_for(symbol).lower() == "equity"
                    except COMMON_EXC:
                        asset_is_equity = False
                allow_non_negative = (
                    provider_is_alpaca
                    or provider_allows_zero
                    or asset_is_equity
                )
                if allow_non_negative:
