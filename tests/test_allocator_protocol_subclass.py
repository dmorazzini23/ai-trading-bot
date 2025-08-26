from ai_trading.core.protocols import AllocatorProtocol


class DummyAllocator(AllocatorProtocol):
    def allocate(self, signals, runtime):
        return {}


def test_subclass_isinstance():
    assert isinstance(DummyAllocator(), AllocatorProtocol)
