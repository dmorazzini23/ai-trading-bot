from tests import optdeps

def test_optdeps_smoke():
    assert isinstance(optdeps.OPTDEPS, dict)
