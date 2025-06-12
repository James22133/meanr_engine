def test_load_config_returns_dict():
    from config import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "ETF_TICKERS" in cfg
