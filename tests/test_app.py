from app import get_language_name


def test_get_language_name_iso2_valid():
    assert get_language_name(language_code="he") == "Hebrew"


def test_get_language_name_iso3_valid():
    assert get_language_name(language_code="arz") == "Egyptian Arabic"


def test_get_language_name_unknown():
    assert get_language_name(language_code="zz") == "Unknown"
