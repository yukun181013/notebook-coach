import pytest

from notebook_coach.sanitize import redact_text, summarize_text


def test_redacts_secret_without_returning_original_value():
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    cleaned, labels = redact_text(f"OPENAI_API_KEY={secret}")
    assert cleaned == "OPENAI_API_KEY=[REDACTED]"
    assert secret not in repr((cleaned, labels))
    assert labels == ["openai_api_key"]


def test_summarize_text_keeps_length_hash_and_marker():
    result = summarize_text("x" * 5000, max_chars=100)
    assert result["truncated"] is True
    assert result["original_chars"] == 5000
    assert len(result["text"]) <= 100
    assert len(result["sha256"]) == 64


def test_redacts_github_token_without_returning_original_value():
    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    cleaned, labels = redact_text(f"value={secret}")

    assert cleaned == "value=[REDACTED]"
    assert secret not in repr((cleaned, labels))
    assert labels == ["github_token"]


def test_redacts_entire_pem_private_key_without_returning_original_value():
    secret = """-----BEGIN PRIVATE KEY-----
not-a-real-private-key-payload
-----END PRIVATE KEY-----"""
    cleaned, labels = redact_text(f"before\n{secret}\nafter")

    assert cleaned == "before\n[REDACTED]\nafter"
    assert secret not in repr((cleaned, labels))
    assert labels == ["pem_private_key"]


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("service_api_key", "abcdefghijklmnopqrstuvwxyz123456"),
        ("access_token", "token-value-abcdefghijklmnopqrstuvwxyz"),
        ("database_password", "correct-horse-battery-staple"),
        ("client_secret", "secret-value-abcdefghijklmnopqrstuvwxyz"),
    ],
)
def test_redacts_sensitive_named_assignments(name, secret):
    cleaned, labels = redact_text(f'{name} = "{secret}"')

    assert name in cleaned
    assert "[REDACTED]" in cleaned
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_entire_unquoted_sensitive_value_with_punctuation():
    secret = "correct#horse;battery-staple"

    cleaned, labels = redact_text(f"password={secret}")

    assert cleaned == "password=[REDACTED]"
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_does_not_redact_ordinary_short_numeric_token_metadata():
    text = "token_count = 32"

    assert redact_text(text) == (text, [])


def test_summarize_text_redacts_before_truncating_and_returns_only_safe_fields():
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    text = f"OPENAI_API_KEY={secret}\n" + ("x" * 200)

    result = summarize_text(text, max_chars=80)

    assert set(result) == {"text", "original_chars", "sha256", "truncated"}
    assert result["text"].startswith("OPENAI_API_KEY=[REDACTED]")
    assert len(result["text"]) <= 80
    assert result["original_chars"] == len(text)
    assert result["truncated"] is True
    assert secret not in repr(result)


def test_summarize_text_allows_zero_character_limit():
    result = summarize_text("visible", max_chars=0)

    assert result["text"] == ""
    assert result["truncated"] is True


def test_summarize_text_rejects_negative_character_limit():
    with pytest.raises(ValueError, match="max_chars"):
        summarize_text("visible", max_chars=-1)
