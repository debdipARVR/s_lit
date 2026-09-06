"""Security, Authentication, and BYOK Encryption Package."""

from .encryption import (
    EncryptionError,
    decrypt_api_key,
    encrypt_api_key,
    generate_fernet_key,
    encrypt_at_rest,
    decrypt_at_rest,
    hash_blind_index,
    mask_api_key,
    resolve_api_credentials,
    resolve_openrouter_credentials,
)
from .sanitizer import (
    sanitize_html,
    sanitize_input_text,
    validate_api_key_format,
    MAX_INPUT_CHARS,
    MAX_WORD_COUNT,
)
from .session_manager import (
    SessionManager,
    get_session_manager,
)
from .ssh_verifier import (
    generate_ssh_verifiable_session,
    verify_ssh_session,
    get_or_create_ssh_authority,
)
from .auth import (
    get_current_user,
    is_authenticated,
    authenticate_user,
    logout_user,
    render_login_gate,
)

__all__ = [
    "EncryptionError",
    "generate_fernet_key",
    "encrypt_api_key",
    "decrypt_api_key",
    "encrypt_at_rest",
    "decrypt_at_rest",
    "hash_blind_index",
    "mask_api_key",
    "resolve_api_credentials",
    "resolve_openrouter_credentials",
    "sanitize_html",
    "sanitize_input_text",
    "validate_api_key_format",
    "MAX_INPUT_CHARS",
    "MAX_WORD_COUNT",
    "SessionManager",
    "get_session_manager",
    "generate_ssh_verifiable_session",
    "verify_ssh_session",
    "get_or_create_ssh_authority",
    "get_current_user",
    "is_authenticated",
    "authenticate_user",
    "logout_user",
    "render_login_gate",
]
