"""Reversible email tokens instead of redaction, and "me" for the signed-in user.

Where a VIRA or database result has an email address, the model sees a token such as
<email:3f9a1c2b4d5e> instead of "<redacted-email>" (recruiter_cli._mask_pii with a source).  Just
before a call is sent, recruiter_cli._call swaps each token back to its address (the decrypt
step), so the agent can act on an address it never saw.  The address never reaches the model,
the graph state, the checkpointer, a trace or the audit log (which stays redacted): only this
process holds it, encrypted with AES-GCM under JENI_PII_KEY (a random key for each process when
that is unset, so tokens die with it).

Who may send what (ToolCallGuard asks allowed()):
  * a colleague's token (from the user directory: search_users, find_user, validate_email) and
    "me" (the signed-in user's address, JENI_USER_EMAIL, set by the host) may go to a share
    recipient or a new owner;
  * any other token, a candidate's included, may go nowhere: those fields still need the
    user's own words.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import threading
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

EMAIL = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+")
TOKEN = re.compile(r"<email:[0-9a-f]{12}>")
ME = "me"
COLLEAGUE, SELF, RECORD = "colleague", "self", "record"
# Fields where a colleague's token or "me" may stand in for an address the user typed.
RECIPIENT_FIELDS = frozenset({"emails", "new_owner_user_email", "email"})


class UnknownToken(LookupError):
    """A token this process didn't issue (another process's, or made up)."""


def _master_key() -> bytes:
    raw = os.environ.get("JENI_PII_KEY", "").strip()
    if not raw:
        return os.urandom(32)
    key = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    if len(key) != 32:
        raise ValueError("JENI_PII_KEY must be 32 random bytes, base64-encoded")
    return key


class Vault:
    """Tokens to addresses, each address encrypted; nothing is ever written out."""

    def __init__(self, key: bytes | None = None, user_email: str | None = None):
        # Both are read on first use, so a .env loaded after this module is imported counts.
        self._master, self._user_email = key, user_email
        self._token_key: bytes | None = None
        self._aead: AESGCM | None = None
        self._entries: dict[str, tuple[bytes, bytes, set[str]]] = {}
        self._lock = threading.Lock()

    @property
    def user_email(self) -> str | None:
        """The signed-in user's address, for "me": set by the host, or JENI_USER_EMAIL."""
        return self._user_email or os.environ.get("JENI_USER_EMAIL", "").strip() or None

    @user_email.setter
    def user_email(self, value: str | None) -> None:
        self._user_email = value

    def _keys(self) -> tuple[bytes, AESGCM]:
        if self._token_key is None:
            master = self._master or _master_key()
            self._token_key = hmac.new(master, b"jeni-pii/token", hashlib.sha256).digest()
            self._aead = AESGCM(hmac.new(master, b"jeni-pii/aead", hashlib.sha256).digest())
        return self._token_key, self._aead

    def token(self, address: str, source: str) -> str:
        """The token for an address (the same one each time), noting where it was seen."""
        address = address.strip()
        token_key, aead = self._keys()
        digest = hmac.new(token_key, address.lower().encode(), hashlib.sha256).hexdigest()
        token = f"<email:{digest[:12]}>"
        with self._lock:
            if token in self._entries:
                self._entries[token][2].add(source)
            else:
                nonce = os.urandom(12)
                sealed = aead.encrypt(nonce, address.encode(), token.encode())
                self._entries[token] = (nonce, sealed, {source})
        return token

    def reveal(self, token: str) -> str:
        with self._lock:
            entry = self._entries.get(token)
        if entry is None:
            raise UnknownToken(token)
        nonce, sealed, _ = entry
        return self._keys()[1].decrypt(nonce, sealed, token.encode()).decode()

    def sources(self, token: str) -> set[str]:
        with self._lock:
            entry = self._entries.get(token.strip())
            return set(entry[2]) if entry else set()

    def tokenize(self, text: str, source: str) -> str:
        return EMAIL.sub(lambda m: self.token(m.group(0), source), text)

    def me(self) -> str | None:
        """A token for the signed-in user's own address, or None when the host set none."""
        return self.token(self.user_email, SELF) if self.user_email else None

    def resolve(self, value: Any) -> Any:
        """A value as it should be sent: tokens swapped back to addresses, everywhere inside."""
        if isinstance(value, str):
            return TOKEN.sub(lambda m: self.reveal(m.group(0)), value)
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        return value

    def allowed(self, field: str, value: Any) -> bool:
        """May `value` stand in for an address the user typed, in this field?"""
        if field not in RECIPIENT_FIELDS or not isinstance(value, str):
            return False
        value = value.strip()
        if value.lower() == ME:
            return self.user_email is not None
        return bool(TOKEN.fullmatch(value)) and bool(self.sources(value) & {COLLEAGUE, SELF})

    def display(self, value: Any) -> Any:
        """For an approval card: "you" for "me", a hint such as p•••r@example.com for a token."""
        if not isinstance(value, str):
            return value
        if value.strip().lower() == ME:
            return "you"
        if not TOKEN.fullmatch(value.strip()):
            return value
        try:
            local, _, domain = self.reveal(value.strip()).partition("@")
        except UnknownToken:
            return "an unknown address"
        return f"{local[:1]}•••{local[-1:] if len(local) > 1 else ''}@{domain}"


VAULT = Vault()


def resolve_me(field: str, value: Any) -> Any:
    """"me" in a recipient field as the signed-in user's token; raises LookupError without one."""
    if field not in RECIPIENT_FIELDS:
        return value
    values = value if isinstance(value, list) else [value]
    out = []
    for v in values:
        if isinstance(v, str) and v.strip().lower() == ME:
            token = VAULT.me()
            if token is None:
                raise LookupError("the signed-in user's email isn't known")
            v = token
        out.append(v)
    return out if isinstance(value, list) else out[0]
