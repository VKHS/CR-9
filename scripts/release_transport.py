"""Authenticated GitHub release downloads without changing artifact identities.

Resolve a GitHub browser-download URL through the release-assets API when a
credential is supplied. Credentials are never forwarded to mirrors or redirected
asset hosts. Archive SHA-256 and inventory verification remain with the caller.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
VERSION = "2022-11-28"
COMPONENT = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _origin(url: str) -> tuple[str, str | None, int | None]:
    p = urllib.parse.urlsplit(url)
    return p.scheme.lower(), p.hostname, p.port or {"https": 443, "http": 80}.get(p.scheme)


def _validate_url(url: str) -> urllib.parse.SplitResult:
    p = urllib.parse.urlsplit(url)
    if p.scheme not in {"http", "https"} or not p.hostname or p.username or p.password:
        raise ValueError("An HTTP(S) artifact URL without embedded credentials is required")
    # Access .port to reject malformed port numbers before sending any request.
    _ = p.port
    if any(ord(c) < 32 for c in url):
        raise ValueError("Control characters are not allowed in an artifact URL")
    return p


class CredentialSafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow ordinary redirects, removing credentials on any origin change."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newurl = urllib.parse.urljoin(req.full_url, newurl)
        before, after = _validate_url(req.full_url), _validate_url(newurl)
        if before.scheme == "https" and after.scheme != "https":
            raise urllib.error.URLError("Refusing an HTTPS artifact redirect to HTTP")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _origin(req.full_url) != _origin(newurl):
            for store in (redirected.headers, redirected.unredirected_hdrs):
                for key in list(store):
                    if key.lower() in {"authorization", "proxy-authorization", "cookie"}:
                        del store[key]
        return redirected


def _headers(url: str, token: str | None, *, binary: bool) -> dict[str, str]:
    _validate_url(url)
    result = {"User-Agent": "cross-morse-artifact-transport",
              "Accept": "application/octet-stream" if binary else "application/vnd.github+json"}
    if _origin(url) == ("https", "api.github.com", 443):
        result["X-GitHub-Api-Version"] = VERSION
        if token:
            result["Authorization"] = f"Bearer {token}"
    return result


def release_coordinates(url: str) -> tuple[str, str, str, str] | None:
    p = _validate_url(url)
    if _origin(url) != ("https", "github.com", 443):
        return None
    parts = p.path.strip("/").split("/")
    if len(parts) != 6 or parts[2:4] != ["releases", "download"]:
        return None
    owner, repo, tag, name = (urllib.parse.unquote(parts[i]) for i in (0, 1, 4, 5))
    if not COMPONENT.fullmatch(owner) or not COMPONENT.fullmatch(repo) or owner in {".", ".."} or repo in {".", ".."}:
        raise ValueError("Invalid GitHub repository in release URL")
    if not tag or not name or "/" in name or "\\" in name or any(ord(c) < 32 for c in tag + name):
        raise ValueError("Invalid GitHub tag or release asset name")
    return owner, repo, tag, name


def _json_get(url: str, token: str | None):
    opener = urllib.request.build_opener(CredentialSafeRedirect())
    request = urllib.request.Request(url, headers=_headers(url, token, binary=False))
    with opener.open(request, timeout=60) as response:
        payload = response.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("GitHub metadata response exceeds the allowed size")
    return json.loads(payload)


@functools.lru_cache(maxsize=16)
def _asset_ids(owner: str, repo: str, tag: str, token: str) -> dict[str, int]:
    """Cache metadata within one process; full artifact hashes are still verified."""
    root = f"{API}/repos/{owner}/{repo}"
    release = _json_get(f"{root}/releases/tags/{urllib.parse.quote(tag, safe='')}", token)
    release_id = release.get("id") if isinstance(release, dict) else None
    if type(release_id) is not int or release_id <= 0:
        raise ValueError("GitHub did not return a valid release identity")
    found: dict[str, int] = {}
    for page in range(1, 10001):
        rows = _json_get(f"{root}/releases/{release_id}/assets?per_page=100&page={page}", token)
        if not isinstance(rows, list):
            raise ValueError("GitHub did not return a release asset list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str) or type(row.get("id")) is not int or row["id"] <= 0:
                raise ValueError("GitHub returned a malformed release asset")
            if row["name"] in found:
                raise ValueError("Ambiguous duplicate release asset name")
            found[row["name"]] = row["id"]
        if len(rows) < 100:
            return found
    raise ValueError("GitHub release asset pagination did not terminate")


def download_url(url: str, token: str | None) -> str:
    coordinates = release_coordinates(url)
    if not token or coordinates is None:
        return url
    owner, repo, tag, name = coordinates
    asset_id = _asset_ids(owner, repo, tag, token).get(name)
    if asset_id is None:
        raise FileNotFoundError(f"Release {owner}/{repo}@{tag} does not contain asset {name}")
    # Build the API URL ourselves, rather than trusting an arbitrary URL in metadata.
    return f"{API}/repos/{owner}/{repo}/releases/assets/{asset_id}"


def download(url: str, target: Path, *, token: str | None = None) -> None:
    """Atomically fetch bytes; an interrupted transfer preserves an existing target."""
    target = Path(target).absolute()
    legacy_temporary = target.with_name(target.name + ".download")
    if any(p.is_symlink() for p in (target, legacy_temporary, *target.parents)):
        raise ValueError("Download paths must not traverse symlinks")
    resolved = download_url(url, token)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(resolved, headers=_headers(resolved, token, binary=True))
    opener = urllib.request.build_opener(CredentialSafeRedirect())
    temporary: Path | None = None
    try:
        with opener.open(request, timeout=600) as response:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if "/releases/assets/" in urllib.parse.urlsplit(resolved).path and content_type == "application/json":
                raise ValueError("GitHub returned asset metadata instead of archive bytes")
            length = response.headers.get("Content-Length")
            expected = int(length) if length is not None else None
            if expected is not None and expected < 0:
                raise ValueError("Invalid download content length")
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{target.name}.", suffix=".download",
                                             dir=target.parent, delete=False) as out:
                temporary = Path(out.name)
                received = 0
                while block := response.read(1 << 20):
                    out.write(block)
                    received += len(block)
                if expected is not None and received != expected:
                    raise OSError("Incomplete artifact transfer")
                out.flush()
                os.fsync(out.fileno())
        if target.is_symlink():
            raise ValueError("Download target changed into a symlink")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
