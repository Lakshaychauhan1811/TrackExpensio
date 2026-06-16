"""Quick integration test for India bank sync + core dashboard APIs."""
import asyncio
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8080"


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


async def link_test_session(session_id: str) -> None:
    from session_manager import SessionManager
    sm = SessionManager()
    await sm.link_session_to_google(session_id, "test_google_user_india_sync")


def main():
    print("=== Health ===")
    code, data = req("GET", "/api/health")
    assert code == 200 and data.get("status") == "healthy", data
    print("OK")

    print("=== AA Status ===")
    code, data = req("GET", "/api/aa/status")
    assert code == 200 and data.get("providers"), data
    print("OK", data.get("providers"))

    print("=== Create session ===")
    code, data = req("POST", "/api/session/create")
    assert code == 200, data
    session_id = data["session_id"]
    print("session", session_id[:8] + "...")

    print("=== Guest SMS (expect 401) ===")
    code, _ = req("POST", "/api/sms/parse", {"session_id": session_id, "sms_text": "Rs. 450 spent on Amazon", "save": False})
    assert code == 401
    print("OK (auth required)")

    print("=== Link session for tests ===")
    asyncio.run(link_test_session(session_id))
    auth = {"session_id": session_id}

    print("=== SMS parse preview ===")
    code, data = req("POST", "/api/sms/parse", {**auth, "sms_text": "Rs. 450 spent on Amazon", "save": False})
    assert code == 200 and data.get("parsed"), data
    print("OK", data["parsed"][0]["merchant"], data["parsed"][0]["amount"])

    print("=== SMS save ===")
    code, data = req("POST", "/api/sms/parse", {**auth, "sms_text": "INR 599 debited from HDFC Bank", "save": True})
    assert code == 200 and data.get("imported", 0) >= 1, data
    print("OK imported", data.get("imported"))

    print("=== AA consent ===")
    code, data = req("POST", "/api/aa/consent", {**auth, "provider": "setu", "bank_name": "HDFC Bank"})
    assert code == 200 and data.get("consent_id"), data
    consent_id = data["consent_id"]
    connection_id = data["connection_id"]
    print("OK consent", consent_id)

    print("=== AA approve ===")
    code, data = req("POST", "/api/aa/approve", {**auth, "consent_id": consent_id, "approved": True})
    assert code == 200 and data.get("approved"), data
    print("OK")

    print("=== AA sync ===")
    code, data = req("POST", "/api/aa/sync", {**auth, "connection_id": connection_id})
    assert code == 200 and data.get("imported", 0) >= 1, data
    print("OK imported", data.get("imported"))

    print("=== AA connections ===")
    code, data = req("GET", f"/api/aa/connections?session_id={session_id}")
    assert code == 200 and data.get("count", 0) >= 1, data
    print("OK count", data["count"])

    print("=== Market quote ===")
    code, data = req("GET", f"/api/market/quote?symbol=AAPL&session_id={session_id}")
    assert code == 200 and data.get("data", {}).get("price"), data
    print("OK price", data["data"]["price"])

    print("=== HTML has india_bank_sync.js ===")
    with urllib.request.urlopen(f"{BASE}/", timeout=10) as resp:
        html = resp.read().decode()
    assert "india_bank_sync.js" in html
    assert 'data-quick-action="expense"' in html
    assert "aaConsentModal" in html
    print("OK")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
