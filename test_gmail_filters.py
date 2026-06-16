"""Unit tests for Gmail/transaction promo filtering."""

from services.transaction_filters import (
    extract_debit_amount,
    extract_merchant_amount,
    is_promotional,
    should_keep_gmail_expense,
    validate_email_for_import,
)


def test_blocks_flipkart_voucher_promo():
    subject = "₹0 joining fee + ₹1,000 Welcome Voucher 🤑"
    body = "Get your Flipkart credit card welcome voucher today!"
    ok, reason = validate_email_for_import("offers@flipkart.com", subject, body)
    assert not ok
    assert reason == "promotional"


def test_blocks_cashback_offer():
    subject = "Get Up to ₹300 Cashback on Google Play 🥳"
    body = "Limited time offer from Paytm"
    ok, reason = validate_email_for_import("noreply@paytm.com", subject, body)
    assert not ok
    assert reason == "promotional"


def test_blocks_unlock_emi_ad():
    subject = "Unlock ₹1 Lakh in minutes with Flipkart EMI!"
    body = "Apply now for instant credit"
    ok, reason = validate_email_for_import("offers@flipkart.com", subject, body)
    assert not ok
    assert reason == "promotional"


def test_blocks_paytm_travel_marketing():
    subject = "Paytm Travel Flight booking"
    body = "Book your next flight with Paytm Travel. Fares from Rs. 3000."
    ok, reason = validate_email_for_import("noreply@paytm.com", subject, body)
    assert not ok
    assert reason in ("not_confirmed_payment", "no_amount", "no_payment_confirmation")


def test_allows_delhi_metro_order():
    subject = "Delhi Metro ticket receipt"
    body = "Order ID 27080216518. Amount paid Rs. 100 for your journey."
    ok, reason = validate_email_for_import("noreply@delhimetrorail.com", subject, body)
    assert ok, reason
    assert extract_merchant_amount(f"{subject}\n{body}") == 100.0


def test_allows_bank_debit_alert():
    subject = "Transaction Alert"
    body = (
        "Rs. 450 debited from A/C XX1234 on 06-Jun-26. "
        "Available balance Rs. 12,000. Transaction ID: TXN123456789"
    )
    ok, reason = validate_email_for_import("alerts@hdfcbank.net", subject, body)
    assert ok, reason
    assert extract_debit_amount(body) == 450.0


def test_blocks_credit_only():
    subject = "Salary credited"
    body = "Rs. 50000 credited to your account. Available balance Rs. 75000."
    ok, reason = validate_email_for_import("alerts@hdfcbank.net", subject, body)
    assert not ok
    assert reason == "credit_not_expense"


def test_purge_paytm_travel_import():
    assert not should_keep_gmail_expense(
        "Paytm Travel Flight booking",
        "Paytm Travel Flight booking",
        "Paytm Travel",
    )


def test_keep_metro_with_amount_paid():
    assert should_keep_gmail_expense(
        "Delhi Metro ticket receipt",
        "Order ID 27080216518. Amount paid Rs. 100",
        "Delhi Metro",
    )


def test_is_promotional_keyword_detection():
    assert is_promotional("Special offer", "Buy now")
    assert not is_promotional("Payment successful", "Order ID 123 paid Rs 500")


if __name__ == "__main__":
    test_blocks_flipkart_voucher_promo()
    test_blocks_cashback_offer()
    test_blocks_unlock_emi_ad()
    test_blocks_paytm_travel_marketing()
    test_allows_delhi_metro_order()
    test_allows_bank_debit_alert()
    test_blocks_credit_only()
    test_purge_paytm_travel_import()
    test_keep_metro_with_amount_paid()
    test_is_promotional_keyword_detection()
    print("All gmail filter tests passed.")
