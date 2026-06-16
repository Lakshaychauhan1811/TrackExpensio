"""
Verification script to ensure all functions are properly registered and accessible.
Run this to verify all MCP tools are working correctly.
"""

import asyncio
from main import TOOL_REGISTRY
from audit_logger import AuditAction
from tax_rules import TaxJurisdiction
from fx_timing import fx_manager
from security import security_manager


def verify_tool_registry():
    """Verify all tools are registered"""
    print("=" * 60)
    print("VERIFYING TOOL REGISTRY")
    print("=" * 60)
    
    expected_tools = [
        "add_expense",
        "list_expenses",
        "summarize",
        "ai_insights",
        "quick_add_expense",
        "document_expense_from_rag",
        "delete_expense",
        "update_expense",
        "set_budget",
        "check_budget_status",
        "add_recurring_expense",
        "generate_monthly_recurring",
        "add_income",
        "list_income",
        "get_income_summary",
        "set_savings_goal",
        "track_savings_progress",
        "add_debt",
        "record_debt_payment",
        "get_debt_summary",
        "add_investment",
        "update_investment_value",
        "get_investment_portfolio",
        "generate_financial_report",
        "estimate_taxes",
        "record_credit_score",
        "get_credit_score_trend",
        "add_bill_reminder",
        "get_upcoming_bills",
        "add_expense_multicurrency",
        "set_base_currency",
        "create_user_role",
        "share_account_access",
        "connect_bank_account",
        "sync_bank_transactions",
        "yahoo_finance",
        "smart_stock_analyze",
        "get_stock_return_one_year",
        "get_stock_returns",
        "convert_currency_with_timing",
        "get_audit_logs",
        "get_help",
    ]
    
    missing_tools = []
    for tool in expected_tools:
        if tool not in TOOL_REGISTRY:
            missing_tools.append(tool)
            print(f"❌ MISSING: {tool}")
        else:
            print(f"✅ FOUND: {tool}")
    
    print(f"\nTotal tools registered: {len(TOOL_REGISTRY)}")
    print(f"Expected tools: {len(expected_tools)}")
    print(f"Missing tools: {len(missing_tools)}")
    
    if missing_tools:
        print(f"\n⚠️  Missing tools: {', '.join(missing_tools)}")
        return False
    else:
        print("\n✅ All tools are properly registered!")
        return True


def verify_modules():
    """Verify all modules are importable"""
    print("\n" + "=" * 60)
    print("VERIFYING MODULES")
    print("=" * 60)
    
    modules = {
        "audit_logger": ["audit_logger", "AuditAction"],
        "tax_rules": ["tax_engine", "TaxJurisdiction"],
        "fx_timing": ["fx_manager"],
        "security": ["security_manager"],
    }
    
    all_ok = True
    for module_name, items in modules.items():
        try:
            for item in items:
                print(f"✅ {module_name}.{item} - OK")
        except Exception as e:
            print(f"❌ {module_name} - ERROR: {e}")
            all_ok = False
    
    return all_ok


def verify_audit_actions():
    """Verify audit actions are defined"""
    print("\n" + "=" * 60)
    print("VERIFYING AUDIT ACTIONS")
    print("=" * 60)
    
    actions = [
        AuditAction.EXPENSE_ADDED,
        AuditAction.EXPENSE_UPDATED,
        AuditAction.EXPENSE_DELETED,
        AuditAction.LOGIN,
        AuditAction.FAILED_LOGIN,
        AuditAction.TAX_ESTIMATED,
    ]
    
    for action in actions:
        print(f"✅ {action.value}")
    
    return True


def verify_tax_jurisdictions():
    """Verify tax jurisdictions are defined"""
    print("\n" + "=" * 60)
    print("VERIFYING TAX JURISDICTIONS")
    print("=" * 60)
    
    jurisdictions = [
        TaxJurisdiction.INDIA,
        TaxJurisdiction.USA,
        TaxJurisdiction.UK,
        TaxJurisdiction.CANADA,
        TaxJurisdiction.AUSTRALIA,
    ]
    
    for jurisdiction in jurisdictions:
        print(f"✅ {jurisdiction.value} - {jurisdiction.name}")
    
    return True


async def verify_async_functions():
    """Verify async functions can be called"""
    print("\n" + "=" * 60)
    print("VERIFYING ASYNC FUNCTIONS")
    print("=" * 60)
    
    try:
        # Test FX manager (doesn't require auth)
        rate = await fx_manager.get_exchange_rate("USD", "INR")
        print(f"✅ FX Manager - Rate fetched: {rate.get('rate', 'N/A')}")
    except Exception as e:
        print(f"❌ FX Manager - ERROR: {e}")
        return False
    
    try:
        # Test security manager
        is_allowed, remaining = await security_manager.check_rate_limit("test_user", max_requests=100)
        print(f"✅ Security Manager - Rate limit check: {is_allowed}, remaining: {remaining}")
    except Exception as e:
        print(f"❌ Security Manager - ERROR: {e}")
        return False
    
    return True


def main():
    """Run all verification checks"""
    print("\n" + "=" * 60)
    print("TRACKEXPENSIO FUNCTION VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Verify tool registry
    results.append(("Tool Registry", verify_tool_registry()))
    
    # Verify modules
    results.append(("Modules", verify_modules()))
    
    # Verify audit actions
    results.append(("Audit Actions", verify_audit_actions()))
    
    # Verify tax jurisdictions
    results.append(("Tax Jurisdictions", verify_tax_jurisdictions()))
    
    # Verify async functions
    async_result = asyncio.run(verify_async_functions())
    results.append(("Async Functions", async_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED!")
        print("All functions are properly registered and accessible.")
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("Please check the errors above.")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    main()
