"""
Multi-Jurisdiction Tax Rules Engine

Supports tax calculations for multiple countries/jurisdictions with
different tax rules, brackets, deductions, and exemptions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class TaxJurisdiction(str, Enum):
    """Supported tax jurisdictions"""
    INDIA = "IN"
    USA = "US"
    UK = "GB"
    CANADA = "CA"
    AUSTRALIA = "AU"
    GERMANY = "DE"
    FRANCE = "FR"
    SINGAPORE = "SG"
    UAE = "AE"  # No income tax
    SWITZERLAND = "CH"


class TaxRuleEngine:
    """Multi-jurisdiction tax calculation engine"""
    
    # Tax brackets for different jurisdictions (2024-2025 rates)
    TAX_BRACKETS = {
        TaxJurisdiction.INDIA: [
            {"min": 0, "max": 300000, "rate": 0.0},
            {"min": 300000, "max": 700000, "rate": 0.05},
            {"min": 700000, "max": 1000000, "rate": 0.10},
            {"min": 1000000, "max": 1200000, "rate": 0.15},
            {"min": 1200000, "max": 1500000, "rate": 0.20},
            {"min": 1500000, "max": float("inf"), "rate": 0.30},
        ],
        TaxJurisdiction.USA: [
            {"min": 0, "max": 11000, "rate": 0.10},
            {"min": 11000, "max": 44725, "rate": 0.12},
            {"min": 44725, "max": 95350, "rate": 0.22},
            {"min": 95350, "max": 201050, "rate": 0.24},
            {"min": 201050, "max": 578125, "rate": 0.32},
            {"min": 578125, "max": 693750, "rate": 0.35},
            {"min": 693750, "max": float("inf"), "rate": 0.37},
        ],
        TaxJurisdiction.UK: [
            {"min": 0, "max": 12570, "rate": 0.0},  # Personal allowance
            {"min": 12570, "max": 50270, "rate": 0.20},
            {"min": 50270, "max": 125140, "rate": 0.40},
            {"min": 125140, "max": float("inf"), "rate": 0.45},
        ],
        TaxJurisdiction.CANADA: [
            {"min": 0, "max": 55867, "rate": 0.15},
            {"min": 55867, "max": 111733, "rate": 0.205},
            {"min": 111733, "max": 173205, "rate": 0.26},
            {"min": 173205, "max": 246752, "rate": 0.29},
            {"min": 246752, "max": float("inf"), "rate": 0.33},
        ],
        TaxJurisdiction.AUSTRALIA: [
            {"min": 0, "max": 18200, "rate": 0.0},
            {"min": 18200, "max": 45000, "rate": 0.19},
            {"min": 45000, "max": 120000, "rate": 0.325},
            {"min": 120000, "max": 180000, "rate": 0.37},
            {"min": 180000, "max": float("inf"), "rate": 0.45},
        ],
    }
    
    # Standard deductions/exemptions (annual)
    STANDARD_DEDUCTIONS = {
        TaxJurisdiction.INDIA: {
            "standard_deduction": 50000,  # Section 16
            "section_80c": 150000,  # Investments, insurance, etc.
            "section_80d": 25000,  # Health insurance
            "section_24b": 200000,  # Home loan interest
            "section_80g": 0,  # Donations
        },
        TaxJurisdiction.USA: {
            "standard_deduction": 14600,  # Single filer 2024
            "itemized_deductions": 0,
        },
        TaxJurisdiction.UK: {
            "personal_allowance": 12570,
            "pension_contributions": 0,
        },
    }
    
    # Tax-exempt categories
    TAX_EXEMPT_CATEGORIES = {
        TaxJurisdiction.INDIA: [
            "Health Insurance Premium",
            "Life Insurance Premium",
            "ELSS Mutual Funds",
            "PPF Contribution",
            "EPF Contribution",
            "NPS Contribution",
            "Home Loan Principal",
            "Tuition Fees",
            "Medical Expenses",
        ],
        TaxJurisdiction.USA: [
            "401k Contribution",
            "IRA Contribution",
            "HSA Contribution",
            "Health Insurance Premium",
            "Mortgage Interest",
            "Charitable Donations",
        ],
    }
    
    def __init__(self, jurisdiction: TaxJurisdiction = TaxJurisdiction.INDIA):
        self.jurisdiction = jurisdiction
    
    def calculate_income_tax(
        self,
        gross_income: float,
        deductions: Optional[Dict[str, float]] = None,
        exemptions: Optional[Dict[str, float]] = None,
        filing_status: str = "single",
    ) -> Dict[str, Any]:
        """
        Calculate income tax based on jurisdiction rules
        
        Args:
            gross_income: Total annual income
            deductions: Dictionary of deduction types and amounts
            exemptions: Dictionary of exemption types and amounts
            filing_status: Filing status (single, married, etc.)
        """
        if self.jurisdiction == TaxJurisdiction.UAE:
            return {
                "gross_income": gross_income,
                "taxable_income": 0,
                "total_tax": 0,
                "effective_rate": 0.0,
                "marginal_rate": 0.0,
                "jurisdiction": self.jurisdiction.value,
            }
        
        # Apply standard deductions
        standard_deductions = self.STANDARD_DEDUCTIONS.get(self.jurisdiction, {})
        total_deductions = sum(standard_deductions.values())
        
        # Add custom deductions
        if deductions:
            total_deductions += sum(deductions.values())
        
        # Apply exemptions
        total_exemptions = 0
        if exemptions:
            total_exemptions = sum(exemptions.values())
        
        # Calculate taxable income
        taxable_income = max(0, gross_income - total_deductions - total_exemptions)
        
        # Get tax brackets
        brackets = self.TAX_BRACKETS.get(self.jurisdiction, [])
        if not brackets:
            return {
                "gross_income": gross_income,
                "taxable_income": taxable_income,
                "total_tax": 0,
                "effective_rate": 0.0,
                "marginal_rate": 0.0,
                "jurisdiction": self.jurisdiction.value,
                "error": "Tax brackets not defined for this jurisdiction",
            }
        
        # Calculate tax using progressive brackets
        total_tax = 0.0
        remaining_income = taxable_income
        marginal_rate = 0.0
        
        for bracket in brackets:
            bracket_min = bracket["min"]
            bracket_max = bracket["max"]
            bracket_rate = bracket["rate"]
            
            if remaining_income <= 0:
                break
            
            if taxable_income > bracket_min:
                taxable_in_bracket = min(
                    remaining_income,
                    bracket_max - bracket_min if bracket_max != float("inf") else remaining_income
                )
                tax_in_bracket = taxable_in_bracket * bracket_rate
                total_tax += tax_in_bracket
                remaining_income -= taxable_in_bracket
                
                # Marginal rate is the highest bracket the income falls into
                if taxable_income >= bracket_min and taxable_income < bracket_max:
                    marginal_rate = bracket_rate
        
        effective_rate = (total_tax / gross_income * 100) if gross_income > 0 else 0.0
        
        return {
            "gross_income": gross_income,
            "total_deductions": total_deductions,
            "total_exemptions": total_exemptions,
            "taxable_income": taxable_income,
            "total_tax": round(total_tax, 2),
            "net_income": round(gross_income - total_tax, 2),
            "effective_rate": round(effective_rate, 2),
            "marginal_rate": round(marginal_rate * 100, 2),
            "jurisdiction": self.jurisdiction.value,
            "tax_year": datetime.now().year,
        }
    
    def get_deductible_categories(self) -> List[str]:
        """Get list of tax-deductible expense categories for this jurisdiction"""
        return self.TAX_EXEMPT_CATEGORIES.get(self.jurisdiction, [])
    
    def estimate_tax_from_expenses(
        self,
        income: float,
        expenses: List[Dict[str, Any]],
        jurisdiction: Optional[TaxJurisdiction] = None,
    ) -> Dict[str, Any]:
        """
        Estimate tax liability considering deductible expenses
        
        Args:
            income: Annual income
            expenses: List of expense dictionaries with 'category' and 'amount'
            jurisdiction: Override default jurisdiction
        """
        if jurisdiction:
            self.jurisdiction = jurisdiction
        
        deductible_categories = self.get_deductible_categories()
        
        # Calculate deductible expenses
        deductible_expenses = {}
        for expense in expenses:
            category = expense.get("category", "")
            amount = expense.get("amount", 0)
            
            if category in deductible_categories:
                if category not in deductible_expenses:
                    deductible_expenses[category] = 0
                deductible_expenses[category] += amount
        
        # Calculate tax with deductions
        return self.calculate_income_tax(
            gross_income=income,
            deductions=deductible_expenses,
        )


# Global instance
tax_engine = TaxRuleEngine()
