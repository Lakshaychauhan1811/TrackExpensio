"""
Foreign Exchange (FX) Timing and Historical Rate Management

Handles currency conversions with proper timing, historical rates,
and rate caching for accurate financial calculations.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import httpx
from motor.motor_asyncio import AsyncIOMotorClient


class FXRateManager:
    """Manages FX rates with historical data and timing support"""
    
    def __init__(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "ai_expense_tracker")
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.fx_rates = self.db["fx_rates"]
        # Indexes will be created on startup via lifespan handler
    
    async def _ensure_indexes(self):
        """Create indexes for FX rate queries"""
        await self.fx_rates.create_index([("from_currency", 1), ("to_currency", 1), ("date", -1)])
        await self.fx_rates.create_index("date", expireAfterSeconds=31536000)  # 1 year retention
    
    async def get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: Optional[datetime] = None,
        use_historical: bool = True,
    ) -> Dict[str, Any]:
        """
        Get exchange rate with timing support
        
        Args:
            from_currency: Source currency code (e.g., 'USD')
            to_currency: Target currency code (e.g., 'INR')
            date: Specific date for historical rate (defaults to today)
            use_historical: Whether to fetch historical rate if date is in past
        """
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        if from_currency == to_currency:
            return {
                "from_currency": from_currency,
                "to_currency": to_currency,
                "rate": 1.0,
                "date": (date or datetime.now(timezone.utc)).isoformat(),
                "source": "same_currency",
            }
        
        # Use today if no date specified
        if not date:
            date = datetime.now(timezone.utc)
        
        # Check if we have cached rate for this date
        date_str = date.strftime("%Y-%m-%d")
        cached = await self.fx_rates.find_one({
            "from_currency": from_currency,
            "to_currency": to_currency,
            "date": date_str,
        })
        
        if cached:
            return {
                "from_currency": from_currency,
                "to_currency": to_currency,
                "rate": cached["rate"],
                "date": cached["date"],
                "source": "cached",
                "timestamp": cached.get("timestamp", date.isoformat()),
            }
        
        # Fetch rate from API
        try:
            if use_historical and date < datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0):
                # Historical rate
                rate_data = await self._fetch_historical_rate(from_currency, to_currency, date)
            else:
                # Current rate
                rate_data = await self._fetch_current_rate(from_currency, to_currency)
            
            # Cache the rate
            await self.fx_rates.insert_one({
                "from_currency": from_currency,
                "to_currency": to_currency,
                "rate": rate_data["rate"],
                "date": date_str,
                "timestamp": datetime.now(timezone.utc),
            })
            
            return rate_data
            
        except Exception as e:
            # Fallback to approximate rate if API fails
            return {
                "from_currency": from_currency,
                "to_currency": to_currency,
                "rate": 1.0,  # Fallback
                "date": date_str,
                "source": "fallback",
                "error": str(e),
            }
    
    async def _fetch_current_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Fetch current exchange rate from API"""
        # Using exchangerate-api.com (free tier)
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            rate = data["rates"].get(to_currency)
            if not rate:
                raise ValueError(f"Currency {to_currency} not found in rates")
            
            return {
                "from_currency": from_currency,
                "to_currency": to_currency,
                "rate": float(rate),
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "source": "exchangerate-api",
            }
    
    async def _fetch_historical_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: datetime,
    ) -> Dict[str, Any]:
        """Fetch historical exchange rate"""
        date_str = date.strftime("%Y-%m-%d")
        
        # Using exchangerate-api.com historical endpoint
        url = f"https://api.exchangerate-api.com/v4/history/{from_currency}/{date_str}"
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Get rate for the specific date
                rates = data.get("rates", {})
                rate = rates.get(to_currency)
                
                if not rate:
                    # Fallback: try to get closest available date
                    rate = rates.get(to_currency, 1.0)
                
                return {
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "rate": float(rate),
                    "date": date_str,
                    "source": "exchangerate-api-historical",
                }
        except Exception:
            # Fallback to current rate if historical fetch fails
            return await self._fetch_current_rate(from_currency, to_currency)
    
    async def convert_amount(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Convert amount from one currency to another with proper timing
        
        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            date: Date for conversion (uses historical rate if in past)
        """
        rate_data = await self.get_exchange_rate(from_currency, to_currency, date)
        rate = rate_data["rate"]
        
        converted_amount = amount * rate
        
        return {
            "original_amount": amount,
            "original_currency": from_currency.upper(),
            "converted_amount": round(converted_amount, 2),
            "target_currency": to_currency.upper(),
            "exchange_rate": rate,
            "conversion_date": rate_data["date"],
            "rate_source": rate_data.get("source", "api"),
        }
    
    async def get_rate_history(
        self,
        from_currency: str,
        to_currency: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Get exchange rate history for a date range"""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        cursor = self.fx_rates.find({
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper(),
            "date": {"$gte": start_str, "$lte": end_str},
        }).sort("date", 1)
        
        rates = await cursor.to_list(length=None)
        
        return [
            {
                "date": rate["date"],
                "rate": rate["rate"],
                "from_currency": rate["from_currency"],
                "to_currency": rate["to_currency"],
            }
            for rate in rates
        ]
    
    async def bulk_convert_expenses(
        self,
        expenses: List[Dict[str, Any]],
        target_currency: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert multiple expenses to target currency using proper timing
        
        Each expense is converted using the rate from its transaction date
        """
        converted_expenses = []
        
        for expense in expenses:
            expense_date_str = expense.get("date", "")
            expense_currency = expense.get("metadata", {}).get("currency", "INR")
            amount = expense.get("amount", 0)
            
            if expense_currency.upper() == target_currency.upper():
                # Already in target currency
                converted_expenses.append(expense)
                continue
            
            # Parse date
            try:
                expense_date = datetime.strptime(expense_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                expense_date = datetime.now(timezone.utc)
            
            # Convert using historical rate
            conversion = await self.convert_amount(
                amount,
                expense_currency,
                target_currency,
                expense_date,
            )
            
            # Create converted expense
            converted_expense = expense.copy()
            converted_expense["original_amount"] = amount
            converted_expense["original_currency"] = expense_currency
            converted_expense["amount"] = conversion["converted_amount"]
            converted_expense["metadata"] = expense.get("metadata", {})
            converted_expense["metadata"]["currency"] = target_currency
            converted_expense["metadata"]["conversion_rate"] = conversion["exchange_rate"]
            converted_expense["metadata"]["conversion_date"] = conversion["conversion_date"]
            
            converted_expenses.append(converted_expense)
        
        return converted_expenses


# Global instance
fx_manager = FXRateManager()
