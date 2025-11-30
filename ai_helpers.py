import os
from typing import Tuple, List, Dict, Any
from groq import Groq


def _get_groq_client() -> Groq:
    """Get Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    return Groq(api_key=api_key)


async def suggest_category(merchant: str, note: str, amount: float) -> Tuple[str, float]:
    """Suggest expense category using AI"""
    try:
        client = _get_groq_client()
        
        prompt = f"""Given the following expense details, suggest the most appropriate category.
Merchant: {merchant}
Description: {note}
Amount: ${amount:.2f}

Common categories: Food, Transportation, Shopping, Bills, Entertainment, Healthcare, Education, Travel, Utilities, Subscriptions, Other

Respond with ONLY the category name (one word, capitalized)."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a financial categorization assistant. Respond with only the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        category = response.choices[0].message.content.strip()
        # Default categories if AI returns something unexpected
        valid_categories = ["Food", "Transportation", "Shopping", "Bills", "Entertainment", 
                           "Healthcare", "Education", "Travel", "Utilities", "Subscriptions", "Other"]
        
        if category not in valid_categories:
            category = "Other"
        
        # Simple confidence based on how specific the input is
        confidence = 0.7 if merchant or note else 0.5
        
        return category, confidence
    except Exception as e:
        # Fallback to "Other" if AI fails
        return "Other", 0.3


async def generate_insights(expense_data: List[Dict[str, Any]]) -> str:
    """Generate AI-powered insights from expense data"""
    try:
        if not expense_data:
            return "No expenses found. Start tracking your expenses to get insights!"
        
        client = _get_groq_client()
        
        # Prepare summary data
        total = sum(float(e.get("amount", 0)) for e in expense_data)
        categories = {}
        for e in expense_data:
            cat = e.get("category", "Other")
            categories[cat] = categories.get(cat, 0) + float(e.get("amount", 0))
        
        top_category = max(categories.items(), key=lambda x: x[1]) if categories else ("None", 0)
        
        prompt = f"""Analyze the following expense data and provide 3-5 actionable insights:

Total Expenses: ${total:.2f}
Number of Transactions: {len(expense_data)}
Top Spending Category: {top_category[0]} (${top_category[1]:.2f})
Category Breakdown: {', '.join(f"{k}: ${v:.2f}" for k, v in list(categories.items())[:5])}

Provide insights in a friendly, actionable format. Focus on:
1. Spending patterns
2. Potential savings opportunities
3. Budget recommendations
4. Category-specific advice

Keep it concise (3-5 bullet points)."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a financial advisor providing helpful spending insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        insights = response.choices[0].message.content.strip()
        return insights
    except Exception as e:
        # Fallback insights if AI fails
        if expense_data:
            total = sum(float(e.get("amount", 0)) for e in expense_data)
            return f"📊 Total expenses: ${total:.2f} across {len(expense_data)} transactions. Consider reviewing your spending patterns to identify savings opportunities."
        return "Unable to generate insights at this time. Please try again later."

