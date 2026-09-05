"""
Kisan Dost - "Farmer's Friend"
================================
A multi-agent AI advisory system for Pakistani smallholder farmers,
built on the OpenAI Agents SDK.

Architecture
------------
Triage Agent (routes the farmer's question)
   |-- Agronomy Agent   -> crop advice, fertilizer plan, irrigation advice
   |-- Pest Doctor Agent -> pest/disease diagnosis + SAFE treatment dosage
   |-- Market Agent      -> mandi price lookup, profit estimation
   |-- Finance Agent     -> govt schemes, profit estimation

Guardrails
----------
- Input guardrail: rejects off-topic or unsafe requests (e.g. human medical
  advice, requests to misuse chemicals).
- Output guardrail: keeps the Pest Doctor's pesticide dosage recommendations
  within hardcoded safe limits and blocks any human-medical-advice language.

Sessions & Context
-------------------
A typed FarmerProfile is carried as context across the whole run, and a
SQLiteSession remembers the conversation so the farmer doesn't have to repeat
their district / land size / crop every message.

Bonus features included
------------------------
- Real weather: irrigation_weather_advisor calls the free Open-Meteo API
  (no key required) instead of a mock forecast string.
- Tracing: every Runner.run() call is wrapped in a named `trace(...)` block
  so you can inspect the run in the OpenAI traces dashboard.

Install:
    pip install openai-agents pydantic requests
Set your key (PowerShell):
    $env:OPENAI_API_KEY="your-key-here"
Run:
    python hykathon.py
"""


import asyncio
import os
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    SQLiteSession,
    function_tool,
    input_guardrail,
    output_guardrail,
    trace,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

MODEL = "gpt-4o-mini"  # swap for whichever model string you have access to


# =========================================================================
# 1. CONTEXT — carried across every turn without re-typing details
# =========================================================================

@dataclass
class FarmerProfile:
    district: str = "Unknown"
    land_size_acres: float = 1.0
    water_availability: str = "medium"   # low / medium / high
    season: str = "Rabi"                 # Rabi / Kharif
    current_crop: Optional[str] = None
    notes: List[str] = field(default_factory=list)


# =========================================================================
# 2. STRUCTURED OUTPUTS — Pydantic models instead of loose text
# =========================================================================

class CropPlan(BaseModel):
    recommended_crops: List[str] = Field(description="Ranked crop options")
    expected_yield_maund_per_acre: float
    expected_profit_pkr_per_acre: int
    reasoning: str


class FertilizerPlan(BaseModel):
    crop: str
    acres: float
    urea_bags: float
    dap_bags: float
    total_cost_pkr: int
    application_schedule: str


class PestDiagnosis(BaseModel):
    likely_pest_or_disease: str
    confidence: str  # low / medium / high
    treatment_name: str
    safe_dosage: str
    safety_note: str


class MandiPrice(BaseModel):
    crop: str
    mandi_name: str
    price_pkr_per_40kg: int
    trend: str  # rising / falling / stable
    advice: str


class IrrigationAdvice(BaseModel):
    should_irrigate_now: bool
    next_irrigation_in_days: int
    frost_risk: bool
    heatwave_risk: bool
    note: str


class ProfitEstimate(BaseModel):
    total_cost_pkr: int
    total_revenue_pkr: int
    net_margin_pkr: int
    break_even_yield_maund_per_acre: float


class GovtScheme(BaseModel):
    scheme_name: str
    eligibility: str
    benefit: str
    how_to_apply: str


# =========================================================================
# 3. FUNCTION TOOLS — one clean, single-responsibility tool per capability
# =========================================================================

_CROP_TABLE = {
    ("Rabi", "low"): [("wheat", 32, 45000), ("gram (chana)", 14, 38000)],
    ("Rabi", "medium"): [("wheat", 40, 58000), ("mustard", 18, 42000)],
    ("Rabi", "high"): [("wheat", 45, 65000), ("potato", 200, 90000)],
    ("Kharif", "low"): [("sorghum (jowar)", 20, 30000), ("mung bean", 10, 35000)],
    ("Kharif", "medium"): [("cotton", 25, 70000), ("maize", 45, 55000)],
    ("Kharif", "high"): [("rice", 50, 85000), ("sugarcane", 700, 95000)],
}


@function_tool
def crop_advisor(
    district: str,
    soil_type: str,
    season: str,
    water_availability: str,
    land_size_acres: float,
) -> CropPlan:
    """Recommend the best crops for a farmer's district, soil, season, water
    availability and land size, with expected yield and profit per acre.

    Args:
        district: Farmer's district, e.g. 'Multan'.
        soil_type: e.g. 'loamy', 'sandy', 'clay'.
        season: 'Rabi' or 'Kharif'.
        water_availability: 'low', 'medium', or 'high'.
        land_size_acres: Total cultivable land in acres.
    """
    key = (season, water_availability.lower())
    options = _CROP_TABLE.get(key, _CROP_TABLE[("Rabi", "medium")])
    crops = [c[0] for c in options]
    best = options[0]
    return CropPlan(
        recommended_crops=crops,
        expected_yield_maund_per_acre=best[1],
        expected_profit_pkr_per_acre=best[2],
        reasoning=(
            f"Based on {season} season with {water_availability} water availability "
            f"in {district} ({soil_type} soil), these crops historically perform best "
            f"for {land_size_acres} acres."
        ),
    )


_NPK_PER_ACRE = {
    "wheat": {"n": 45, "p": 30, "k": 0},
    "cotton": {"n": 60, "p": 30, "k": 30},
    "rice": {"n": 55, "p": 25, "k": 0},
    "maize": {"n": 60, "p": 25, "k": 25},
    "sugarcane": {"n": 100, "p": 50, "k": 50},
}
UREA_PRICE_PKR_PER_BAG = 3200   # 50kg bag, 46% N
DAP_PRICE_PKR_PER_BAG = 11500   # 50kg bag, 18% N + 46% P


@function_tool
def fertilizer_calculator(crop: str, acres: float) -> FertilizerPlan:
    """Compute NPK requirement for a crop and convert it into bags of Urea
    and DAP with total cost in PKR.

    Args:
        crop: Crop name, e.g. 'wheat', 'cotton'.
        acres: Land size in acres.
    """
    npk = _NPK_PER_ACRE.get(crop.lower(), {"n": 50, "p": 25, "k": 0})
    total_n = npk["n"] * acres
    total_p = npk["p"] * acres

    # DAP supplies both N (18%) and P (46%); remaining N comes from Urea (46%)
    dap_bags = round(total_p / (0.46 * 50), 1)
    n_from_dap = dap_bags * 50 * 0.18
    remaining_n = max(total_n - n_from_dap, 0)
    urea_bags = round(remaining_n / (0.46 * 50), 1)

    total_cost = round(urea_bags * UREA_PRICE_PKR_PER_BAG + dap_bags * DAP_PRICE_PKR_PER_BAG)

    return FertilizerPlan(
        crop=crop,
        acres=acres,
        urea_bags=urea_bags,
        dap_bags=dap_bags,
        total_cost_pkr=total_cost,
        application_schedule=(
            "Apply full DAP + 1/3 Urea at sowing; remaining Urea split between "
            "first and second irrigation."
        ),
    )


# Hardcoded safe-dosage table (ml or g of product per acre) — the output
# guardrail below checks against this table so nothing unsafe ever ships.
_SAFE_DOSAGE_LIMITS = {
    "imidacloprid": "80-100 ml/acre",
    "cypermethrin": "150-200 ml/acre",
    "mancozeb": "600-800 g/acre",
    "chlorpyrifos": "500-600 ml/acre",
}

_PEST_LIBRARY = {
    "whitefly": ("imidacloprid", "Tiny white flying insects under leaves, sooty mould, cotton/vegetable crops."),
    "aphid": ("imidacloprid", "Small green/black insects clustering on new growth, curling leaves."),
    "bollworm": ("cypermethrin", "Holes in cotton bolls or fruit, caterpillar damage."),
    "rust": ("mancozeb", "Orange/brown powdery spots on wheat leaves."),
    "termite": ("chlorpyrifos", "Wilting plants, hollow stems, soil tunnels near roots."),
}


@function_tool
def pest_disease_doctor(symptom_description: str, crop: str) -> PestDiagnosis:
    """Diagnose the likely pest or disease from a farmer's plain-language
    description of symptoms, and return a treatment name with a SAFE dosage.
    Never provides human medical advice — this is for crop pests/diseases only.

    Args:
        symptom_description: What the farmer observed, e.g. 'leaves curling, tiny white insects'.
        crop: The affected crop, e.g. 'cotton'.
    """
    desc = symptom_description.lower()
    match = None
    for pest, (treatment, _) in _PEST_LIBRARY.items():
        if pest in desc or any(word in desc for word in pest.split()):
            match = pest
            break

    if not match:
        if "white" in desc and "insect" in desc:
            match = "whitefly"
        elif "hole" in desc:
            match = "bollworm"
        elif "orange" in desc or "rust" in desc:
            match = "rust"
        else:
            match = "aphid"

    treatment = _PEST_LIBRARY[match][0]
    dosage = _SAFE_DOSAGE_LIMITS[treatment]

    return PestDiagnosis(
        likely_pest_or_disease=match,
        confidence="medium",
        treatment_name=treatment,
        safe_dosage=dosage,
        safety_note=(
            "Spray in the evening, wear a mask and gloves, keep children and "
            "animals away for 24 hours, and do not exceed the labeled dosage."
        ),
    )


_MANDI_PRICES = {
    "wheat": 3200,
    "cotton": 8500,
    "sugarcane": 400,
    "rice": 4200,
    "maize": 2800,
    "potato": 2200,
    "mustard": 5600,
}


@function_tool
def mandi_price_lookup(crop: str, district: str) -> MandiPrice:
    """Return the typical current wholesale (mandi) price for a crop near
    the farmer's district, in PKR per 40kg.

    Args:
        crop: Crop name, e.g. 'wheat'.
        district: Farmer's district, used to name the nearest mandi.
    """
    price = _MANDI_PRICES.get(crop.lower(), 3000)
    return MandiPrice(
        crop=crop,
        mandi_name=f"{district} Mandi",
        price_pkr_per_40kg=price,
        trend="stable",
        advice="Prices are close to seasonal average — selling now is reasonable, "
        "but check 2-3 nearby mandis before committing a full load.",
    )


def _fetch_live_forecast(district: str) -> dict:
    """Real weather via Open-Meteo (free, no API key). Returns a small dict
    with next-3-day max/min temps and rain, or an error note on failure."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": district, "count": 1},
            timeout=8,
        ).json()
        if not geo.get("results"):
            return {"error": f"Location '{district}' not found"}
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=8,
        ).json()
        daily = wx.get("daily", {})
        return {
            "max_temps": daily.get("temperature_2m_max", []),
            "min_temps": daily.get("temperature_2m_min", []),
            "rain_mm": daily.get("precipitation_sum", []),
        }
    except Exception as e:
        return {"error": str(e)}


@function_tool
def irrigation_weather_advisor(
    district: str, crop_stage: str, days_since_last_rain: int
) -> IrrigationAdvice:
    """Advise on irrigation timing and flag frost/heatwave risk, using a real
    3-day weather forecast for the farmer's district plus crop growth stage.

    Args:
        district: Farmer's district, e.g. 'Multan', used to fetch live weather.
        crop_stage: e.g. 'seedling', 'flowering', 'grain filling'.
        days_since_last_rain: Number of days since the last rainfall.
    """
    forecast = _fetch_live_forecast(district)

    max_temps = forecast.get("max_temps", [])
    min_temps = forecast.get("min_temps", [])
    rain_mm = forecast.get("rain_mm", [])

    frost_risk = any(t <= 2 for t in min_temps) if min_temps else False
    heatwave_risk = any(t >= 40 for t in max_temps) if max_temps else False
    rain_coming = any(r >= 5 for r in rain_mm) if rain_mm else False

    should_irrigate = (
        not rain_coming
        and (days_since_last_rain >= 7 or crop_stage.lower() in ("flowering", "grain filling"))
    )
    next_days = 2 if should_irrigate else max(7 - days_since_last_rain, 1)

    if "error" in forecast:
        note = f"Live forecast unavailable ({forecast['error']}) — advice based on crop stage only."
    else:
        note = (
            "Irrigate soon — critical growth stage or dry soil."
            if should_irrigate
            else "Soil moisture likely adequate — hold off a few days."
        )
        if rain_coming:
            note += " Rain expected in the next 3 days — you can delay irrigation."
        if heatwave_risk:
            note += " Heatwave expected (40°C+) — irrigate early morning or evening to reduce stress."
        if frost_risk:
            note += " Frost risk (near 0°C) — a light irrigation the evening before can protect the crop."

    return IrrigationAdvice(
        should_irrigate_now=should_irrigate,
        next_irrigation_in_days=next_days,
        frost_risk=frost_risk,
        heatwave_risk=heatwave_risk,
        note=note,
    )


@function_tool
def profit_estimator(
    crop: str, acres: float, expected_yield_maund_per_acre: float,
    price_pkr_per_40kg: int, input_cost_pkr_per_acre: int,
) -> ProfitEstimate:
    """Full-season profit estimate: total input cost vs expected revenue,
    net margin, and the break-even yield needed.

    Args:
        crop: Crop name.
        acres: Land size in acres.
        expected_yield_maund_per_acre: Expected yield in maunds (40kg units) per acre.
        price_pkr_per_40kg: Expected/current mandi price per 40kg.
        input_cost_pkr_per_acre: Total input cost (seed+fertilizer+labor+etc) per acre.
    """
    total_cost = round(input_cost_pkr_per_acre * acres)
    total_revenue = round(expected_yield_maund_per_acre * acres * price_pkr_per_40kg)
    net_margin = total_revenue - total_cost
    break_even_yield = round(input_cost_pkr_per_acre / price_pkr_per_40kg, 2) if price_pkr_per_40kg else 0

    return ProfitEstimate(
        total_cost_pkr=total_cost,
        total_revenue_pkr=total_revenue,
        net_margin_pkr=net_margin,
        break_even_yield_maund_per_acre=break_even_yield,
    )


_GOVT_SCHEMES = {
    "punjab": [
        ("Kisan Card (Punjab)", "Small/medium farmers with <25 acres in Punjab",
         "Subsidized loans, fertilizer and seed at discounted rates",
         "Register at your nearest Punjab Agriculture Dept office with CNIC and land record."),
    ],
    "sindh": [
        ("Sindh Agri Loan Scheme", "Registered farmers in Sindh",
         "Low-interest agri loans for inputs and machinery",
         "Apply via Zarai Taraqiati Bank Limited (ZTBL) branch with land ownership documents."),
    ],
    "kpk": [
        ("KP Agri Support Program", "Farmers in Khyber Pakhtunkhwa",
         "Subsidized seed/fertilizer and small machinery grants",
         "Contact district agriculture extension office."),
    ],
}


@function_tool
def govt_support_finder(province: str, need: str) -> GovtScheme:
    """Surface a relevant government support scheme (Kisan Card, subsidized
    fertilizer, or agri-loan) for the farmer's province and stated need.

    Args:
        province: 'Punjab', 'Sindh', or 'KPK'.
        need: What the farmer needs help with, e.g. 'fertilizer subsidy', 'loan'.
    """
    key = province.strip().lower()
    options = _GOVT_SCHEMES.get(key, _GOVT_SCHEMES["punjab"])
    name, eligibility, benefit, how_to_apply = options[0]
    return GovtScheme(
        scheme_name=name,
        eligibility=eligibility,
        benefit=benefit,
        how_to_apply=how_to_apply,
    )


# =========================================================================
# 4. GUARDRAILS
# =========================================================================

class TopicSafetyCheck(BaseModel):
    is_farming_related: bool
    is_safe: bool
    reasoning: str


guardrail_agent = Agent(
    name="Guardrail Check",
    model=MODEL,
    instructions=(
        "Classify the user's message. is_farming_related should be true only if "
        "the message is about crops, pests, fertilizer, mandi prices, irrigation, "
        "weather for farming, profit/finance for farming, or govt agri schemes. "
        "is_safe should be false if the user is asking for human medical advice/"
        "diagnosis, or asking how to misuse chemicals/pesticides to harm people, "
        "animals, or the environment."
    ),
    output_type=TopicSafetyCheck,
)


@input_guardrail
async def farming_topic_guardrail(
    ctx: RunContextWrapper[FarmerProfile], agent: Agent, input_data
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    check = result.final_output
    tripwire = (not check.is_farming_related) or (not check.is_safe)
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=tripwire)


@output_guardrail
async def pesticide_safety_guardrail(
    ctx: RunContextWrapper[FarmerProfile], agent: Agent, output: PestDiagnosis
) -> GuardrailFunctionOutput:
    """Blocks the Pest Doctor's output if the recommended dosage doesn't match
    the hardcoded safe-dosage table, or if the text strays into human-medical
    advice territory."""
    treatment = output.treatment_name.lower()
    known_safe = _SAFE_DOSAGE_LIMITS.get(treatment)

    unsafe = False
    reason = "OK"

    if known_safe is None:
        unsafe = True
        reason = f"'{treatment}' is not in the approved safe-dosage table."
    elif output.safe_dosage.strip() != known_safe:
        unsafe = True
        reason = f"Dosage '{output.safe_dosage}' does not match approved safe dosage '{known_safe}'."

    human_medical_terms = ["human dose", "for humans", "swallow", "ingest to treat", "human illness"]
    if any(term in output.safety_note.lower() for term in human_medical_terms):
        unsafe = True
        reason = "Output strayed into human-medical advice territory."

    return GuardrailFunctionOutput(output_info={"reason": reason}, tripwire_triggered=unsafe)


# =========================================================================
# 5. SPECIALIST AGENTS
# =========================================================================

agronomy_agent = Agent[FarmerProfile](
    name="Agronomy Agent",
    model=MODEL,
    instructions=(
        "Tum agronomy expert ho. Crop selection, fertilizer planning, aur irrigation "
        "timing mein madad karo. Roman Urdu/English mix mein practical jawab do. "
        "Hamesha relevant tool call karo takay numbers sahih hon."
    ),
    tools=[crop_advisor, fertilizer_calculator, irrigation_weather_advisor],
)

pest_doctor_agent = Agent[FarmerProfile](
    name="Pest Doctor Agent",
    model=MODEL,
    instructions=(
        "Tum crop pest & disease specialist ho. Farmer ke symptoms sunno aur "
        "pest_disease_doctor tool call karo. SIRF crop-related pest/disease advice do, "
        "kabhi bhi human medical advice mat do."
    ),
    tools=[pest_disease_doctor],
    output_type=PestDiagnosis,
    output_guardrails=[pesticide_safety_guardrail],
)

market_agent = Agent[FarmerProfile](
    name="Market Agent",
    model=MODEL,
    instructions=(
        "Tum mandi/market expert ho. Crop prices aur profit estimates do. "
        "Tools use karo taake numbers accurate hon."
    ),
    tools=[mandi_price_lookup, profit_estimator],
)

finance_agent = Agent[FarmerProfile](
    name="Finance Agent",
    model=MODEL,
    instructions=(
        "Tum agri-finance aur govt schemes expert ho. Kisan Card, subsidies, aur "
        "loans ke baray mein batao, aur zaroorat parhay to profit_estimator bhi use karo."
    ),
    tools=[govt_support_finder, profit_estimator],
)

triage_agent = Agent[FarmerProfile](
    name="Kisan Dost Triage",
    model=MODEL,
    instructions=(
        "Tum 'Kisan Dost' ho - farmer ke sawal ko sun kar sahi specialist agent ko "
        "route karo: crop/fertilizer/irrigation -> Agronomy Agent; pest/disease -> "
        "Pest Doctor Agent; prices/selling -> Market Agent; govt schemes/loans/overall "
        "profit -> Finance Agent. Agar sawal simple ho to khud bhi friendly jawab de sakte ho."
    ),
    handoffs=[agronomy_agent, pest_doctor_agent, market_agent, finance_agent],
    input_guardrails=[farming_topic_guardrail],
)


# =========================================================================
# 6. MAIN LOOP — Sessions + typed FarmerProfile context
# =========================================================================

async def main():
    print("=" * 55)
    print("  Kisan Dost - Multi-Agent Farming Advisory System")
    print("  (type 'exit' to quit)")
    print("=" * 55)

    district = input("Aap ka district: ").strip() or "Multan"
    land_size = input("Zameen (acres): ").strip()
    water = input("Paani ki availability (low/medium/high): ").strip() or "medium"
    season = input("Season (Rabi/Kharif): ").strip() or "Rabi"

    profile = FarmerProfile(
        district=district,
        land_size_acres=float(land_size) if land_size else 5.0,
        water_availability=water,
        season=season,
    )

    session = SQLiteSession("kisan_dost_session")

    while True:
        user_input = input("\nAap: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Kisan Dost: Allah Hafiz! Khush raho, fasal acchi ho. 🌾")
            break
        if not user_input:
            continue

        try:
            with trace(workflow_name="Kisan Dost Turn", group_id=profile.district):
                result = await Runner.run(
                    triage_agent,
                    user_input,
                    context=profile,
                    session=session,
                )
            print(f"\nKisan Dost: {result.final_output}")
        except InputGuardrailTripwireTriggered:
            print(
                "\nKisan Dost: Maazrat, yeh sawal farming se related nahi lagta ya "
                "unsafe hai — main sirf kheti-baari mein madad kar sakta hoon."
            )
        except OutputGuardrailTripwireTriggered:
            print(
                "\nKisan Dost: Maazrat, is jawab mein dosage safety issue tha, "
                "isliye main isay show nahi kar sakta. Local agri expert se raabta karein."
            )


if __name__ == "__main__":
    asyncio.run(main())