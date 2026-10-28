"""
Household-level features from the ReCon SL Wave 1 survey, and the consumption target.

Timing rule: every feature is an answer from the Wave 1 household survey, so it is known
before any bill is seen. Answers that could contain the bill are excluded
(total monthly expenditure; Wave 3 billing table; solar units generated).
"""
import numpy as np
import pandas as pd

from config import RAW, THRESHOLD_UNITS, WINDOW_MONTHS, MIN_VALID_MONTHS

# appliance_type values in w1_appliances.csv grouped into the loads that matter for a bill
APPL = {
    "n_fridges": ["Refrigerator"],
    "n_freezers": ["Separate Freezer"],
    "n_washing_machines": ["Washing Machine"],
    "n_tvs": ["TV"],
    "n_computers": ["Laptops", "Computers"],
    "n_irons": ["Electric Iron including electric steam iron"],
    "n_rice_cookers": ["Rice cooker"],
    "n_electric_cooking": ["Electric Oven", "Microwave",
                           "Electric cook tops (induction cookers, Infra-red cookers, hot plates)"],
    "n_water_heaters": ["Geyser / Hot water systems for bathrooms which operate from electricity",
                        "Electric water heater to heat water for drinking purposes"],
    "n_water_pumps": ["Electric Water pump"],
    "n_kettles": ["Electric Kettle"],
}

HOUSE_TYPE = {"Single House - Single Floor": "single floor", "Single House - Double Floor": "two floors",
              "Single House - More than 2 floors": "3+ floors", "Flat": "flat/apartment",
              "Condominium/ Luxury apartments": "flat/apartment"}
BUILT = {"Before 1980": 0, "1980-1989": 1, "1990-1999": 2, "2000-2009": 3, "2010-2019": 4, "In 2020 or After 2020": 5}
SEC = {"SEC A": "A", "SEC B": "B", "SEC C": "C", "SEC D": "D", "SEC E": "E"}
HOT_WATER = {
    "We have an in-built water heating system powered solely by the grid supply.": "electric (grid)",
    "We have an in-built water heating system powered both by solar and the grid supply.": "solar + grid",
    "We have an in-built water heating system powered solely by solar energy.": "solar only",
    "We don't have an inbuilt system for water heating, we use water heated through an electric kettle/heater.": "kettle/heater",
    "None, we do not use hot water for bathing or body wash purposes.": "none",
}

NUMERIC = ["household_members", "children_under_15", "adults_65_plus", "members_working", "avg_hours_at_home",
           "floor_area_sqft", "rooms", "bedrooms", "built_period",
           "n_acs", "ac_hours_week", "n_inverter_acs", "n_fans", "fan_hours_week",
           "n_bulbs", "lighting_kwh_week", "n_appliances", *APPL.keys()]
FLAGS = ["solar", "solar_on_grid", "backup_generator", "electric_cooking", "home_business", "renting"]
CATEG = ["house_type", "socio_economic_class", "hot_water", "area"]
FEATURES = NUMERIC + FLAGS + CATEG

LABELS = {  # human-readable names for figures and the app
    "household_members": "household members", "children_under_15": "children under 15",
    "adults_65_plus": "adults 65+", "members_working": "members working", "avg_hours_at_home": "avg hours at home / week",
    "floor_area_sqft": "floor area (sq ft)", "rooms": "rooms", "bedrooms": "bedrooms",
    "built_period": "built period (0 = pre-1980 … 5 = 2020+)", "n_acs": "air conditioners", "ac_hours_week": "AC hours / week",
    "n_inverter_acs": "inverter ACs", "n_fans": "fans", "fan_hours_week": "fan hours / week", "n_bulbs": "bulbs",
    "lighting_kwh_week": "lighting kWh / week", "n_appliances": "appliances (total)", "n_fridges": "refrigerators",
    "n_freezers": "freezers", "n_washing_machines": "washing machines", "n_tvs": "TVs", "n_computers": "computers/laptops",
    "n_irons": "irons", "n_rice_cookers": "rice cookers", "n_electric_cooking": "oven/microwave/cooktop",
    "n_water_heaters": "electric water heaters", "n_water_pumps": "water pumps", "n_kettles": "kettles",
    "solar": "has solar", "solar_on_grid": "solar on-grid", "backup_generator": "backup generator",
    "electric_cooking": "cooks with grid electricity", "home_business": "business at home", "renting": "renting",
    "house_type": "house type", "socio_economic_class": "socio-economic class", "hot_water": "hot water for bathing",
    "area": "CEB/LECO service area",
}


def _yes(s):
    return s.astype(str).str.strip().str.lower().eq("yes").astype(int)


def load_raw():
    r = lambda f: pd.read_csv(RAW / f, low_memory=False)
    return {k: r(f) for k, f in {
        "hh": "w1_household_information_and_history.csv", "gen": "w1_electricity_generation_water_heating_cooking.csv",
        "dem": "w1_demographics.csv", "app": "w1_appliances.csv", "ac": "w1_ac_roster.csv", "fan": "w1_fan_roster.csv",
        "light": "w1_light_roster.csv", "room": "w1_room_roster.csv", "dates": "survey_dates.csv",
        "cons": "monthly_consumption.csv"}.items()}


def build_features(t):
    hh = t["hh"].set_index("household_ID")
    X = pd.DataFrame(index=hh.index)

    d = t["dem"].copy()
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    g = d.groupby("household_ID")
    X["household_members"] = hh["no_of_household_members"]
    X["children_under_15"] = g["age"].apply(lambda a: (a < 15).sum())
    X["adults_65_plus"] = g["age"].apply(lambda a: (a >= 65).sum())
    X["members_working"] = g["member_went_out_for_work_or_not_during_last_week"].apply(lambda s: s.astype(str).str.startswith("Yes").sum())
    X["avg_hours_at_home"] = g["no_of_hours_stayed_at_home_during_last_week"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").clip(0, 168).mean())

    X["floor_area_sqft"] = hh["floor_area"]
    rm = t["room"].groupby("household_ID")
    X["rooms"] = rm.size()
    X["bedrooms"] = rm["main_purpose_of_the_room"].apply(lambda s: s.astype(str).str.contains("bed", case=False).sum())
    X["built_period"] = hh["built_year_of_the_house"].map(BUILT)

    ac = t["ac"].copy()
    hrs = lambda df, a, b: pd.to_numeric(df[a], errors="coerce").fillna(0) + pd.to_numeric(df[b], errors="coerce").fillna(0)
    ac["h"] = hrs(ac, "no_of_hours_ac_was_on_during_daytime_last_week", "no_of_hours_ac_was_on_during_night_last_week")
    ag = ac.groupby("household_ID")
    X["n_acs"] = ag.size()
    X["ac_hours_week"] = ag["h"].sum()
    X["n_inverter_acs"] = ag["is_the_ac_inverter_or_not"].apply(lambda s: s.astype(str).eq("Yes").sum())

    fan = t["fan"].copy()
    fan["h"] = hrs(fan, "no_of_hours_fan_was_on_during_daytime_last_week", "no_of_hours_fan_was_on_during_night_last_week")
    X["n_fans"] = fan.groupby("household_ID").size()
    X["fan_hours_week"] = fan.groupby("household_ID")["h"].sum()

    li = t["light"].copy()
    li["h"] = hrs(li, "no_of_hours_bulb_was_on_during_daytime_last_week", "no_of_hours_bulb_was_on_during_night_last_week")
    li["kwh"] = li["h"].clip(0, 168) * pd.to_numeric(li["wattage_of_the_bulb"], errors="coerce").clip(0, 200).fillna(0) / 1000
    X["n_bulbs"] = li.groupby("household_ID").size()
    X["lighting_kwh_week"] = li.groupby("household_ID")["kwh"].sum()

    ap = t["app"]
    X["n_appliances"] = ap.groupby("household_ID").size()
    for col, kinds in APPL.items():
        X[col] = ap[ap.appliance_type.isin(kinds)].groupby("household_ID").size()

    gen = t["gen"].set_index("household_ID")
    X["solar"] = _yes(gen["generate_electicity_using_solar_energy"])
    X["solar_on_grid"] = _yes(gen["solar_system_ongrid_or_offgird"])
    X["backup_generator"] = _yes(gen["have_backup_generator"])
    X["electric_cooking"] = _yes(gen["electricity_from_national_grid_used_for_cooking"])
    X["home_business"] = _yes(hh["is_there_business_carried_out_in_the_household"])
    X["renting"] = hh["own_the_house_or_living_on_rent"].astype(str).str.startswith("No, I am living on rent").astype(int)

    X["house_type"] = hh["type_of_house"].map(HOUSE_TYPE).fillna("other")
    X["socio_economic_class"] = hh["socio_economic_class"].map(SEC).fillna("unknown")
    X["hot_water"] = gen["water_heating_method_for_bathing"].map(HOT_WATER).fillna("non-electric/other")
    area = hh["electricity_provider_csc_area"].astype(str).str.title()
    top = area.value_counts().index[:12]
    X["area"] = area.where(area.isin(top), "Other")

    cnt = ["rooms", "bedrooms", "n_acs", "ac_hours_week", "n_inverter_acs", "n_fans", "fan_hours_week", "n_bulbs",
           "lighting_kwh_week", "n_appliances", *APPL.keys(), "children_under_15", "adults_65_plus", "members_working"]
    X[cnt] = X[cnt].fillna(0)
    return X[FEATURES]


def build_target(t):
    """Median monthly kWh over the 12 months up to the Wave 1 survey month; label = above 90 units."""
    m = t["cons"].copy()
    m["kwh"] = pd.to_numeric(m["consumption"].astype(str).str.replace(",", ""), errors="coerce")
    m["month"] = pd.to_datetime(m["month"])
    s = t["dates"].assign(w1=lambda d: pd.to_datetime(d["wave1"]))[["household_ID", "w1"]]
    m = m.merge(s, on="household_ID")
    end = m.w1 + pd.offsets.MonthEnd(0)
    win = m[(m.month <= end) & (m.month > end - pd.DateOffset(months=WINDOW_MONTHS))]
    g = win.groupby("household_ID")["kwh"]
    info = pd.DataFrame({"valid": g.apply(lambda k: (k > 0).sum()), "zero_share": g.apply(lambda k: (k == 0).mean()),
                         "median_kwh": win[win.kwh > 0].groupby("household_ID")["kwh"].median()})
    keep = (info.valid >= MIN_VALID_MONTHS) & (info.zero_share <= 0.5)
    info["high_user"] = (info.median_kwh > THRESHOLD_UNITS).astype(int)
    return info, keep
