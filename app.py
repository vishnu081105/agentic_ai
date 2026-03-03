import os
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from openai import OpenAI

load_dotenv()


def web_search(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Fetch web results and normalize fields for downstream use."""
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    results: list[dict[str, str]] = []
    for item in raw_results:
        results.append(
            {
                "title": str(item.get("title", "Untitled")).strip(),
                "url": str(item.get("href", "")).strip(),
                "snippet": str(item.get("body", "")).strip(),
            }
        )
    return results


def format_search_context(results: list[dict[str, str]]) -> str:
    if not results:
        return "No web results were available."

    lines = []
    for idx, item in enumerate(results, start=1):
        lines.append(
            f"{idx}. {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}"
        )
    return "\n\n".join(lines)


def build_prompt(destination: str, days: int, budget: int, search_context: str) -> str:
    return f"""
You are an autonomous travel planning assistant.
Use the web context below to create a realistic and useful plan.

Web context:
{search_context}

Task requirements:
1. Create a day-by-day itinerary for {destination} for exactly {days} days.
2. Suggest attractions and activities for each day.
3. Split the total budget (${budget}) into daily estimates with categories.
4. Include practical local transport, food, and safety tips.
5. Keep recommendations concise and actionable.
""".strip()


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if not output:
        return ""

    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks).strip()


def generate_plan_from_openai(client: OpenAI, prompt: str) -> str:
    # New OpenAI SDKs expose `responses`; older SDKs expose `chat.completions`.
    if hasattr(client, "responses"):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.4,
        )
        text = _extract_responses_text(response)
        if text:
            return text

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = completion.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = str(part.get("text", "")).strip()
                else:
                    text = str(getattr(part, "text", "")).strip()
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts).strip()

    raise RuntimeError("OpenAI returned an empty response.")


def classify_openai_error(exc: Exception) -> str:
    msg = str(exc)
    lower = msg.lower()

    if "insufficient_quota" in lower or "exceeded your current quota" in lower:
        return (
            "OpenAI quota exceeded. Billing/quota is not available for this API key. "
            "A local fallback plan is shown below."
        )
    if "invalid_api_key" in lower or "incorrect api key" in lower:
        return (
            "Invalid OpenAI API key. Check OPENAI_API_KEY in .env. "
            "A local fallback plan is shown below."
        )
    if "rate limit" in lower or "error code: 429" in lower:
        return (
            "OpenAI rate limit reached. Please retry in a moment. "
            "A local fallback plan is shown below."
        )
    if "connection" in lower or "timed out" in lower:
        return (
            "Network error while calling OpenAI. Check internet connection. "
            "A local fallback plan is shown below."
        )
    return f"OpenAI request failed: {msg}. A local fallback plan is shown below."


def extract_attractions(
    destination: str, results: list[dict[str, str]], limit: int = 12
) -> list[str]:
    pool: list[str] = []
    seen: set[str] = set()

    for item in results:
        candidate_titles = [item.get("title", ""), item.get("snippet", "")]
        for text in candidate_titles:
            text = text.strip()
            if not text:
                continue
            cleaned = re.sub(r"\s+", " ", text)
            if len(cleaned) < 8:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(cleaned)
            if len(pool) >= limit:
                return pool

    if pool:
        return pool

    return [
        f"City center walk in {destination}",
        f"Top museum or cultural district in {destination}",
        f"Local food street and market in {destination}",
        f"Popular scenic viewpoint in {destination}",
        f"Nearby day-trip highlight from {destination}",
        f"Evening neighborhood exploration in {destination}",
    ]


def generate_local_fallback_plan(
    destination: str,
    days: int,
    budget: int,
    results: list[dict[str, str]],
    reason: str,
) -> str:
    attractions = extract_attractions(destination, results)

    daily_budget = round(budget / max(days, 1), 2)
    lodging = round(daily_budget * 0.4, 2)
    food = round(daily_budget * 0.25, 2)
    transport = round(daily_budget * 0.15, 2)
    activities = round(daily_budget * 0.15, 2)
    misc = round(daily_budget * 0.05, 2)

    lines = []
    lines.append(f"## {days}-Day Travel Plan for {destination}")
    lines.append("")
    lines.append(f"**Mode:** Local fallback planner")
    lines.append(f"**Why fallback was used:** {reason}")
    lines.append(f"**Total Budget:** ${budget}")
    lines.append(f"**Estimated Daily Budget:** ${daily_budget}")
    lines.append("")
    lines.append("### Daily Budget Split")
    lines.append(
        f"- Lodging: ${lodging} | Food: ${food} | Transport: ${transport} | "
        f"Activities: ${activities} | Misc: ${misc}"
    )
    lines.append("")
    lines.append("### Itinerary")

    for day in range(1, days + 1):
        morning = attractions[(day - 1) % len(attractions)]
        afternoon = attractions[(day * 2 - 1) % len(attractions)]
        evening = attractions[(day * 3 - 1) % len(attractions)]

        lines.append(f"**Day {day}**")
        lines.append(f"- Morning: {morning}")
        lines.append(f"- Afternoon: {afternoon}")
        lines.append(f"- Evening: {evening}")
        lines.append(
            f"- Spend target: ${daily_budget} "
            f"(prioritize bookings for top attractions early)."
        )
        lines.append("")

    lines.append("### Practical Tips")
    lines.append("- Use public transport day passes when possible.")
    lines.append("- Keep 5-10% of budget reserved for price changes.")
    lines.append("- Book high-demand attractions at least 2-7 days in advance.")
    lines.append("- Save offline maps and emergency contact numbers.")
    lines.append("")

    if results:
        lines.append("### Web Sources Snapshot")
        for item in results[:5]:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            lines.append(f"- [{title}]({url})")

    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="Agentic Travel Planner", layout="wide")
    st.title("Agentic AI Travel Planner")
    st.write("Plan your trip with web-informed recommendations.")

    destination = st.text_input("Destination")
    days = st.number_input("Number of Days", min_value=1, max_value=30, value=3)
    budget = st.number_input("Budget (USD)", min_value=100, value=1000, step=100)

    if st.button("Generate Travel Plan"):
        destination = destination.strip()
        if not destination:
            st.error("Please enter a destination.")
            return

        with st.spinner("Researching destination and preparing itinerary..."):
            search_query = (
                f"{destination} attractions travel costs local transport best time to visit"
            )
            results = web_search(search_query)
            search_context = format_search_context(results)
            prompt = build_prompt(destination, int(days), int(budget), search_context)

            plan_text = ""
            fallback_reason = ""
            api_key = os.getenv("OPENAI_API_KEY", "").strip()

            if api_key:
                try:
                    client = get_openai_client(api_key)
                    plan_text = generate_plan_from_openai(client, prompt)
                except Exception as exc:
                    fallback_reason = classify_openai_error(exc)
            else:
                fallback_reason = (
                    "OPENAI_API_KEY is missing in .env. A local fallback plan is shown below."
                )

            if not plan_text:
                plan_text = generate_local_fallback_plan(
                    destination=destination,
                    days=int(days),
                    budget=int(budget),
                    results=results,
                    reason=fallback_reason or "OpenAI returned empty output.",
                )
                st.warning(fallback_reason or "Switched to local fallback planner.")
            else:
                st.success("Trip plan generated with OpenAI.")

        st.markdown(plan_text)


if __name__ == "__main__":
    main()
