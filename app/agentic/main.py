#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from .. import mongo
from datetime import datetime
from openai import OpenAI
import voyageai


open_ai = OpenAI()
voyage_ai = voyageai.Client()


def _format_recent_articles(recent_articles):
    if not recent_articles:
        return "    No recent articles read."

    formatted_lines = ["Last articles read:"]

    for i, article in enumerate(recent_articles, 1):
        title = article.get('title', 'Unknown title')
        if len(title) > 80:
            title = title[:77] + "..."

        sections = article.get('sections', [])
        if sections:
            section_text = f"[{', '.join(sections)}] "
        else:
            section_text = ""

        clicked_at = article.get('clicked_at')
        if clicked_at:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            click_time = clicked_at.replace(tzinfo=timezone.utc)
            time_diff = now - click_time

            if time_diff.days >= 7:
                time_text = f"{time_diff.days} days ago"
            elif time_diff.days >= 1:
                time_text = "yesterday"
            elif time_diff.total_seconds() >= 3600:
                hours = int(time_diff.total_seconds() / 3600)
                time_text = f"{hours} hours ago"
            elif time_diff.total_seconds() >= 60:
                minutes = int(time_diff.total_seconds() / 60)
                time_text = f"{minutes} minutes ago"
            else:
                time_text = "just now"
        else:
            time_text = "recently"

        weekday = article.get('weekday_of_click', 'Unknown')

        formatted_lines.append(f"    {i}. {section_text}{title}")
        formatted_lines.append(f"       Clicked {time_text} on {weekday}")

    formatted_lines.append("")

    return "\n".join(formatted_lines)


def _format_user_stats(stats):
    formatted_lines = []

    recent_articles_text = _format_recent_articles(stats.get('recent_articles', []))
    formatted_lines.extend(recent_articles_text.splitlines())

    top_sections = stats.get('top_sections', [])
    if top_sections:
        formatted_lines.append("")
        formatted_lines.append("    Most read sections:")
        for section in top_sections:
            count = section.get('count', 0)
            section_name = section.get('section', 'Unknown')
            formatted_lines.append(f"    - {section_name}: {count} articles")

    day_stats = stats.get('articles_by_day_of_week', [])
    if day_stats:
        formatted_lines.append("")
        formatted_lines.append("    Reading habits by day of week:")
        for day_stat in day_stats:
            day_name = day_stat.get('day_of_week', 'Unknown')
            count = day_stat.get('count', 0)
            formatted_lines.append(f"    - {day_name}: {count} articles")

    return "\n".join(formatted_lines)


def _user_prompt_prefix(username):
    data = mongo.db.users.find_one({ 'username' : username },
                                   { "engagement" : 1, "stats" : 1 })
    if data is None:
        return ""
    if not (engagement := data.get('engagement')):
        return ""
    if not (stats := data.get('stats')):
        return ""

    first_seen = 'unknown date and time'
    if 'first_seen' in engagement and engagement['first_seen']:
        try:
            first_seen_dt = datetime.fromisoformat(engagement['first_seen'])
            first_seen = first_seen_dt.strftime('%B %d %Y')
        except ValueError:
            pass

    active_days = engagement['active_days_28']
    inactive_days = engagement['gaps_count_28']

    prompt_prefix = f"""

    You are an agent to invent and conduct user acquisition and retention
    experiments for the news website called ist.media. Here's information about
    the current user with username { username }.

    This user has been first seen on { first_seen } and was active on { active_days } days
    in the last 28 days, and with { inactive_days } days of no activity.

    Their current engagement indexes: { engagement['smoothed_indexes'] }.
    These numbers are exponential moving averages and show the frequency of recent
    article consumption. The derived momentum is: { engagement['momentum'] }.
    A value larger than 1 indicates increasing reading activity, a value lower
    than 1 indicates decreasing activity.

    { _format_user_stats(stats) }

    """
    return prompt_prefix


def _ai_agent(username: str, task: str):
    if not (context := _user_prompt_prefix(username)):
        return ""
    response = open_ai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            { "role" : "user", "content" : f"{context}\n\nTask: {task}" }
        ],
        max_tokens = 1000,
        temperature = 0.5
    )
    return response.choices[0].message.content


def ai_agent_compute_user_summary(username):
    task = """Please provide a one-paragraph summary about the user.

    Mention their username. If the username looks like a hash key, it is an
    anonymous user, and you can expect that some of those users only show a
    reasonably short history of activity.

    Never show parantheses, and don't use technical terms,
    or variable names, or dict keys, to describe things, but rather explain
    for a non-technical user in business terms.
    """
    return _ai_agent(username, task)


def ai_agent_compute_user_aquisition_promo(username):
    task = """Please check if the user is a good candidate for a user
    acquisition promo (advertising them to register with the website)
    by checking their usage metrics. If you find a good and convincing
    argument for them to register, decide for one of these promotions:

    ID 1: acq_page: Provide a promotional text to register
    with the page, like "Hey, I see you have been reading ... and were active
    ... on our page, seems you like .... I tell you what: register with us
    and you will receive interesting benefits ..."

    ID 2: acq_join&win: Have the users become part of a lotterie to win an iPhone 17.

    ID 3: acq_discount: Provide the users 20% discount on the paid subscription,
    should the user register with the page.

    Use the history of articles and sections that have been consumed, besides
    other metrics that you find useful, to decide. Generate a one-paragraph promo
    text if you decide for ID 1.

    Please ONLY return a Python list, first element shall be a boolean value, e.g.
    True or False, indicating if this user is a candidate for promotion, second element
    shall be the type of the promo, as an integer 1, 2, or 3, the third
    element shall be your promotional text as a string, and the fourth element shall
    be your explanation about why you decided for ID 1, 2, or 3, respectively. If you
    decided for ID 1, explain why you did not decide for 2, or 3.
    Only return a list [bool, int, str, str] and nothing else.
    No ```python or ```. Just pure Python list.
    """
    return _ai_agent(username, task)
