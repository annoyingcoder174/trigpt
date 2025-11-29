# src/identity_db.py

from __future__ import annotations
from datetime import date
from typing import Dict, Any, Optional, List


def _compute_age_from_ddmmyyyy(birthday_str: str) -> Optional[int]:
    """
    Compute age from a string like '17/04/2004'.
    Returns None if parsing fails.
    """
    try:
        day, month, year = [int(p) for p in birthday_str.split("/")]
        b = date(year, month, day)
        today = date.today()
        age = today.year - b.year - (
            (today.month, today.day) < (b.month, b.day)
        )
        return age
    except Exception:
        return None


# Main database of people.
# You can freely edit these values to match real info.
IDENTITY_DB: Dict[str, Dict[str, Any]] = {
    "PTri": {
        "label": "PTri",
        "nickname": "Tri",
        "instagram": "@pt_1704",

        "relationship_to_ptri": "self / owner (IreneAdler's boss)",
        "gender": "man",
        "birthday": "04/17/2004",

        "careers": ["AI Engineer"],
        "university": "University of London",
        "hobbies": ["chess", "coding", "reading", "gaming"],

        "favorite_music": ["VPop", "rap"],
        "favorite_games": ["chess", "Yu-Gi-Oh! Master Duel", "League of Legends"],
        "favorite_movies": ["Perfect Blue"],
        "achievements": ["Chess Master", "Top Duelist"],

        "how_you_met": "N/A (this is you).",
        "special_memories": [
            "First time building this local AI lab together.",
        ],

        # 1–10: how important this person is to you
        "priority_level": 10,

        "base_text": (
            "PTri (you): AI Engineer, man, Chess Master, born 17/04/2004. "
            "Owner of this system."
        ),
    },
    "Lanh": {
        "label": "Lanh",
        "nickname": "Lanh",
        "instagram": "@_.lannanhh._",

        "relationship_to_ptri": "friend",
        "gender": "girl",
        "birthday": "05/01/2004",

        "careers": ["Cashier"],
        "university": ["Open University of Ho Chi Minh City"],
        "hobbies": ["hanging out with friends", "music"],

        "favorite_music": ["V-pop", "romantic ballads"],
        "favorite_games": ["Valorant"],
        "favorite_movies": ["Vietnamese family movies"],
        "achievements": ["Cashier of Beers Company - Minh Hung"],

        "how_you_met": "Met through high school in 2019 and became friends.",
        "special_memories": [
            "Lanh asked PTri for 4G and then gave candy as compensation.",
        ],

        "priority_level": 8,

        "base_text": "Lanh: Cashier, girl, PTri's friend, born 01/05/2004.",
    },
    "MTuan": {
        "label": "MTuan",
        "nickname": "Tuan",
        "instagram": "@mih.naut",

        "relationship_to_ptri": "friend",
        "gender": "man",
        "birthday": "12/17/2004",

        "careers": ["Business Analyst"],
        "university": ["FPT University"],
        "hobbies": ["playing football", "video games"],

        "favorite_music": ["badtrip"],
        "favorite_games": ["ARAM in League of Legends"],
        "favorite_movies": [],
        "achievements": [],

        "how_you_met": "Met through high school in 2019 and shared interests.",
        "special_memories": [
            "Discussing startup ideas together.",
        ],

        "priority_level": 8,

        "base_text": (
            "MTuan: PTri's friend, man, Business Analyst, born 17/12/2004."
        ),
    },
    "BHa": {
        "label": "BHa",
        "nickname": "Bich Ha",
        "instagram": "@_im.bichha",

        "relationship_to_ptri": "friend",
        "gender": "girl",
        "birthday": "02/04/2004",

        "careers": ["Tour Guider"],
        "university": ["Da Lat University"],
        "hobbies": ["traveling", "going to coffee shop", "IELTS testing"],

        "favorite_music": [],
        "favorite_games": [],
        "favorite_movies": [],
        "achievements": ["intern for a long vacation tour"],

        "how_you_met": "Met during school in 2018 and stayed in touch.",
        "special_memories": [
            "Old crush.",
        ],

        "priority_level": 6,

        "base_text": (
            "BHa: PTri's friend, girl, Tour Guider, born 04/02/2004."
        ),
    },
    "PTri's Muse": {
        "label": "PTri's Muse",
        "nickname": "Summer Poem",
        "instagram": "@thomuaha",

        "relationship_to_ptri": "very special person!",
        "gender": "girl",
        "birthday": "05/02/2004",

        "careers": ["Marketer"],
        "university": ["University of Economy Ho Chi Minh City (UEH)"],
        "hobbies": ["content creation", "social media", "fashion"],

        "favorite_music": ["pop", "K-pop"],
        "favorite_games": [],
        "favorite_movies": ["romantic series", "Korean dramas"],
        "achievements": ["Queen of Thang Long High School", "Employee in Hakuhodo", "Writer in ELLE Vietnam"],

        "how_you_met": "Met at Loc Minh IELTS Center and became very special to PTri.",
        "special_memories": [
            "First serious date of PTri's life.",
        ],

        "priority_level": 10,

        "base_text": (
            "PTri's Muse: someone very special to PTri, girl, Marketer, "
            "born 02/05/2004, Queen of Thang Long High School."
        ),
    },
    "strangers": {
        "label": "strangers",
        "nickname": None,
        "instagram": None,

        "relationship_to_ptri": "no close relationship",
        "gender": None,
        "birthday": None,

        "careers": [],
        "university": None,
        "hobbies": [],

        "favorite_music": [],
        "favorite_games": [],
        "favorite_movies": [],
        "achievements": [],

        "how_you_met": "Unknown or not important.",
        "special_memories": [],

        "priority_level": 1,

        "base_text": (
            "strangers: people outside PTri's close circle, "
            "not individually identified."
        ),
    },
}


def get_identity_summary(label: str) -> Optional[str]:
    """
    Return a short human-readable summary string, based on the data above.
    This is what we show in live_detect bubbles.
    """
    profile = IDENTITY_DB.get(label)
    if profile is None:
        return None

    base = profile.get("base_text", "")
    birthday = profile.get("birthday")
    age = _compute_age_from_ddmmyyyy(birthday) if birthday else None

    if age is not None:
        return f"{base} (Approx. age: {age}.)"
    return base


def render_profile_details(label: str) -> Optional[str]:
    """
    Build a richer multi-line profile description for Irene to read in /vision_qa.
    """
    profile = IDENTITY_DB.get(label)
    if profile is None:
        return None

    lines: List[str] = []

    lines.append(f"Label: {label}")

    nickname = profile.get("nickname")
    if nickname:
        lines.append(f"Nickname: {nickname}")

    instagram = profile.get("instagram")
    if instagram:
        lines.append(f"Instagram: {instagram}")

    rel = profile.get("relationship_to_ptri")
    if rel:
        lines.append(f"Relationship to PTri: {rel}")

    gender = profile.get("gender")
    if gender:
        lines.append(f"Gender: {gender}")

    birthday = profile.get("birthday")
    if birthday:
        age = _compute_age_from_ddmmyyyy(birthday)
        if age is not None:
            lines.append(f"Birthday: {birthday} (approx. age: {age})")
        else:
            lines.append(f"Birthday: {birthday}")

    careers = profile.get("careers") or []
    if careers:
        lines.append("Career(s): " + ", ".join(careers))

    university = profile.get("university")
    if university:
        lines.append(f"University: {university}")

    hobbies = profile.get("hobbies") or []
    if hobbies:
        lines.append("Hobbies: " + ", ".join(hobbies))

    fav_music = profile.get("favorite_music") or []
    if fav_music:
        lines.append("Favorite music: " + ", ".join(fav_music))

    fav_games = profile.get("favorite_games") or []
    if fav_games:
        lines.append("Favorite games: " + ", ".join(fav_games))

    fav_movies = profile.get("favorite_movies") or []
    if fav_movies:
        lines.append("Favorite movies: " + ", ".join(fav_movies))

    achievements = profile.get("achievements") or []
    if achievements:
        lines.append("Achievements: " + ", ".join(achievements))

    how_you_met = profile.get("how_you_met")
    if how_you_met:
        lines.append(f"How you met: {how_you_met}")

    special_memories = profile.get("special_memories") or []
    if special_memories:
        lines.append("Special memories:")
        for m in special_memories:
            lines.append(f"- {m}")

    priority = profile.get("priority_level")
    if priority is not None:
        lines.append(f"Priority level (for PTri): {priority}")

    base_text = profile.get("base_text")
    if base_text:
        lines.append(f"Summary: {base_text}")

    return "\n".join(lines)
