from __future__ import annotations
from datetime import date
from typing import Dict, Any, Optional, List


def _compute_age_from_date_str(birthday_str: str) -> Optional[int]:
    """
    Try to compute age from a string date.
    Supports both '17/04/2004' (DD/MM/YYYY) and '04/17/2004' (MM/DD/YYYY).
    Returns None if parsing fails.
    """
    if not birthday_str:
        return None

    parts = birthday_str.split("/")
    if len(parts) != 3:
        return None

    def _safe_date(d: int, m: int, y: int) -> Optional[date]:
        try:
            return date(y, m, d)
        except Exception:
            return None

    try:
        p1, p2, p3 = [int(p) for p in parts]
    except Exception:
        return None

    # Try DD/MM/YYYY first
    candidates: List[date] = []
    d1 = _safe_date(p1, p2, p3)
    if d1:
        candidates.append(d1)

    # Also try MM/DD/YYYY if it looks plausible
    d2 = _safe_date(p2, p1, p3)
    if d2 and d2 != d1:
        candidates.append(d2)

    if not candidates:
        return None

    # Pick the earliest date (in case both are valid)
    b = min(candidates)
    today = date.today()
    age = today.year - b.year - (
        (today.month, today.day) < (b.month, b.day)
    )
    return age


# ================= MAIN DATABASE ===================

IDENTITY_DB: Dict[str, Dict[str, Any]] = {
    "PTri's Muse": {
        "label": "Hạ Thi",
        "nickname": "Summer Poem",
        "instagram": "@thomuaha",

        "relationship_to_ptri": "very special person!",
        "gender": "girl",
        "birthday": "May 2nd, 2004",

        "careers": ["Marketer"],
        "university": ["University of Economy Ho Chi Minh City (UEH)"],
        "hobbies": ["content creation", "social media", "fashion", "Black Pink"],

        "favorite_music": ["pop", "K-pop"],
        "favorite_games": [],
        "favorite_movies": ["romantic series", "Korean dramas"],
        "achievements": [
            "Queen of Thang Long High School",
            "Staff in Hakuhodo Vietnam",
            "Content creater in ELLE Vietnam",
        ],

        "how_you_met": "Met at Loc Minh IELTS Center and gradually became very special to PTri.",
        "special_memories": [
            "Considered the first serious date in PTri's life.",
        ],

        "priority_level": 10,

        "base_text": (
            "PTri's Muse: a very special person to PTri, girl, marketer, "
            "born 05/02/2004, Queen of Thang Long High School."
        ),
        # You can fill these with longer free-text facts if you want
        "personality": "out-going",
        "fun_facts": ["#2 Literature in Lam Dong Province", "lost connection with PTri"],
        "relationship_notes": "strangers now",

        "aliases": [
            "Nàng thơ",
            "nàng thơ",
            "nàng thơ của ptri",
            "Nàng Thơ của PTri",
            "Summer Poem",
            "Thơ Mùa Hạ",
        ],
    },

    "PTri": {
        "label": "PTri",
        "nickname": "Tri",
        "instagram": "@pt_1704",

        "relationship_to_ptri": "self / owner (IreneAdler's boss)",
        "gender": "man",
        "birthday": "April 17th, 2004",

        "careers": ["AI Engineer"],
        "university": "University of London",
        "hobbies": ["chess", "coding", "reading", "gaming"],

        "favorite_music": ["VPop", "rap", "US-UK", "K-Pop"],
        "favorite_games": ["chess", "Yu-Gi-Oh! Master Duel", "League of Legends"],
        "favorite_movies": ["Perfect Blue"],
        "achievements": [
            "Chess Master",
            "Top Rating Duelist",
            "#1 chess server 12a12",
            "TriGPT owner",
            "Canada international student",
            "IELTS 7.0 (Reading: 7.0, Listening: 7.5, Writing: 7.0, Speaking: 6.5)",
            "IBM AI Developer, Engineer",
            "Software Engineer at ImageSource Inc., Washington, U.S"
        ],

        "how_you_met": "N/A (this is PTri himself, owner of the system).",
        "special_memories": [
            "First time building this local AI lab together.",
        ],

        "priority_level": 10,

        "base_text": (
            "PTri: AI Engineer, man, Chess Master, born 04/17/2004, "
            "owner of this system and IreneAdler's boss."
        ),

        "personality": "introvert, overthinking, creative, progressive",
        "fun_facts": [],
        "relationship_notes": "",
    },

    "Lanh": {
        "label": "Lanh",
        "nickname": "Lanh",
        "instagram": "@_.lannanhh._",

        "relationship_to_ptri": "friend",
        "gender": "girl",
        "birthday": "May 1st,2004",

        "careers": ["Accountant"],
        "university": ["Open University of Ho Chi Minh City"],
        "hobbies": ["hanging out with friends", "music"],

        "favorite_music": ["V-pop", "romantic ballads"],
        "favorite_games": ["Valorant"],
        "favorite_movies": ["Vietnamese family movies"],
        "achievements": ["Accountant of Beers Company - Minh Hung"],

        "how_you_met": (
            "Met through high school in 2019 and became friends. "
            "Also classmates with Xuân Việt, Khánh Nguyên, MTuan, Bình, PTrinh and PTri."
        ),
        "special_memories": [
            "Once asked PTri for 4G and then gave candy as compensation, turning into a shared joke.",
        ],

        "priority_level": 7,

        "base_text": "Lanh: PTri's friend, girl, cashier, born 01/05/2004.",

        "personality": "funny, sensitive",
        "fun_facts": ["has many boys/men flirt and chase"],
        "relationship_notes": "close friend",
    },

    "MTuan": {
        "label": "MTuan",
        "nickname": "Tuấn Dế",
        "instagram": "@mih.naut",

        "relationship_to_ptri": "friend",
        "gender": "man",
        "birthday": "December 17th, 2004",

        "careers": ["Business Analyst"],
        "university": ["FPT University"],
        "hobbies": ["playing football", "video games"],

        "favorite_music": ["badtrip"],
        "favorite_games": ["ARAM in League of Legends"],
        "favorite_movies": [],
        "achievements": [],

        "how_you_met": (
            "Met through high school in 2019 and bonded over shared interests. "
            "Also classmates with Xuân Việt, Khánh Nguyên, Bình, Lanh, PTrinh and PTri."
        ),
        "special_memories": [
            "Often discussed startup ideas together and mixed them with game talk.",
        ],

        "priority_level": 8,

        "base_text": (
            "MTuan: PTri's friend, man, Business Analyst direction, born 17/12/2004."
        ),

        "personality": "childish",
        "fun_facts": ["good at thinking in chess", "bad at AD Carry role in LOL"],
        "relationship_notes": "work mate",
    },

    "BHa": {
        "label": "BHa",
        "nickname": "Bích Hà",
        "instagram": "@_im.bichha",

        "relationship_to_ptri": "friend",
        "gender": "girl",
        "birthday": "February 4th, 2004",

        "careers": ["Tour Guider"],
        "university": ["Da Lat University"],
        "hobbies": ["traveling", "going to coffee shop", "IELTS testing"],

        "favorite_music": [],
        "favorite_games": [],
        "favorite_movies": [],
        "achievements": ["intern for a long vacation tour"],

        "how_you_met": "Met during school in 2018 and stayed in touch afterwards.",
        "special_memories": [
            "At one time, Bích Hà was an old crush in PTri's memory.",
        ],

        "priority_level": 8,

        "base_text": (
            "BHa: PTri's friend, girl, tour guider, born 04/02/2004."
        ),

        "personality": "funny, out-going, diligent",
        "fun_facts": ["used to hate PTri so bad"],
        "relationship_notes": "close friend",
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

        "personality": "",
        "fun_facts": [],
        "relationship_notes": "",
    },

    "Bình": {
        "label": "BinhLe",
        "nickname": "why u scammer",
        "instagram": "@binhle_252",

        "relationship_to_ptri": "friend",
        "gender": "man",
        "birthday": "February 25th, 2004",

        "careers": ["Cyber Security"],
        "university": ["Posts and Telecommunications Institute of Technology"],
        "hobbies": ["badminton", "gym", "URF"],

        "favorite_music": ["remix", "US-UK"],
        "favorite_games": [
            "URF in League of Legends",
            "Elden Ring",
            "Sekiro",
            "Black Myth Wukong",
        ],
        "favorite_movies": [],
        "achievements": ["#1 server badminton Lâm Đồng"],

        "how_you_met": (
            "Met through high school in 2019 and kept in touch, often talking in a playful, phonily way until 2023. "
            "Also classmates with MTuan, Xuân Việt, Khánh Nguyên, Lanh, PTrinh and PTri."
        ),
        "special_memories": [
            "Binh once 'scammed' PTri an ice cream at the airport when PTri was about to go to Canada, which became a shared joke.",
        ],

        "priority_level": 8,

        "base_text": (
            "BinhLe: PTri's friend and classmate, man, cyber security direction, "
            "born 25/02/2004, usually speaks English with PTri."
        ),

        "personality": "introvert, free-thinking",
        "fun_facts": ["love badminton than even his own cyber security major"],
        "relationship_notes": "English friend",
    },

    "Hoài Thương": {
        "label": "HThuong",
        "nickname": "na hoài",
        "instagram": "@hoaithuongg._",

        "relationship_to_ptri": "friend",
        "gender": "girl",
        "birthday": "July 17th, 2004",

        "careers": ["International Business"],
        "university": ["Ho Chi Minh City University of Economics and Finance (UEF)"],
        "hobbies": ["coffee", "gym", "study date"],

        "favorite_music": ["V-Pop"],
        "favorite_games": [],
        "favorite_movies": ["K-drama", "Chinese romance drama"],
        "achievements": ["being a model for a fashion studio"],

        "how_you_met": (
            "Was a classmate in the extra class of Mrs Hà in 2016, then a classmate again in grade 9 in 2018. "
            "Stayed in touch afterwards as a deep-talking friend and emotional support for PTri, especially when he felt isolated in Canada."
        ),
        "special_memories": [
            "Hoai Thuong asked for a farewell meetup before PTri went to Canada and was also the first person picking him up when he returned to Vietnam.",
        ],

        "priority_level": 9,

        "base_text": (
            "HThuong: old classmate and deep-talking mate of PTri, girl, International Business, model, "
            "born 17/07/2004, studying at UEF."
        ),

        "personality": "out-going, funny, caring",
        "fun_facts": ["used to be a rice paper seller in grade 10, 11"],
        "relationship_notes": "deep-talking mate",
    },

    "Xuân Việt": {
        "label": "XViet",
        "nickname": "Canyon 2k4",
        "instagram": "@xviett_2601",

        "relationship_to_ptri": "friend",
        "gender": "man",
        "birthday": "January 26th, 2004",

        "careers": ["Local Business (Soy Milk)"],
        "university": [""],
        "hobbies": ["coffee", "cloud hunt", "drinking", "video games", "chess"],

        "favorite_music": ["V-Pop"],
        "favorite_games": ["League of Legends: Wild Rift"],
        "favorite_movies": [],
        "achievements": ["the wealthiest guy in 12a12"],

        "how_you_met": (
            "Met in 2016 in Nguyen Du Secondary School's Rubik community. "
            "Later became high school classmates in 2019 and deskmates with PTri. "
            "Also classmates with Khánh Nguyên, MTuan, BinhLe, Lanh, PTrinh and PTri. "
            "Was also PTri's main chess rival in high school."
        ),
        "special_memories": [
            "Once played a chess match with PTri that continued even after their class had ended for about half an hour.",
        ],

        "priority_level": 9,

        "base_text": (
            "XViet: PTri's classmate, chessmate and deskmate, man, runs a well-known soy milk business in Da Lat City (Dung Béo), "
            "born 26/01/2004, known as the wealthiest guy in 12a12."
        ),

        "personality": "talkative, funny, out-going",
        "fun_facts": ["The only one once beating PTri in a chess game"],
        "relationship_notes": "close friend",
    },

    "Khánh Nguyên": {
        "label": "KNguyen",
        "nickname": "Sakura Đao",
        "instagram": "@kah.nyge",

        "relationship_to_ptri": "friend",
        "gender": "girl",
        "birthday": "October 9th, 2004",

        "careers": ["Bartender"],
        "university": ["Da Lat University"],
        "hobbies": ["coffee", "cloud hunt", "drinking"],

        "favorite_music": ["V-Pop"],
        "favorite_games": ["Arena of Valor"],
        "favorite_movies": [],
        "achievements": ["one of the first startup-minded students in 12a12"],

        "how_you_met": (
            "High school classmates in 2019, then deskmates in 2020 and 2021. "
            "Also classmates with Xuân Việt, BinhLe, MTuan, Lanh, PTrinh and PTri."
        ),
        "special_memories": [
            "Frequently asked PTri for tissues so often that almost all of his tissues were used by her. "
            "She was also the only classmate who called him while he was living alone in Canada.",
        ],

        "priority_level": 8,

        "base_text": (
            "KNguyen: PTri's classmate and deskmate, girl, bartender, "
            "born 09/10/2004, has a strong startup mindset."
        ),

        "personality": "out-going, funny, talkative",
        "fun_facts": ["used to hate PTri at first"],
        "relationship_notes": "old deskmate",
    },
    "PTrinh": {
        "label": "Phương Trinh",
        "nickname": "Chin",
        "instagram": "@chinuowo",

        "relationship_to_ptri": "friend",
        "gender": "girl",
        "birthday": "August 9th, 2004",

        "careers": ["Investor"],
        "university": ["Ho Chi Minh City University of Economics and Finance - UEF"],
        "hobbies": ["coffee", "badminton", "hang out"],

        "favorite_music": ["badtrip"],
        "favorite_games": [],
        "favorite_movies": [],
        "achievements": ["internship in trading company"],

        "how_you_met": (
            "Met through high school in 2019. 7 years of exam-mates in secondary school and high school."
            "Also classmates with Xuân Việt, Khánh Nguyên, Bình, Lanh, MTuan and PTri."
        ),
        "special_memories": [
            "PTri's love strategist in high school. PTrinh cried when PTri first went to Canada for studying abroad.",
        ],

        "priority_level": 8,

        "base_text": (
            "PTrinh: PTri's friend, girl, investor, trader, born 09/08/2004."
        ),

        "personality": "mature, cool",
        "fun_facts": ["PTri is known by PTrinh's parents and in reverse too."],
        "relationship_notes": "deep-talking mate",
    },
}


# ================= HELPER FUNCTIONS ===================

def _find_record_by_label_or_key(label_or_key: str) -> Optional[Dict[str, Any]]:
    """
    Find a profile either by direct key in IDENTITY_DB or by its 'label' field.
    """
    if label_or_key in IDENTITY_DB:
        return IDENTITY_DB[label_or_key]

    # Search by 'label'
    for key, info in IDENTITY_DB.items():
        if info.get("label") == label_or_key:
            return info

    return None


def get_identity_summary(identity_key: str) -> Optional[str]:
    """
    Return a short, one-line English summary for the given identity key.
    This is used in /live_detect as 'identity_info'.

    It must NOT speak as Irene (no 'I', 'me', 'my'); just describe the person
    relative to PTri. Priority level is kept internal and not verbalised.
    """
    info = IDENTITY_DB.get(identity_key)
    if not info:
        return None

    name = identity_key
    label = info.get("label") or name
    rel = info.get("relationship_to_ptri") or "connection"
    careers = info.get("careers") or []
    university = info.get("university")

    # Normalize university into string
    if isinstance(university, list):
        uni_str = ", ".join(u for u in university if u)
    else:
        uni_str = university or ""

    career_str = ", ".join(careers) if careers else ""
    pieces = [f"{label}: {rel} of PTri"]

    if career_str:
        pieces.append(f"career: {career_str}")
    if uni_str:
        pieces.append(f"university: {uni_str}")

    return " | ".join(pieces)


def render_profile_details(identity_label: str) -> str:
    """
    Render a multi-line factual description for an identity, used as context
    for Irene (not directly shown to the user).

    Rules:
    - No 'I', 'me', 'my'; always refer to PTri or 'this person'.
    - English only (Vietnamese style is handled at the LLM layer).
    - Include only facts from the database; do not invent new info.
    """
    info = _find_record_by_label_or_key(identity_label)
    if not info:
        return f"No stored profile for label '{identity_label}'."

    key_name = None
    for k, v in IDENTITY_DB.items():
        if v is info:
            key_name = k
            break

    display_name = key_name or identity_label
    label = info.get("label") or display_name
    nickname = info.get("nickname")
    rel = info.get("relationship_to_ptri") or "connection"
    instagram = info.get("instagram")

    birthday = info.get("birthday")
    age = _compute_age_from_date_str(birthday) if birthday else None

    careers = info.get("careers") or []
    university = info.get("university")
    hobbies = info.get("hobbies") or []
    favorite_music = info.get("favorite_music") or []
    favorite_games = info.get("favorite_games") or []
    favorite_movies = info.get("favorite_movies") or []
    achievements = info.get("achievements") or []
    how_you_met = info.get("how_you_met")
    special_memories = info.get("special_memories") or []
    personality = info.get("personality")
    fun_facts = info.get("fun_facts") or []
    relationship_notes = info.get("relationship_notes")
    gender = info.get("gender")  # ⬅️ for pronoun rules
    # priority = info.get("priority_level", 5)  # internal only

    lines: List[str] = []
    lines.append(f"Name key: {display_name}")
    lines.append(f"Profile label: {label}")
    lines.append(f"Relationship to PTri: {rel}")

    # Explicit gender + pronoun guidance so Irene stops mixing 'anh ấy' / 'cô ấy'
    if gender == "man":
        lines.append(
            "Gender: male. In Vietnamese, ALWAYS use male pronouns for this person "
            "(for example: 'anh ấy', 'cậu ấy') and NEVER use female ones like "
            "'cô ấy', 'chị ấy'."
        )
    elif gender == "girl":
        lines.append(
            "Gender: female. In Vietnamese, ALWAYS use female pronouns for this person "
            "(for example: 'cô ấy', 'chị ấy') and NEVER use male ones like "
            "'anh ấy', 'cậu ấy'."
        )

    if nickname:
        lines.append(f"Nickname: {nickname}")
    if instagram:
        lines.append(f"Instagram: {instagram}")
    if birthday:
        if age is not None:
            lines.append(f"Birthday: {birthday} (approx. age {age})")
        else:
            lines.append(f"Birthday: {birthday}")
    if careers:
        lines.append(f"Careers: {', '.join(careers)}")
    if university:
        if isinstance(university, list):
            lines.append(f"University: {', '.join(u for u in university if u)}")
        else:
            lines.append(f"University: {university}")
    if hobbies:
        lines.append(f"Hobbies: {', '.join(hobbies)}")
    if favorite_music:
        lines.append(f"Favourite music: {', '.join(favorite_music)}")
    if favorite_games:
        lines.append(f"Favourite games: {', '.join(favorite_games)}")
    if favorite_movies:
        lines.append(f"Favourite movies: {', '.join(favorite_movies)}")
    if achievements:
        lines.append(f"Achievements: {', '.join(achievements)}")
    if how_you_met:
        lines.append(f"How PTri met this person: {how_you_met}")
    if special_memories:
        lines.append("Special memories with PTri:")
        for m in special_memories:
            lines.append(f"- {m}")

    if personality:
        lines.append(f"Personality description: {personality}")
    if fun_facts:
        lines.append("Fun facts / small details:")
        for f in fun_facts:
            lines.append(f"- {f}")
    if relationship_notes:
        lines.append(f"Relationship notes: {relationship_notes}")

    # Priority level is kept internal and not verbalised.

    return "\n".join(lines)
