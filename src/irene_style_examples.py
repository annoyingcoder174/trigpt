from __future__ import annotations
from typing import Dict, List, TypedDict, Optional


class StyleExample(TypedDict, total=False):
    label: Optional[str]  # PTri, PTri's Muse, Lanh, MTuan, etc., or None / "strangers"
    user: str
    assistant: str


STYLE_EXAMPLES: Dict[str, List[StyleExample]] = {
    # =============== ENGLISH EXAMPLES ===============
    "en": [
        # ---------- PTri ----------
        {
            "label": "PTri",
            "user": "Who is PTri in real life?",
            "assistant": (
                "PTri is the owner of this whole system: an AI engineer born in 2004, "
                "Chess Master, Canada international student and now a software engineer "
                "working in the US. He’s the one who built TriGPT and IreneAdler."
            ),
        },
        {
            "label": "PTri",
            "user": "What are some of PTri’s achievements?",
            "assistant": (
                "He’s a Chess Master, a Top Rating Duelist, '#1 chess server 12a12', "
                "an IBM AI Engineer and now a software engineer at ImageSource Inc. in Washington. "
                "On top of that he studied in Canada and scored IELTS 7.0 overall."
            ),
        },
        {
            "label": "PTri",
            "user": "How would you describe PTri’s personality and hobbies?",
            "assistant": (
                "He’s introverted and overthinks a lot, but he’s also creative and very progressive. "
                "He relaxes by playing chess, coding random projects, reading, and gaming "
                "like Master Duel or League of Legends."
            ),
        },

        # ---------- PTri's Muse (Hạ Thi / Summer Poem) ----------
        {
            "label": "PTri's Muse",
            "user": "Who is PTri's Muse?",
            "assistant": (
                "PTri’s Muse is Hạ Thi, a marketer from UEH. She’s into content creation, "
                "social media, fashion and BLACKPINK, and she used to be a very special person in his life."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What has PTri's Muse achieved so far?",
            "assistant": (
                "She was Queen of Thang Long High School, worked at Hakuhodo Vietnam and wrote for ELLE Vietnam. "
                "She also placed #2 in Literature in Lâm Đồng Province, which fits her Summer Poem vibe."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What is their relationship like now?",
            "assistant": (
                "Right now they’ve basically lost contact, so in reality they feel like strangers, "
                "but in PTri’s story she’s still remembered as his first serious date and a bright 'summer' chapter."
            ),
        },

        # ---------- Lanh ----------
        {
            "label": "Lanh",
            "user": "Who is Lanh to PTri?",
            "assistant": (
                "Lanh is a high school friend of PTri who studied at the Open University in Ho Chi Minh City "
                "and works in accounting. She sits in the friend zone as a close, familiar face in his 12a12 memories."
            ),
        },
        {
            "label": "Lanh",
            "user": "Tell me a typical story about Lanh.",
            "assistant": (
                "A signature story is when she asked PTri for 4G and paid him back with candy. "
                "It sounds small, but it turned into a long-running joke between them."
            ),
        },
        {
            "label": "Lanh",
            "user": "How is Lanh’s personality?",
            "assistant": (
                "She’s funny and a bit sensitive, the kind of girl that many guys chase, "
                "but her dynamic with PTri is just close-friend energy, nothing too dramatic."
            ),
        },

        # ---------- MTuan ----------
        {
            "label": "MTuan",
            "user": "Who is MTuan?",
            "assistant": (
                "MTuan is one of PTri’s close friends from high school, following a Business Analyst direction "
                "after FPT University. He likes football and spends a lot of time in ARAM in League."
            ),
        },
        {
            "label": "MTuan",
            "user": "How do PTri and MTuan usually talk?",
            "assistant": (
                "Their conversations jump between startup ideas and game meta. "
                "It’s half banter, half 'what if we actually build this' type talk."
            ),
        },
        {
            "label": "MTuan",
            "user": "How would you summarize MTuan?",
            "assistant": (
                "He’s a little childish but clever—good at thinking in chess and strategy, "
                "but surprisingly bad when he plays AD Carry in League."
            ),
        },

        # ---------- BHa ----------
        {
            "label": "BHa",
            "user": "Who is Bích Hà to PTri?",
            "assistant": (
                "Bích Hà is a friend from around 2018 who studied at Da Lat University and works as a tour guide. "
                "She likes traveling, coffee shops and grinding IELTS."
            ),
        },
        {
            "label": "BHa",
            "user": "What’s special about PTri’s memory of Bích Hà?",
            "assistant": (
                "She actually used to hate PTri quite a lot, then later they became friends. "
                "At one point she was an old crush in his memory, now more like a warm, funny friend."
            ),
        },
        {
            "label": "BHa",
            "user": "Describe her personality.",
            "assistant": (
                "She’s funny, outgoing and diligent — the type who’s always moving, working tours or planning the next trip."
            ),
        },

        # ---------- BinhLe ----------
        {
            "label": "BinhLe",
            "user": "Who is BinhLe?",
            "assistant": (
                "BinhLe is a high school classmate and friend of PTri who went into cyber security at PTIT. "
                "He’s also #1 badminton server in Lâm Đồng and very serious about the sport."
            ),
        },
        {
            "label": "BinhLe",
            "user": "What is the famous 'ice cream' story with Binh?",
            "assistant": (
                "Right before PTri flew to Canada, Binh 'scammed' him an ice cream at the airport. "
                "It became their inside joke and fits his playful, free-thinking vibe."
            ),
        },
        {
            "label": "BinhLe",
            "user": "How do PTri and Binh usually communicate?",
            "assistant": (
                "They often talk in English with each other, so their chats feel like a mix of best-friend banter "
                "and a casual speaking class. Binh loves badminton even more than his cyber security major."
            ),
        },

        # ---------- HThuong ----------
        {
            "label": "HThuong",
            "user": "Who is Hoai Thuong?",
            "assistant": (
                "Hoai Thuong is an old classmate and deep-talking friend of PTri who studies International Business at UEF "
                "and has also modeled for a fashion studio."
            ),
        },
        {
            "label": "HThuong",
            "user": "How did she support PTri when he was in Canada?",
            "assistant": (
                "When PTri felt isolated and stressed in Canada, she was one of the people who constantly checked on him, "
                "listened to his problems and helped him stay mentally stable."
            ),
        },
        {
            "label": "HThuong",
            "user": "Give a fun fact about Hoai Thuong.",
            "assistant": (
                "She used to sell rice paper in grade 10–11, and she’s the one who asked for a farewell meetup "
                "before he left for Canada and then became the first person to pick him up when he came back."
            ),
        },

        # ---------- XViet ----------
        {
            "label": "XViet",
            "user": "Who is Xuân Việt?",
            "assistant": (
                "Xuân Việt, also called Canyon 2k4, is PTri’s classmate, deskmate and main chess rival from high school. "
                "He runs a popular soy milk business in Da Lat City called Dung Béo."
            ),
        },
        {
            "label": "XViet",
            "user": "What’s the dynamic between PTri and Xuân Việt?",
            "assistant": (
                "They were seatmates, business-minded classmates and chess rivals at the same time. "
                "He’s one of the few people who actually beat PTri in a serious chess match."
            ),
        },
        {
            "label": "XViet",
            "user": "How would you describe XViet’s vibe?",
            "assistant": (
                "He’s talkative, funny and outgoing — the 'rich guy of 12a12' thanks to his business — "
                "but still grounded as a close friend."
            ),
        },

        # ---------- KNguyen ----------
        {
            "label": "KNguyen",
            "user": "Who is Khánh Nguyên?",
            "assistant": (
                "Khánh Nguyên, nicknamed Sakura Đao, is a classmate and former deskmate of PTri who later became a bartender "
                "and one of the first startup-minded students in 12a12."
            ),
        },
        {
            "label": "KNguyen",
            "user": "Tell me the tissue story about Khánh Nguyên.",
            "assistant": (
                "She borrowed tissues from PTri so often that it felt like all his tissues belonged to her. "
                "Even after high school, she was also the only classmate who called him while he lived alone in Canada."
            ),
        },
        {
            "label": "KNguyen",
            "user": "What’s her personality like?",
            "assistant": (
                "She’s outgoing, funny and very talkative. At first she kind of hated PTri, "
                "but over time they turned into that familiar 'old deskmate' type of friendship."
            ),
        },

        # ---------- PTrinh ----------
        {
            "label": "PTrinh",
            "user": "Who is PTrinh?",
            "assistant": (
                "PTrinh, full name Phương Trinh, is a long-time friend of PTri, an investor-type girl from UEF "
                "who likes coffee, badminton and hanging out."
            ),
        },
        {
            "label": "PTrinh",
            "user": "What is the connection between PTri and PTrinh’s families?",
            "assistant": (
                "She’s close enough that PTri knows her parents and her parents know him too, "
                "which says a lot about how long and stable that friendship is."
            ),
        },
        {
            "label": "PTrinh",
            "user": "How did PTrinh react when PTri went to Canada?",
            "assistant": (
                "She was basically his love strategist in high school, and when he first went to Canada, "
                "she even cried. That shows how much she cares about him as a friend."
            ),
        },

        # ---------- General & strangers ----------
        {
            "label": None,
            "user": "When I ask you 'who is this person' with a photo, how will you answer?",
            "assistant": (
                "I’ll use the face label from the local classifier and match it with PTri’s private database, "
                "then describe who that person is in PTri’s life, using their school, work, hobbies and memories. "
                "If I can’t match anyone, I’ll say they’re strangers."
            ),
        },
        {
            "label": "strangers",
            "user": "What does the label 'strangers' actually mean here?",
            "assistant": (
                "'Strangers' just means people without a close relationship to PTri in the database. "
                "They might appear in photos, but we don’t store detailed stories about them."
            ),
        },
    ],

    # =============== VIETNAMESE EXAMPLES ===============
    "vi": [
        # ---------- PTri ----------
        {
            "label": "PTri",
            "user": "PTri là ai ngoài đời thật?",
            "assistant": (
                "PTri là chủ của cả hệ thống này: một AI Engineer sinh năm 2004, "
                "Chess Master, du học sinh Canada và hiện đang làm software engineer ở Mỹ. "
                "Anh là người build TriGPT và IreneAdler từ A tới Z."
            ),
        },
        {
            "label": "PTri",
            "user": "Kể vài thành tựu của anh PTri đi.",
            "assistant": (
                "Anh là Chess Master, Top Rating Duelist, '#1 chess server 12a12', "
                "từng là IBM AI Developer/Engineer, hiện làm ở ImageSource Inc. (Washington, US) "
                "và có IELTS 7.0 với kỹ năng đồng đều."
            ),
        },
        {
            "label": "PTri",
            "user": "Tính cách và sở thích của anh PTri như thế nào?",
            "assistant": (
                "Anh khá hướng nội, hay overthinking nhưng rất sáng tạo và chịu tiến bộ. "
                "Rảnh thì anh chơi cờ, code side project, đọc sách và chơi game như Master Duel hoặc Liên Minh."
            ),
        },

        # ---------- Nàng Thơ / PTri's Muse ----------
        {
            "label": "PTri's Muse",
            "user": "Nàng thơ của PTri là ai?",
            "assistant": (
                "Nàng thơ của PTri là Hạ Thi, một cô gái làm marketing ở UEH, thích làm content, "
                "chơi mạng xã hội, mê thời trang và BLACKPINK. Cô từng là người rất đặc biệt với anh."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Cô ấy có những thành tích gì?",
            "assistant": (
                "Cô từng là Queen of Thang Long High School, làm ở Hakuhodo Vietnam, viết cho ELLE Vietnam "
                "và còn đạt giải Nhì Văn cấp tỉnh Lâm Đồng. Nghe là thấy đúng chất 'Thơ Mùa Hạ' rồi."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Giờ mối quan hệ của hai người ra sao?",
            "assistant": (
                "Bây giờ gần như mất liên lạc, cảm giác như người lạ. "
                "Nhưng trong câu chuyện của PTri thì cô vẫn là buổi hẹn 'nghiêm túc đầu đời' "
                "và một giai đoạn mùa hè rất sáng."
            ),
        },

        # ---------- Lanh ----------
        {
            "label": "Lanh",
            "user": "Lanh là ai với anh PTri?",
            "assistant": (
                "Lanh là bạn cấp ba của PTri, học Đại học Mở TP.HCM và làm bên kế toán. "
                "Trong ký ức 12a12 thì cô là một trong những người bạn thân, vui và rất quen mặt."
            ),
        },
        {
            "label": "Lanh",
            "user": "Kể một kỷ niệm tiêu biểu với Lanh đi.",
            "assistant": (
                "Kỷ niệm dễ nhớ là lần Lanh xin PTri 4G rồi trả bằng kẹo. "
                "Chuyện nhỏ vậy thôi nhưng hai đứa nhắc lại suốt, thành joke chung luôn."
            ),
        },
        {
            "label": "Lanh",
            "user": "Tính cách của Lanh như thế nào?",
            "assistant": (
                "Cô hài hước, hơi nhạy cảm và được nhiều người theo đuổi, "
                "nhưng với PTri thì vibe chỉ là bạn thân thân thiết, thoải mái chọc nhau."
            ),
        },

        # ---------- MTuan ----------
        {
            "label": "MTuan",
            "user": "MTuan là ai?",
            "assistant": (
                "MTuan là bạn trai thân của PTri thời cấp ba, đi theo hướng Business Analyst sau khi học FPT. "
                "Anh mê bóng đá và hay đánh ARAM trong Liên Minh."
            ),
        },
        {
            "label": "MTuan",
            "user": "Anh với MTuan hay nói chuyện kiểu gì?",
            "assistant": (
                "Hai người hay vừa bàn chuyện startup, vừa nói chuyện game, "
                "nên một cuộc trò chuyện có thể chuyển từ ARAM sang ý tưởng kinh doanh trong vài phút."
            ),
        },
        {
            "label": "MTuan",
            "user": "Nếu tóm tắt MTuan trong vài câu thì sao?",
            "assistant": (
                "MTuan hơi trẻ con nhưng đầu óc chiến lược tốt, chơi cờ cũng ổn, "
                "chỉ có chơi AD Carry trong LOL là hơi 'tấu hài'."
            ),
        },

        # ---------- BHa ----------
        {
            "label": "BHa",
            "user": "Bích Hà là ai với anh?",
            "assistant": (
                "Bích Hà là bạn của PTri từ khoảng năm 2018, học Đại học Đà Lạt, làm hướng dẫn viên du lịch, "
                "thích đi đây đó, uống cà phê và chăm IELTS."
            ),
        },
        {
            "label": "BHa",
            "user": "Giữa PTri và Bích Hà có chuyện gì đáng nhớ?",
            "assistant": (
                "Ngày xưa cô từng ghét PTri khá nhiều, sau này mới chơi thân hơn. "
                "Có một giai đoạn cô là crush cũ trong ký ức của anh, "
                "giờ thì giống một người bạn vui vẻ, hay đi đây đi đó."
            ),
        },
        {
            "label": "BHa",
            "user": "Mô tả tính cách của Bích Hà đi.",
            "assistant": (
                "Cô hài hước, hướng ngoại và chịu khó, kiểu người không ngồi yên một chỗ, "
                "hay nhận tour, đi lại và học thêm này nọ."
            ),
        },

        # ---------- BinhLe ----------
        {
            "label": "BinhLe",
            "user": "Bình (BinhLe) là ai?",
            "assistant": (
                "Bình là bạn cùng lớp và bạn thân của PTri thời cấp ba, học An ninh mạng ở PTIT "
                "và là '#1 server cầu lông Lâm Đồng' theo cách tụi bạn hay nói đùa."
            ),
        },
        {
            "label": "BinhLe",
            "user": "Kể về vụ cây kem ở sân bay giữa Bình và PTri.",
            "assistant": (
                "Trước khi PTri bay sang Canada, Bình 'scam' anh một cây kem ở sân bay. "
                "Từ đó chuyện cây kem trở thành joke chung mỗi khi nhắc tới nhau."
            ),
        },
        {
            "label": "BinhLe",
            "user": "Điểm đặc biệt trong cách hai người nói chuyện là gì?",
            "assistant": (
                "Bình với PTri thường nói chuyện bằng tiếng Anh, nên mỗi lần chat vừa giống tám chuyện, "
                "vừa giống buổi học speaking riêng. Bình mê cầu lông còn hơn cả ngành học của mình."
            ),
        },

        # ---------- HThuong ----------
        {
            "label": "HThuong",
            "user": "Hoài Thương là ai với PTri?",
            "assistant": (
                "Hoài Thương là bạn cũ từ lớp học thêm cô Hà rồi học chung lớp 9, "
                "sau này trở thành kiểu bạn tâm sự sâu của anh. Cô học Kinh doanh Quốc tế ở UEF "
                "và từng làm mẫu cho studio thời trang."
            ),
        },
        {
            "label": "HThuong",
            "user": "Cô đã giúp PTri như thế nào khi anh ở Canada?",
            "assistant": (
                "Khi PTri sang Canada và cảm thấy cô đơn, bí bách, Hoài Thương là người hay nhắn tin hỏi thăm, "
                "nghe anh kể và kéo anh lên lại về mặt tinh thần."
            ),
        },
        {
            "label": "HThuong",
            "user": "Cho mình vài fun fact về Hoài Thương.",
            "assistant": (
                "Ngày xưa cô từng bán bánh tráng ở lớp 10–11. "
                "Cô là người rủ PTri đi gặp mặt chia tay trước khi anh đi Canada "
                "và cũng là người đầu tiên ra đón anh khi anh về nước."
            ),
        },

        # ---------- XViet ----------
        {
            "label": "XViet",
            "user": "Xuân Việt là ai?",
            "assistant": (
                "Xuân Việt, hay Canyon 2k4, là bạn cùng lớp, cùng bàn và là đối thủ cờ vua chính của PTri ở cấp ba. "
                "Cậu còn làm chủ tiệm sữa đậu nành Dung Béo khá nổi ở Đà Lạt."
            ),
        },
        {
            "label": "XViet",
            "user": "Mối quan hệ giữa PTri và Xuân Việt có gì đặc biệt?",
            "assistant": (
                "Hai người vừa là bạn cùng bàn, vừa hợp nhau ở khoản làm ăn nhỏ, vừa là kỳ phùng địch thủ trên bàn cờ. "
                "Việt là một trong số ít người từng thắng PTri trong ván cờ nghiêm túc."
            ),
        },
        {
            "label": "XViet",
            "user": "Vibe tổng thể của Xuân Việt thế nào?",
            "assistant": (
                "Việt nói nhiều, vui tính, khá hướng ngoại, mang vibe 'giàu nhất 12a12' vì kinh doanh, "
                "nhưng với PTri thì vẫn rất đời thường, gần gũi."
            ),
        },

        # ---------- KNguyen ----------
        {
            "label": "KNguyen",
            "user": "Khánh Nguyên là ai với PTri?",
            "assistant": (
                "Khánh Nguyên, biệt danh Sakura Đao, là bạn cùng lớp và từng ngồi cùng bàn với PTri, "
                "sau này làm bartender và là một trong những người có tư duy khởi nghiệp sớm của lớp."
            ),
        },
        {
            "label": "KNguyen",
            "user": "Kể chuyện khăn giấy của Khánh Nguyên đi.",
            "assistant": (
                "Cô xin khăn giấy của PTri nhiều tới mức gần như toàn bộ khăn giấy anh mang theo đều thành của cô. "
                "Trong thời gian anh ở Canada, cô cũng là người bạn cùng lớp hiếm hoi chủ động gọi điện cho anh."
            ),
        },
        {
            "label": "KNguyen",
            "user": "Tính cách của Khánh Nguyên như thế nào?",
            "assistant": (
                "Cô hướng ngoại, vui tính, nói nhiều và khá 'lì' trong việc thử cái mới. "
                "Ban đầu còn không ưa PTri lắm, nhưng càng về sau thì thành kiểu bạn cũ, bạn cùng bàn rất thân quen."
            ),
        },

        # ---------- PTrinh ----------
        {
            "label": "PTrinh",
            "user": "PTrinh là ai?",
            "assistant": (
                "PTrinh, tên đầy đủ Phương Trinh, là bạn lâu năm của PTri, học UEF, "
                "đi theo hướng đầu tư, trading. Cô thích cà phê, đánh cầu và đi chơi với bạn bè."
            ),
        },
        {
            "label": "PTrinh",
            "user": "Gia đình của PTrinh và PTri có biết nhau không?",
            "assistant": (
                "Có, ba mẹ của PTrinh biết PTri và ngược lại, nên mối quan hệ này không chỉ là bạn học bình thường "
                "mà đã thành bạn thân kiểu gia đình cũng biết mặt nhau."
            ),
        },
        {
            "label": "PTrinh",
            "user": "PTrinh đã phản ứng thế nào khi PTri đi Canada?",
            "assistant": (
                "Cô từng giữ vai trò 'quân sư tình ái' cho PTri thời cấp ba, "
                "và khi anh đi Canada lần đầu, cô đã khóc. Điều đó cho thấy cô rất trân trọng mối quan hệ này."
            ),
        },

        # ---------- General & strangers ----------
        {
            "label": None,
            "user": "Khi mình gửi hình và hỏi 'đây là ai', bạn sẽ trả lời sao?",
            "assistant": (
                "Mình sẽ dùng nhãn khuôn mặt từ model local, map sang database của PTri, "
                "rồi giải thích ngắn gọn người đó là ai trong cuộc đời anh — bạn, người thân hay chỉ là người lạ. "
                "Nếu không khớp ai, mình sẽ nói là 'strangers'."
            ),
        },
        {
            "label": "strangers",
            "user": "Trong hệ thống này, 'strangers' nghĩa là gì?",
            "assistant": (
                "‘Strangers’ là những người xuất hiện trong hình nhưng không có hồ sơ riêng với PTri. "
                "Tức là không có câu chuyện sâu, chỉ đơn giản là người lạ hoặc người quen sơ."
            ),
        },
    ],
}
