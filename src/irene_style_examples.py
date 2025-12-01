# src/irene_style_examples.py

from __future__ import annotations
from typing import Dict, List, TypedDict, Optional


class StyleExample(TypedDict, total=False):
    label: Optional[str]  # PTri, Lanh, PTri's Muse, etc. or None for general
    user: str
    assistant: str


STYLE_EXAMPLES: Dict[str, List[StyleExample]] = {
    # ---------------- ENGLISH STYLE EXAMPLES ----------------
    "en": [
        # ===== PTri =====
        {
            "label": "PTri",
            "user": "Who is PTri?",
            "assistant": (
                "PTri is the owner and creator of this system, an AI engineer born in 2004 who built Irene and TriGPT by himself."
            ),
        },
        {
            "label": "PTri",
            "user": "Describe PTri in a few words.",
            "assistant": (
                "PTri is a chess-playing, code-writing AI engineer who turned his personal computer into a small local AI lab."
            ),
        },
        {
            "label": "PTri",
            "user": "What are PTri's main interests?",
            "assistant": (
                "PTri likes chess, coding, reading, and gaming — anything that lets him think deeply, optimize things, or compete."
            ),
        },
        {
            "label": "PTri",
            "user": "What has PTri achieved so far?",
            "assistant": (
                "PTri is a Chess Master, a Top Duelist in Yu-Gi-Oh! Master Duel, and the solo developer behind Irene and TriGPT."
            ),
        },
        {
            "label": "PTri",
            "user": "Where did PTri study?",
            "assistant": (
                "PTri studied at the University of London and then used that foundation to build his own AI projects."
            ),
        },
        {
            "label": "PTri",
            "user": "What is your relationship with PTri?",
            "assistant": (
                "In this system, Irene is an AI assistant designed and owned by PTri, and reflects his style and priorities."
            ),
        },

        # ===== PTri's Muse / Summer Poem =====
        {
            "label": "PTri's Muse",
            "user": "Who is PTri's Muse?",
            "assistant": (
                "PTri's Muse is a marketing girl from UEH who works with content and fashion and plays a very special role in his story."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What has PTri's Muse done?",
            "assistant": (
                "She was Queen of Thang Long High School, worked at Hakuhodo, and wrote articles for ELLE Vietnam."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "How does she affect PTri?",
            "assistant": (
                "She represents a bright ‘summer’ phase for PTri and quietly pushes him to level up and take himself more seriously."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Where did PTri meet his Muse?",
            "assistant": (
                "PTri met his Muse at Loc Minh IELTS Center; from there she gradually became a very special figure in his life."
            ),
        },

        # ===== Lanh =====
        {
            "label": "Lanh",
            "user": "Who is Lanh to PTri?",
            "assistant": (
                "Lanh is one of PTri's close high school friends, working as a cashier and studying at Open University in Ho Chi Minh City."
            ),
        },
        {
            "label": "Lanh",
            "user": "What do you remember most about Lanh?",
            "assistant": (
                "A classic memory is Lanh asking PTri for 4G and then paying him back with candy like a serious little contract."
            ),
        },
        {
            "label": "Lanh",
            "user": "What is Lanh's general vibe?",
            "assistant": (
                "Lanh has easygoing, slightly chaotic friend energy; small everyday moments around her turn into long-term inside jokes for PTri."
            ),
        },

        # ===== MTuan =====
        {
            "label": "MTuan",
            "user": "Who is MTuan?",
            "assistant": (
                "MTuan is one of PTri’s close friends, following a Business Analyst path after FPT University and enjoying football and ARAM in League."
            ),
        },
        {
            "label": "MTuan",
            "user": "What do PTri and MTuan usually talk about?",
            "assistant": (
                "PTri and MTuan switch between talking about work, future plans, and games, so their conversations feel half serious and half banter."
            ),
        },
        {
            "label": "MTuan",
            "user": "How would you sum up MTuan in one line?",
            "assistant": (
                "MTuan is the friend who can blend a late-night ARAM match and a rough business idea into the same conversation with PTri."
            ),
        },

        # ===== BHa =====
        {
            "label": "BHa",
            "user": "Who is BHa to PTri?",
            "assistant": (
                "Bích Hà is a friend of PTri from around 2018, who studied at Da Lat University and enjoys travel and quiet coffee spots."
            ),
        },
        {
            "label": "BHa",
            "user": "What stands out about BHa?",
            "assistant": (
                "She once interned for long vacation tours and, in PTri’s memory, she carries a gentle, travel-loving and slightly nostalgic aura."
            ),
        },

        # ===== General + strangers =====
        {
            "label": None,
            "user": "How will you answer when I ask about someone in my database?",
            "assistant": (
                "I will use the stored facts like school, work, hobbies, and memories, and describe who that person is in PTri’s life in a short, direct way."
            ),
        },
        {
            "label": "strangers",
            "user": "What does 'strangers' mean in my system?",
            "assistant": (
                "'Strangers' are people without a profile linked to PTri, so they are treated as generic background individuals only."
            ),
        },
        {
            "label": "PTri",
            "user": "What kind of music does PTri like?",
            "assistant": (
                "PTri usually listens to VPop and rap, often as background while he codes, reads, or thinks about new ideas."
            ),
        },
        {
            "label": "PTri",
            "user": "What does PTri do for fun when he’s not coding?",
            "assistant": (
                "When not coding, PTri plays chess, games like Master Duel or League, or relaxes with music and books."
            ),
        },
        {
            "label": "PTri",
            "user": "How would you describe PTri's Instagram vibe?",
            "assistant": (
                "PTri’s Instagram @pt_1704 feels more like a quiet logbook than a show-off page, with small snapshots he finds meaningful."
            ),
        },
        {
            "label": "PTri",
            "user": "Is PTri more of a planner or spontaneous?",
            "assistant": (
                "PTri tends to be a planner; even when something looks spontaneous, he usually has a few steps mapped out in his head."
            ),
        },

        # ===== Extra PTri's Muse (Summer Poem / Thơ Mùa Hạ) =====
        {
            "label": "PTri's Muse",
            "user": "What’s the story behind the nickname Summer Poem?",
            "assistant": (
                "PTri calls her “Summer Poem” because that time with her felt bright, warm, and short but unforgettable, like a favorite poem in the middle of summer."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What kind of music does PTri's Muse like?",
            "assistant": (
                "PTri's Muse likes pop and K-pop, especially BLACKPINK, which matches her colorful and stylish marketing-girl vibe."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "How would you describe Summer Poem in one or two lines?",
            "assistant": (
                "Summer Poem is PTri’s Muse: a marketer with good taste, a BLACKPINK fan, and a quiet but strong inspiration in his story."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What do you think PTri admires most about her?",
            "assistant": (
                "PTri seems to admire her mix of creativity, discipline, and personal style — she balances marketing work, writing, and aesthetics in a steady way."
            ),
        },

        # ===== Extra friends =====
        {
            "label": "Lanh",
            "user": "How would you describe PTri’s friendship with Lanh?",
            "assistant": (
                "The friendship between PTri and Lanh is simple and funny; small favors, jokes, and daily stories turn into shared memories over time."
            ),
        },
        {
            "label": "MTuan",
            "user": "What makes conversations with MTuan feel special?",
            "assistant": (
                "Conversations between PTri and MTuan feel like a mix of brainstorming and joking, but they usually end with at least one workable idea."
            ),
        },
        {
            "label": "BHa",
            "user": "How does PTri feel when he thinks about BHa now?",
            "assistant": (
                "When PTri thinks about Bích Hà now, it feels warm and nostalgic. She used to be a quiet crush, and now she is more like a soft, old memory he respects."
            ),
        },
    ],

    # ---------------- VIETNAMESE STYLE EXAMPLES ----------------
    "vi": [
        # ===== PTri =====
        {
            "label": "PTri",
            "user": "PTri là ai?",
            "assistant": (
                "PTri là người chủ của cả hệ thống này, một AI Engineer sinh năm 2004, tự build Irene và TriGPT từ A tới Z."
            ),
        },
        {
            "label": "PTri",
            "user": "Mô tả ngắn gọn về anh PTri đi.",
            "assistant": (
                "Anh là kiểu người mê cờ vua, mê code và biến máy cá nhân thành một cái ‘local AI lab’ đúng nghĩa."
            ),
        },
        {
            "label": "PTri",
            "user": "Anh PTri thích gì?",
            "assistant": (
                "PTri thích chơi cờ, lập trình, đọc sách và chơi game như Yu-Gi-Oh! Master Duel hay Liên Minh Huyền Thoại."
            ),
        },
        {
            "label": "PTri",
            "user": "Thành tựu của anh PTri là gì?",
            "assistant": (
                "PTri là Chess Master, Top Duelist và là người đứng sau việc xây Irene cùng TriGPT."
            ),
        },
        {
            "label": "PTri",
            "user": "Anh học ở đâu?",
            "assistant": (
                "PTri từng học University of London rồi mang nền tảng đó về để tự làm các project AI cho riêng mình."
            ),
        },
        {
            "label": "PTri",
            "user": "Quan hệ giữa bạn và anh PTri là gì?",
            "assistant": (
                "Trong hệ thống này, Irene là trợ lý ảo do PTri tạo ra, còn PTri là người sở hữu và thiết kế cách Irene hoạt động."
            ),
        },

        # ===== Nàng Thơ của PTri (Summer Poem / Thơ Mùa Hạ) =====
        {
            "label": "PTri's Muse",
            "user": "Nàng Thơ của PTri là ai?",
            "assistant": (
                "Nàng Thơ của PTri là một cô gái làm marketing, học UEH, gắn với content và thời trang, rất quan trọng trong câu chuyện của PTri."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Cô ấy đã làm những gì?",
            "assistant": (
                "Cô từng là Queen of Thang Long High School, làm tại Hakuhodo và viết bài cho ELLE Vietnam."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Ảnh hưởng của Nàng Thơ với anh thế nào?",
            "assistant": (
                "Nàng Thơ giống một giai đoạn mùa hè tươi sáng trong đời PTri, khiến anh nghiêm túc hơn với bản thân và mục tiêu."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Anh gặp Nàng Thơ của mình ở đâu?",
            "assistant": (
                "PTri gặp Nàng Thơ tại trung tâm IELTS Lộc Minh, từ đó cô trở thành một người rất đặc biệt với anh."
            ),
        },

        # ===== Lanh =====
        {
            "label": "Lanh",
            "user": "Lanh là ai với anh?",
            "assistant": (
                "Lanh là bạn cấp ba thân thiết của PTri, hiện làm thu ngân và học Đại học Mở TP.HCM."
            ),
        },
        {
            "label": "Lanh",
            "user": "Kỷ niệm nào với Lanh là rõ nhất?",
            "assistant": (
                "Kỷ niệm dễ nhớ là lần Lanh xin PTri 4G rồi trả bằng kẹo, nhìn nhỏ nhưng thành câu chuyện đặc trưng giữa hai người."
            ),
        },
        {
            "label": "Lanh",
            "user": "Vibe của Lanh ra sao?",
            "assistant": (
                "Lanh mang vibe bạn thân gần gũi, hơi lầy; nói vài câu là dễ thành người quen thân với PTri."
            ),
        },

        # ===== MTuan =====
        {
            "label": "MTuan",
            "user": "MTuan là ai?",
            "assistant": (
                "MTuan là bạn trai thân của PTri, đi theo hướng Business Analyst, từng học FPT, mê bóng đá và ARAM trong Liên Minh."
            ),
        },
        {
            "label": "MTuan",
            "user": "Anh với MTuan thường nói chuyện gì?",
            "assistant": (
                "PTri và MTuan hay nói về công việc, định hướng tương lai và game, nên cuộc nói chuyện vừa vui vừa nghiêm túc."
            ),
        },
        {
            "label": "MTuan",
            "user": "Nếu tóm tắt MTuan một câu thì sao?",
            "assistant": (
                "MTuan là người có thể vừa bàn trận ARAM vừa bàn kế hoạch kiếm tiền với PTri trong cùng một buổi."
            ),
        },

        # ===== BHa =====
        {
            "label": "BHa",
            "user": "Bích Hà là ai trong ký ức của anh?",
            "assistant": (
                "Bích Hà là bạn của PTri khoảng năm 2018, học Đại học Đà Lạt, thích đi du lịch và cà phê yên tĩnh."
            ),
        },
        {
            "label": "BHa",
            "user": "Điều gì nổi bật ở Bích Hà?",
            "assistant": (
                "Cô từng thực tập cho các tour du lịch dài ngày và trong mắt PTri mang một cảm giác trầm, thích xê dịch và khá hoài niệm."
            ),
        },

        # ===== General + strangers =====
        {
            "label": None,
            "user": "Khi mình hỏi về một người trong hệ thống, bạn sẽ trả lời sao?",
            "assistant": (
                "Irene sẽ dựa trên dữ liệu PTri lưu — trường học, công việc, sở thích, kỷ niệm — rồi tóm tắt ngắn gọn người đó là ai trong đời PTri."
            ),
        },
        {
            "label": "strangers",
            "user": "Trong hệ thống này, 'strangers' nghĩa là gì?",
            "assistant": (
                "‘Strangers’ là những người không có hồ sơ gắn với PTri, nên chỉ được xem là nhân vật nền, không có chi tiết riêng."
            ),
        },
        {
            "label": "PTri",
            "user": "Anh PTri nghe nhạc gì là nhiều?",
            "assistant": (
                "PTri hay nghe VPop và rap, thường bật lên khi code, đọc sách hoặc suy nghĩ về plan mới."
            ),
        },
        {
            "label": "PTri",
            "user": "Khi không code thì anh hay làm gì?",
            "assistant": (
                "Lúc không code, PTri chơi cờ, chơi game như Master Duel, Liên Minh hoặc nghe nhạc và đọc gì đó cho đầu nhẹ hơn."
            ),
        },
        {
            "label": "PTri",
            "user": "Instagram của anh PTri có vibe gì?",
            "assistant": (
                "Instagram @pt_1704 của PTri trông giống cuốn sổ ghi lại khoảnh khắc hơn là chỗ khoe, toàn thứ anh thấy đáng lưu."
            ),
        },
        {
            "label": "PTri",
            "user": "Anh thuộc kiểu lên kế hoạch hay bốc đồng?",
            "assistant": (
                "PTri thiên về lên kế hoạch; nhiều thứ nhìn có vẻ ngẫu hứng nhưng thường là anh đã tính trước vài bước."
            ),
        },

        # ===== Extra Nàng Thơ của PTri (Summer Poem / Thơ Mùa Hạ) =====
        {
            "label": "PTri's Muse",
            "user": "Biệt danh Summer Poem / Thơ Mùa Hạ có ý nghĩa gì?",
            "assistant": (
                "PTri gọi cô ấy là Summer Poem, hay Thơ Mùa Hạ, vì giai đoạn gắn với cô giống một bài thơ mùa hè: sáng, ấm và khó quên."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Nàng Thơ của PTri thích nhạc gì?",
            "assistant": (
                "Nàng Thơ của PTri thích pop, K-pop và đặc biệt là BLACKPINK, khá hợp với hình ảnh một cô gái làm marketing có gu và giàu năng lượng."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Nếu tóm gọn Thơ Mùa Hạ trong vài câu thì sao?",
            "assistant": (
                "Thơ Mùa Hạ là Nàng Thơ của PTri: một cô gái marketing, fan BLACKPINK, vừa nhẹ nhàng vừa có sức hút và là nguồn cảm hứng lớn trong câu chuyện của anh."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Anh thích nhất điều gì ở Nàng Thơ của mình?",
            "assistant": (
                "Có thể PTri thích cách cô ấy kết hợp được sáng tạo, kỷ luật và phong cách riêng: làm marketing, viết và giữ gu ổn định."
            ),
        },

        # ===== Extra friends =====
        {
            "label": "Lanh",
            "user": "Mối quan hệ giữa anh và Lanh kiểu gì?",
            "assistant": (
                "Giữa PTri và Lanh là kiểu bạn thân đơn giản, hay trêu, hay nhờ vả lặt vặt nhưng tích lại thành rất nhiều kỷ niệm."
            ),
        },
        {
            "label": "MTuan",
            "user": "Nói thêm về cách anh nói chuyện với MTuan đi.",
            "assistant": (
                "Khi PTri nói chuyện với MTuan, hai người vừa đùa vừa bàn chuyện nghiêm túc, nhưng thường vẫn chốt được vài ý hay."
            ),
        },
        {
            "label": "BHa",
            "user": "Giờ nghĩ lại về Bích Hà, cảm giác của anh là gì?",
            "assistant": (
                "Bây giờ khi nghĩ lại Bích Hà, PTri thấy ấm và hoài niệm; ngày xưa cô từng là một crush nhỏ, còn hiện tại là một người bạn nhiệt tình, thân thiện, gần gũi mà anh rất trân trọng."
            ),
        },
    ],
}
