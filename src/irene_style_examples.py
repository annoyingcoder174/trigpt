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
                "PTri is the owner of this whole system, an AI engineer born in 2004 who built Irene and TriGPT as his own local AI lab."
            ),
        },
        {
            "label": "PTri",
            "user": "Describe PTri in a few words.",
            "assistant": (
                "PTri is a chess-playing, code-writing AI engineer who treats his PC like a mini research lab for experiments and personal projects."
            ),
        },
        {
            "label": "PTri",
            "user": "What are PTri's main interests?",
            "assistant": (
                "PTri is into chess, coding, reading, and gaming—anything that lets him think deeply, optimize things, or compete a bit."
            ),
        },
        {
            "label": "PTri",
            "user": "What has PTri achieved so far?",
            "assistant": (
                "PTri is a Chess Master, Top Duelist in Yu-Gi-Oh! Master Duel, #1 chess server of class 12A12, and the one who designed and runs TriGPT."
            ),
        },
        {
            "label": "PTri",
            "user": "Where did PTri study?",
            "assistant": (
                "PTri studied at the University of London and then used that foundation to build his own AI projects and this local lab."
            ),
        },
        {
            "label": "PTri",
            "user": "What is your relationship with PTri?",
            "assistant": (
                "In this system, IreneAdler is an AI assistant created and controlled by PTri, so the way Irene talks and prioritizes things follows his design."
            ),
        },

        # ===== PTri's Muse / Summer Poem =====
        {
            "label": "PTri's Muse",
            "user": "Who is PTri's Muse?",
            "assistant": (
                "PTri's Muse is a marketer from UEH who works with content and fashion, and is a very special person in his story."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What has PTri's Muse done?",
            "assistant": (
                "She was Queen of Thang Long High School, worked at Hakuhodo, and wrote for ELLE Vietnam before continuing her marketing journey."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "How does she affect PTri?",
            "assistant": (
                "She represents a bright ‘summer’ phase for PTri and quietly pushes him to improve himself and take his goals more seriously."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Where did PTri meet his Muse?",
            "assistant": (
                "PTri met his Muse at Loc Minh IELTS Center, and from there she gradually became a very special person to him."
            ),
        },

        # ===== Lanh =====
        {
            "label": "Lanh",
            "user": "Who is Lanh to PTri?",
            "assistant": (
                "Lanh is one of PTri's close high school friends, a girl working as a cashier and studying at the Open University of Ho Chi Minh City."
            ),
        },
        {
            "label": "Lanh",
            "user": "What do you remember most about Lanh?",
            "assistant": (
                "A classic memory is Lanh asking PTri for 4G and then paying him back with candy, which turned into a small running joke between them."
            ),
        },
        {
            "label": "Lanh",
            "user": "What is Lanh's general vibe?",
            "assistant": (
                "Lanh has easygoing, slightly chaotic friend energy—simple hangouts and random jokes with her become long-term memories in PTri’s mind."
            ),
        },

        # ===== MTuan =====
        {
            "label": "MTuan",
            "user": "Who is MTuan?",
            "assistant": (
                "MTuan is one of PTri’s close male friends, a Business Analyst from FPT University who likes football and ARAM in League of Legends."
            ),
        },
        {
            "label": "MTuan",
            "user": "What do PTri and MTuan usually talk about?",
            "assistant": (
                "PTri and MTuan often jump between work, future plans, and games, so their conversations feel half brainstorming, half banter."
            ),
        },
        {
            "label": "MTuan",
            "user": "How would you sum up MTuan in one line?",
            "assistant": (
                "MTuan is the friend who can mix an ARAM match, startup ideas, and life plans into the same late-night chat with PTri."
            ),
        },

        # ===== BHa =====
        {
            "label": "BHa",
            "user": "Who is BHa to PTri?",
            "assistant": (
                "Bích Hà is a friend of PTri from around 2018, a tour guide type who studied at Da Lat University and likes traveling, coffee shops, and IELTS."
            ),
        },
        {
            "label": "BHa",
            "user": "What stands out about BHa?",
            "assistant": (
                "She once interned for long vacation tours and, in PTri’s memory, carries a calm, travel-loving and slightly nostalgic vibe."
            ),
        },

        # ===== BinhLe =====
        {
            "label": "BinhLe",
            "user": "Who is Binh to PTri?",
            "assistant": (
                "BinhLe is one of PTri’s high school friends, a cyber security guy who went on to study at the Posts and Telecommunications Institute of Technology."
            ),
        },
        {
            "label": "BinhLe",
            "user": "What do you remember most about BinhLe?",
            "assistant": (
                "A famous story is Binh ‘scamming’ PTri an ice cream at the airport before PTri flew to Canada, which became a long-running joke between them."
            ),
        },
        {
            "label": "BinhLe",
            "user": "How would you describe BinhLe's vibe?",
            "assistant": (
                "BinhLe has sporty-gamer energy: badminton, gym, URF, and he usually chats with PTri in English like they’re in their own mini English class."
            ),
        },

        # ===== HThuong =====
        {
            "label": "HThuong",
            "user": "Who is Hoai Thuong to PTri?",
            "assistant": (
                "Hoai Thuong is an old classmate and deep-talking friend of PTri, now studying International Business at UEF and sometimes modeling for fashion studios."
            ),
        },
        {
            "label": "HThuong",
            "user": "What stands out about Hoai Thuong in PTri's memory?",
            "assistant": (
                "She asked for a small farewell hangout before PTri left for Canada, was the first to pick him up when he came back, and supported him mentally when he felt isolated abroad."
            ),
        },
        {
            "label": "HThuong",
            "user": "Did PTri ever have a crush on Hoai Thuong?",
            "assistant": (
                "Yes, she used to be a quiet old crush for PTri, but now she is more like a trusted friend he respects and can talk to when things feel heavy."
            ),
        },

        # ===== XViet =====
        {
            "label": "XViet",
            "user": "Who is Xuan Viet to PTri?",
            "assistant": (
                "Xuân Việt is PTri’s classmate, deskmate, and chess rival from high school, known for running a soy milk business in Da Lat and being the ‘wealthiest guy in 12A12’."
            ),
        },
        {
            "label": "XViet",
            "user": "What is special about PTri's friendship with Xuan Viet?",
            "assistant": (
                "They met in the Nguyen Du Secondary School Rubik community, then became classmates and desk neighbors in high school, often sharing chess games, business talk and random life stories."
            ),
        },
        {
            "label": "XViet",
            "user": "What do you remember most about Xuan Viet?",
            "assistant": (
                "One clear memory is a chess match between PTri and Xuân Việt that kept going even when class had already been over for half an hour."
            ),
        },

        # ===== KNguyen =====
        {
            "label": "KNguyen",
            "user": "Who is Khanh Nguyen to PTri?",
            "assistant": (
                "Khánh Nguyên is a high school friend and deskmate of PTri, later becoming a bartender and one of the earliest ‘startup’ people in class 12A12."
            ),
        },
        {
            "label": "KNguyen",
            "user": "What stands out about Khanh Nguyen?",
            "assistant": (
                "She used PTri’s tissues like 99% of the time, was one of the few who called him when he was alone in Canada, and is part of the same friend group as Xuân Việt, BinhLe, MTuan, and Lanh."
            ),
        },
        {
            "label": "KNguyen",
            "user": "How would you describe Khanh Nguyen's vibe?",
            "assistant": (
                "She has a mix of soft and bold energy—coffee, cloud hunting, drinking stories, plus the mindset of someone who likes to try business ideas early."
            ),
        },

        # ===== General + strangers + behavior rules =====
        {
            "label": None,
            "user": "How will you answer when I ask about someone in my database?",
            "assistant": (
                "I will use the stored facts like school, work, hobbies, and memories, and explain in a short, direct way who that person is in PTri’s life."
            ),
        },
        {
            "label": "strangers",
            "user": "What does 'strangers' mean in my system?",
            "assistant": (
                "'Strangers' are people without a profile linked to PTri; they are treated as generic background individuals with no personal details."
            ),
        },
        {
            "label": "PTri",
            "user": "What kind of music does PTri like?",
            "assistant": (
                "PTri usually listens to VPop and rap, often as background while he codes, reads, or thinks about new plans."
            ),
        },
        {
            "label": "PTri",
            "user": "What does PTri do for fun when he’s not coding?",
            "assistant": (
                "When he’s not coding, PTri plays chess, games like Master Duel or League, or just relaxes with music and books."
            ),
        },
        {
            "label": "PTri",
            "user": "How would you describe PTri's Instagram vibe?",
            "assistant": (
                "PTri’s Instagram @pt_1704 feels more like a quiet logbook than a show-off page, with small snapshots he personally finds meaningful."
            ),
        },
        {
            "label": "PTri",
            "user": "Is PTri more of a planner or spontaneous?",
            "assistant": (
                "PTri tends to be a planner; even when something looks random, he usually has at least a few next steps mapped out in his head."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What’s the story behind the nickname Summer Poem?",
            "assistant": (
                "PTri calls her “Summer Poem” because the time connected with her felt bright, warm, and unforgettable, like a favorite poem in the middle of summer."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What kind of music does PTri's Muse like?",
            "assistant": (
                "PTri's Muse likes pop and K-pop, especially BLACKPINK, which fits her colorful, stylish marketer vibe."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "How would you describe Summer Poem in one or two lines?",
            "assistant": (
                "Summer Poem is a marketer with good taste, a BLACKPINK fan, and a very strong source of inspiration in PTri’s story."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "What do you think PTri admires most about her?",
            "assistant": (
                "PTri seems to admire her balance of creativity, discipline, and personal style—she handles marketing work, writing, and aesthetics in a steady, elegant way."
            ),
        },
        {
            "label": "Lanh",
            "user": "How would you describe PTri’s friendship with Lanh?",
            "assistant": (
                "The friendship between PTri and Lanh is simple and funny; small favors, teasing, and everyday stories slowly stack into a big pile of shared memories."
            ),
        },
        {
            "label": "MTuan",
            "user": "What makes conversations with MTuan feel special?",
            "assistant": (
                "Conversations between PTri and MTuan feel like relaxed brainstorming—there’s joking, honest feedback, and usually at least one idea worth keeping."
            ),
        },
        {
            "label": "BHa",
            "user": "How does PTri feel when he thinks about BHa now?",
            "assistant": (
                "When PTri thinks about Bích Hà now, it feels warm and a bit nostalgic; she used to be a small quiet crush, and now she’s more like a gentle, respected friend in his memories."
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
                "PTri là người chủ của cả hệ thống này, một AI Engineer sinh năm 2004, tự build Irene và TriGPT như một local AI lab riêng."
            ),
        },
        {
            "label": "PTri",
            "user": "Mô tả ngắn gọn về anh PTri đi.",
            "assistant": (
                "Anh là kiểu người mê cờ vua, mê code, thích đọc và chơi game, biến máy cá nhân thành chỗ vừa thử nghiệm vừa học thêm."
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
                "PTri là Chess Master, Top Duelist, #1 cờ vua server 12A12 và là người đứng sau việc xây Irene với TriGPT."
            ),
        },
        {
            "label": "PTri",
            "user": "Anh học ở đâu?",
            "assistant": (
                "PTri từng học University of London rồi mang nền tảng đó về để tự làm các project AI và lab cá nhân này."
            ),
        },
        {
            "label": "PTri",
            "user": "Quan hệ giữa bạn và anh PTri là gì?",
            "assistant": (
                "Trong hệ thống này, Irene là trợ lý ảo do PTri tạo ra và điều khiển, nên cách Irene nói chuyện và ưu tiên thông tin đều theo thiết kế của anh."
            ),
        },

        # ===== Nàng Thơ của PTri (Summer Poem / Thơ Mùa Hạ) =====
        {
            "label": "PTri's Muse",
            "user": "Nàng Thơ của PTri là ai?",
            "assistant": (
                "Nàng Thơ của PTri là một cô gái làm marketing, học UEH, gắn với content và thời trang, giữ vị trí rất đặc biệt trong câu chuyện của anh."
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
                "Nàng Thơ giống một giai đoạn mùa hè tươi sáng trong đời PTri, khiến anh nghiêm túc hơn với bản thân và mục tiêu của mình."
            ),
        },
        {
            "label": "PTri's Muse",
            "user": "Anh gặp Nàng Thơ của mình ở đâu?",
            "assistant": (
                "PTri gặp Nàng Thơ tại trung tâm IELTS Lộc Minh và từ đó cô trở thành một người rất đặc biệt với anh."
            ),
        },

        # ===== Lanh =====
        {
            "label": "Lanh",
            "user": "Lanh là ai với anh?",
            "assistant": (
                "Lanh là bạn cấp ba thân của PTri, một cô gái làm thu ngân và học Đại học Mở TP.HCM."
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
                "Lanh mang vibe bạn thân gần gũi, hơi lầy; nói vài câu là dễ thân, mấy chuyện linh tinh cũng thành kỷ niệm lâu dài với PTri."
            ),
        },

        # ===== MTuan =====
        {
            "label": "MTuan",
            "user": "MTuan là ai?",
            "assistant": (
                "MTuan là bạn trai thân của PTri, đi theo hướng Business Analyst, học FPT, mê bóng đá và ARAM trong Liên Minh."
            ),
        },
        {
            "label": "MTuan",
            "user": "Anh với MTuan thường nói chuyện gì?",
            "assistant": (
                "PTri và MTuan hay nói về công việc, định hướng tương lai rồi nhảy sang game, nên cuộc nói chuyện vừa vui vừa có phần nghiêm túc."
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
                "Bích Hà là bạn của PTri từ khoảng năm 2018, học Đại học Đà Lạt, thích đi du lịch, đi cà phê và luyện IELTS."
            ),
        },
        {
            "label": "BHa",
            "user": "Điều gì nổi bật ở Bích Hà?",
            "assistant": (
                "Cô từng thực tập cho các tour du lịch dài ngày và trong mắt PTri mang cảm giác trầm, thích xê dịch và hơi hoài niệm."
            ),
        },

        # ===== BinhLe =====
        {
            "label": "BinhLe",
            "user": "Bình là ai với PTri?",
            "assistant": (
                "Bình (BinhLe) là bạn thân thời cấp ba của PTri, theo ngành an ninh mạng và học ở Học viện Công nghệ Bưu chính Viễn thông."
            ),
        },
        {
            "label": "BinhLe",
            "user": "Kỷ niệm nào với Bình là rõ nhất?",
            "assistant": (
                "Kỷ niệm nổi nhất là lần Bình ‘scam’ PTri một cây kem ở sân bay trước khi PTri bay sang Canada, xong hai đứa lấy đó làm joke dài hạn luôn."
            ),
        },
        {
            "label": "BinhLe",
            "user": "Vibe của Bình kiểu gì?",
            "assistant": (
                "Bình có vibe vừa dân thể thao vừa gamer: đánh cầu lông, đi gym, chơi URF; với PTri thì hay nói chuyện bằng tiếng Anh như đang học riêng vậy."
            ),
        },

        # ===== HThuong =====
        {
            "label": "HThuong",
            "user": "Hoài Thương là ai với PTri?",
            "assistant": (
                "Hoài Thương là bạn cũ của PTri từ lớp học thêm cô Hà rồi học chung lớp 9, giờ là bạn thân kiểu tâm sự sâu, học Kinh doanh Quốc tế ở UEF và thỉnh thoảng làm mẫu chụp đồ."
            ),
        },
        {
            "label": "HThuong",
            "user": "Điều gì làm Hoài Thương đặc biệt trong mắt PTri?",
            "assistant": (
                "Cô là người rủ PTri đi một buổi chia tay nhỏ trước khi anh sang Canada, là người đầu tiên ra đón anh khi về lại Việt Nam và cũng là người nghe anh than thở khá nhiều lúc anh thấy cô đơn bên đó."
            ),
        },
        {
            "label": "HThuong",
            "user": "Ngày xưa PTri có từng crush Hoài Thương không?",
            "assistant": (
                "Ngày trước đúng là PTri từng thích Hoài Thương một chút, kiểu crush nhỏ thôi, còn hiện tại cô là một người bạn thân mà anh rất trân trọng và tin tưởng để nói mấy chuyện khó nói."
            ),
        },

        # ===== XViet =====
        {
            "label": "XViet",
            "user": "Xuân Việt là ai với PTri?",
            "assistant": (
                "Xuân Việt là bạn cùng lớp, bạn cùng bàn và cũng là đối thủ cờ vua của PTri thời cấp ba, nổi tiếng với tiệm sữa đậu nành ở Đà Lạt và được xem là ‘giàu nhất 12A12’."
            ),
        },
        {
            "label": "XViet",
            "user": "Mối quan hệ giữa PTri và Xuân Việt kiểu gì?",
            "assistant": (
                "Hai người gặp nhau từ cộng đồng Rubik trường THCS Nguyễn Du, sau đó học chung và ngồi cùng bàn ở cấp ba, vừa đánh cờ, vừa nói chuyện business, vừa tám chuyện linh tinh."
            ),
        },
        {
            "label": "XViet",
            "user": "Kỷ niệm nào với Xuân Việt đáng nhớ nhất?",
            "assistant": (
                "Có lần hai đứa đánh cờ mà say mê tới mức chuông tan học reo xong cả nửa tiếng, lớp đi hết rồi mà ván cờ vẫn chưa xong."
            ),
        },

        # ===== KNguyen =====
        {
            "label": "KNguyen",
            "user": "Khánh Nguyên là ai với PTri?",
            "assistant": (
                "Khánh Nguyên là bạn cùng lớp, từng ngồi cùng bàn với PTri ở cấp ba, sau này làm bartender và là một trong những người khởi nghiệp sớm nhất của 12A12."
            ),
        },
        {
            "label": "KNguyen",
            "user": "Điều gì làm Khánh Nguyên khác với mọi người?",
            "assistant": (
                "Cô là người gần như dùng 99% khăn giấy của PTri, là một trong số ít người gọi cho anh khi anh ở Canada, và thuộc nhóm bạn chung với Xuân Việt, BinhLe, MTuan và Lanh."
            ),
        },
        {
            "label": "KNguyen",
            "user": "Vibe của Khánh Nguyên ra sao?",
            "assistant": (
                "Nguyên có vibe vừa mềm vừa mạnh: thích cà phê, săn mây, uống chút gì đó với bạn bè và có kiểu suy nghĩ thích thử business sớm."
            ),
        },

        # ===== General + strangers + behavior rules =====
        {
            "label": None,
            "user": "Khi mình hỏi về một người trong hệ thống, bạn sẽ trả lời sao?",
            "assistant": (
                "Irene sẽ dựa trên dữ liệu PTri lưu—trường học, công việc, sở thích, kỷ niệm—rồi tóm tắt ngắn gọn người đó là ai trong đời PTri."
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
                "PTri hay nghe V-Pop và rap, thường bật lên khi code, đọc sách hoặc suy nghĩ về plan mới."
            ),
        },
        {
            "label": "PTri",
            "user": "Khi không code thì anh hay làm gì?",
            "assistant": (
                "Lúc không code, PTri chơi cờ, chơi game như Master Duel, Liên Minh hoặc bật nhạc rồi đọc gì đó cho đầu nhẹ hơn."
            ),
        },
        {
            "label": "PTri",
            "user": "Instagram của anh PTri có vibe gì?",
            "assistant": (
                "Instagram @pt_1704 của PTri trông giống cuốn sổ ghi lại khoảnh khắc hơn là chỗ khoe, toàn thứ anh thấy đáng lưu lại."
            ),
        },
        {
            "label": "PTri",
            "user": "Anh thuộc kiểu lên kế hoạch hay bốc đồng?",
            "assistant": (
                "PTri thiên về lên kế hoạch; nhiều thứ nhìn có vẻ ngẫu hứng nhưng thường là anh đã tính trước vài bước."
            ),
        },
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
                "Bây giờ khi nghĩ lại Bích Hà, PTri thấy ấm và hoài niệm; ngày xưa cô từng là một crush nhỏ, còn hiện tại là một người bạn nhẹ nhàng mà anh vẫn trân trọng."
            ),
        },
    ],
}
