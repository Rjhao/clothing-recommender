"""Sample data generator for clothing knowledge graph."""
import random
from datetime import datetime, timedelta
from typing import List


class SampleDataGenerator:
    """Generate sample data for testing the recommendation system."""

    # Brand names
    BRANDS = [
        "ZARA", "H&M", "UNIQLO", "Nike", "Adidas", "Gucci", "Prada", "Louis Vuitton",
        "Zegna", "Bally", "Calvin Klein", "Tommy Hilfiger", "Ralph Lauren",
        "Massimo Dutti", "COS", "& Other Stories", "Arket", "Acne Studios"
    ]

    # Colors
    COLORS = [
        "黑色", "白色", "灰色", "海军蓝", "藏青", "酒红", "墨绿", "卡其",
        "米白", "驼色", "粉色", "玫瑰金", "香槟色", "焦糖色", "天蓝"
    ]

    # Materials
    MATERIALS = [
        "棉", "麻", "丝绸", "羊毛", "羊绒", "牛仔布", "皮革", "蕾丝",
        "涤纶", "腈纶", "氨纶", "府绸", "法兰绒", "灯芯绒"
    ]

    # Price ranges by category (min, max)
    PRICE_RANGES = {
        "top": (99, 2999),
        "bottom": (149, 1999),
        "dress": (199, 4999),
        "outerwear": (299, 5999),
        "shoes": (299, 3999),
        "accessories": (49, 1999)
    }

    # Sample clothing items
    SAMPLE_CLOTHING = {
        "top": [
            {"name": "简约纯棉T恤", "style": "minimalist", "season": "summer"},
            {"name": "条纹Polo衫", "style": "casual", "season": "summer"},
            {"name": "宽松亚麻衬衫", "style": "bohemian", "season": "spring"},
            {"name": "复古印花衬衫", "style": "vintage", "season": "spring"},
            {"name": "街头印花T恤", "style": "streetwear", "season": "summer"},
            {"name": "商务免烫衬衫", "style": "business", "season": "all_season"},
            {"name": "运动功能性T恤", "style": "sporty", "season": "summer"},
            {"name": "蕾丝透视上衣", "style": "romantic", "season": "spring"},
        ],
        "bottom": [
            {"name": "直筒休闲裤", "style": "minimalist", "season": "all_season"},
            {"name": "牛仔直筒裤", "style": "casual", "season": "all_season"},
            {"name": "波西米亚长裙", "style": "bohemian", "season": "summer"},
            {"name": "高腰阔腿裤", "style": "vintage", "season": "spring"},
            {"name": "工装休闲裤", "style": "streetwear", "season": "fall"},
            {"name": "西装正装裤", "style": "business", "season": "all_season"},
            {"name": "运动短裤", "style": "sporty", "season": "summer"},
            {"name": "雪纺碎花裙", "style": "romantic", "season": "summer"},
        ],
        "dress": [
            {"name": "小黑裙", "style": "minimalist", "season": "all_season"},
            {"name": "休闲连衣裙", "style": "casual", "season": "summer"},
            {"name": "波西米亚长裙", "style": "bohemian", "season": "summer"},
            {"name": "复古方领裙", "style": "vintage", "season": "spring"},
            {"name": "街头风卫衣裙", "style": "streetwear", "season": "fall"},
            {"name": "商务连衣裙", "style": "business", "season": "all_season"},
            {"name": "运动休闲裙", "style": "sporty", "season": "summer"},
            {"name": "浪漫蕾丝裙", "style": "romantic", "season": "spring"},
        ],
        "outerwear": [
            {"name": "极简风衣", "style": "minimalist", "season": "spring"},
            {"name": "牛仔外套", "style": "casual", "season": "fall"},
            {"name": "流苏开衫", "style": "bohemian", "season": "spring"},
            {"name": "复古西装外套", "style": "vintage", "season": "fall"},
            {"name": "工装夹克", "style": "streetwear", "season": "fall"},
            {"name": "商务西装", "style": "business", "season": "fall"},
            {"name": "运动外套", "style": "sporty", "season": "spring"},
            {"name": "粗花呢外套", "style": "romantic", "season": "fall"},
        ],
        "shoes": [
            {"name": "极简小白鞋", "style": "minimalist", "season": "all_season"},
            {"name": "帆布休闲鞋", "style": "casual", "season": "summer"},
            {"name": "流苏乐福鞋", "style": "bohemian", "season": "spring"},
            {"name": "复古玛丽珍", "style": "vintage", "season": "spring"},
            {"name": "高帮运动鞋", "style": "streetwear", "season": "all_season"},
            {"name": "商务正装鞋", "style": "business", "season": "all_season"},
            {"name": "跑步运动鞋", "style": "sporty", "season": "all_season"},
            {"name": "绑带高跟鞋", "style": "romantic", "season": "spring"},
        ],
        "accessories": [
            {"name": "极简皮带", "style": "minimalist", "season": "all_season"},
            {"name": "帆布腰带", "style": "casual", "season": "all_season"},
            {"name": "波西米亚项链", "style": "bohemian", "season": "all_season"},
            {"name": "复古手表", "style": "vintage", "season": "all_season"},
            {"name": "街头棒球帽", "style": "streetwear", "season": "all_season"},
            {"name": "商务领带", "style": "business", "season": "all_season"},
            {"name": "运动手表", "style": "sporty", "season": "all_season"},
            {"name": "珍珠耳环", "style": "romantic", "season": "all_season"},
        ]
    }

    # Occasions mapping
    OCCASIONS_BY_STYLE = {
        "minimalist": ["daily", "work", "formal"],
        "casual": ["daily", "casual", "party"],
        "bohemian": ["casual", "party", "daily"],
        "vintage": ["casual", "party", "formal"],
        "streetwear": ["casual", "party", "daily"],
        "business": ["work", "formal"],
        "sporty": ["sports", "casual", "daily"],
        "romantic": ["party", "casual", "daily"]
    }

    @classmethod
    def generate_clothing_data(cls, num_items: int = 100) -> List[dict]:
        """Generate sample clothing data."""
        clothing_items = []
        clothing_id = 1

        for category, templates in cls.SAMPLE_CLOTHING.items():
            # Calculate how many items per category
            num_per_category = num_items // len(cls.SAMPLE_CLOTHING)
            price_range = cls.PRICE_RANGES[category]

            for i in range(num_per_category):
                template = random.choice(templates)
                item = {
                    "clothing_id": clothing_id,
                    "name": template["name"],
                    "category": category,
                    "brand": random.choice(cls.BRANDS),
                    "color": random.choice(cls.COLORS),
                    "material": random.choice(cls.MATERIALS),
                    "style": template["style"],
                    "season": template["season"],
                    "occasion": random.choice(cls.OCCASIONS_BY_STYLE.get(template["style"], ["daily"])),
                    "price": round(random.uniform(*price_range), 2)
                }
                clothing_items.append(item)
                clothing_id += 1

        return clothing_items

    @classmethod
    def generate_user_data(cls, num_users: int = 50) -> List[dict]:
        """Generate sample user data."""
        import names

        users = []
        user_id = 1
        genders = ["male", "female", "other"]
        body_shapes = ["slim", "athletic", "average", "curvy", "plus_size"]
        skin_tones = ["fair", "light", "medium", "tan", "dark"]

        for _ in range(num_users):
            gender = random.choice(genders)
            user = {
                "user_id": user_id,
                "username": names.get_first_name(gender=gender if gender != "other" else None),
                "gender": gender,
                "age": random.randint(18, 65),
                "height": round(random.uniform(155, 190), 1),
                "weight": round(random.uniform(45, 95), 1),
                "body_shape": random.choice(body_shapes),
                "skin_tone": random.choice(skin_tones)
            }
            users.append(user)
            user_id += 1

        return users

    @classmethod
    def generate_interactions(
        cls,
        users: List[dict],
        clothing: List[dict],
        min_interactions: int = 5,
        max_interactions: int = 20
    ) -> List[dict]:
        """Generate sample user-clothing interactions."""
        interactions = []
        interaction_id = 1

        for user in users:
            num_interactions = random.randint(min_interactions, max_interactions)
            interacted_items = random.sample(
                clothing,
                min(num_interactions, len(clothing))
            )

            for item in interacted_items:
                interaction_type = random.choices(
                    ["view", "buy", "rate"],
                    weights=[0.6, 0.2, 0.2]
                )[0]

                days_ago = random.randint(0, 90)
                timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()

                interaction = {
                    "interaction_id": interaction_id,
                    "user_id": user["user_id"],
                    "clothing_id": item["clothing_id"],
                    "type": interaction_type,
                    "timestamp": timestamp,
                    "duration": random.randint(5, 300) if interaction_type == "view" else 0,
                    "score": round(random.uniform(3, 5), 1) if interaction_type == "rate" else None
                }
                interactions.append(interaction)
                interaction_id += 1

        return interactions


if __name__ == "__main__":
    # Generate sample data
    clothing = SampleDataGenerator.generate_clothing_data(100)
    users = SampleDataGenerator.generate_user_data(50)
    interactions = SampleDataGenerator.generate_interactions(users, clothing)

    print(f"Generated {len(clothing)} clothing items")
    print(f"Generated {len(users)} users")
    print(f"Generated {len(interactions)} interactions")

    # Sample output
    print("\nSample clothing:")
    for item in clothing[:3]:
        print(f"  {item}")

    print("\nSample user:")
    print(f"  {users[0]}")

    print("\nSample interactions:")
    for interaction in interactions[:3]:
        print(f"  {interaction}")