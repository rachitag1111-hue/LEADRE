"""
Dataset Generation and Preparation for LEADRE.
Generates realistic display advertising catalogs, user profiles, cross-domain interaction
sequences, and prepares curriculum datasets for Stages 1-4.
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from .schemas import AdItem, UserProfile, InteractionEvent, UserHistory
from .prompt_builder import PromptBuilder


CATEGORIES = {
    "Automobile": [
        ("SAIC Volkswagen New Energy ID.3", "Automobile Products", "SAIC Volkswagen New Energy", {"brand": "Volkswagen", "series": "ID.3"}),
        ("BYD Seal EV", "Automobile Products", "BYD New Energy", {"brand": "BYD", "series": "Seal"}),
        ("Tesla Model Y Performance", "Automobile Products", "Tesla", {"brand": "Tesla", "series": "Model Y"}),
        ("NIO ES6 SUV", "Automobile Products", "NIO Smart EV", {"brand": "NIO", "series": "ES6"}),
    ],
    "Consumer Electronics": [
        ("Apple iPhone 16 Pro Max", "Smartphones", "Mobile Phones", {"brand": "Apple", "chip": "A18 Pro", "storage": "256GB"}),
        ("Sony WH-1000XM5 Wireless Headphones", "Audio Equipment", "Noise Cancelling", {"brand": "Sony", "battery": "30h"}),
        ("DJI Mini 4 Pro Drone", "Smart Photography", "UAV", {"brand": "DJI", "camera": "4K HDR"}),
        ("Dell XPS 16 Laptop", "Computers", "Workstation Laptops", {"brand": "Dell", "cpu": "Core Ultra 9"}),
    ],
    "Education": [
        ("Python AI & Machine Learning Masterclass", "Online Courses", "Professional Skills", {"field": "Data Science", "level": "Advanced"}),
        ("IELTS 7.5 Speaking Coaching Camp", "Language Training", "Study Abroad", {"target": "IELTS", "mode": "1-on-1"}),
        ("Children STEM Robotics Bootcamp", "K12 Education", "Programming", {"age_group": "8-14", "kit": "Included"}),
    ],
    "Travel & Hospitality": [
        ("Bali 7-Day Island Luxury Tour", "Vacation Packages", "International Travel", {"dest": "Bali", "hotel": "5-Star"}),
        ("Kyoto Autumn Leaves Scenic Express", "Vacation Packages", "Cultural Tours", {"dest": "Japan", "duration": "5 Days"}),
        ("Swiss Alps Skiing Holiday", "Winter Sports", "Adventure Travel", {"dest": "Switzerland", "season": "Winter"}),
    ],
    "Fashion & Cosmetics": [
        ("Estee Lauder Advanced Night Repair Serum", "Skincare", "Anti-Aging", {"brand": "Estee Lauder", "volume": "50ml"}),
        ("Nike Air Zoom Alphafly Next% 3", "Footwear", "Marathon Running Shoes", {"brand": "Nike", "cushion": "ZoomX"}),
        ("Dior Sauvage Eau de Parfum", "Fragrances", "Men's Perfume", {"brand": "Dior", "type": "EDP"}),
    ],
    "Finance": [
        ("High-Yield Wealth Growth Plan", "Financial Services", "Investment Funds", {"risk": "Medium", "term": "12 Months"}),
        ("Flexible Personal Micro-Credit", "Consumer Finance", "Credit Loans", {"limit": "200000", "apr": "4.35%"}),
    ]
}

CONTENT_TOPICS = [
    ("play short video", "reality show comedy"),
    ("play short video", "shuttlecock freestyle trick"),
    ("play short video", "Emotional test & psychology"),
    ("play short video", "Electric vehicle acceleration review"),
    ("play short video", "Autonomous driving safety test"),
    ("play short video", "Smartphone camera low-light comparison"),
    ("play short video", "Quick home cooking pasta recipe"),
    ("search", "best mirrorless camera for vloggers"),
    ("search", "electric suv winter range ratings"),
    ("search", "beginner machine learning roadmap"),
]

CITIES = ["Haidian, Beijing", "Chaoyang, Beijing", "Pudong, Shanghai", "Nanshan, Shenzhen", "Tianhe, Guangzhou", "Chengdu", "Hangzhou"]
OCCUPATIONS = ["Internet industry", "Finance & Banking", "Education & Research", "Healthcare", "Engineering", "Marketing", "Student"]
CONSUMPTION_LEVELS = ["low", "medium", "high", "very high"]


class DataGenerator:
    """
    Generates synthetic catalogs, simulated users, histories, and training datasets.
    """
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.prompt_builder = PromptBuilder(seed=seed)

    def generate_ad_catalog(self) -> List[AdItem]:
        """Creates a realistic display ad catalog."""
        ads = []
        ad_idx = 1
        for cat_name, items in CATEGORIES.items():
            for name, p_type, sec_cat, attrs in items:
                # ECPM between 1.0 and 25.0
                base_ecpm = round(random.uniform(2.0, 18.0), 2)
                ad = AdItem(
                    ad_id=f"ad_{ad_idx:04d}",
                    name=name,
                    product_type=p_type,
                    first_cat=cat_name,
                    second_cate=sec_cat,
                    attributes=attrs,
                    ecpm=base_ecpm
                )
                ads.append(ad)
                ad_idx += 1
        return ads

    def generate_user_profile(self, user_id: str) -> UserProfile:
        age = random.choice([21, 23, 25, 28, 32, 36, 42, 48])
        sex = random.choice(["male", "female"])
        residence = random.choice(CITIES)
        education = random.choice(["bachelor's", "master's", "doctoral", "associate"])
        occupation = random.choice(OCCUPATIONS)
        cons = random.choice(CONSUMPTION_LEVELS)
        arpu = round(random.uniform(5.0, 300.0), 2)
        return UserProfile(
            user_id=user_id,
            age=age,
            sex=sex,
            residence=residence,
            education_level=education,
            occupation=occupation,
            consumption_level=cons,
            arpu=arpu
        )

    def generate_user_history(
        self,
        user_id: str,
        catalog: List[AdItem],
        num_ad_events: int = 5,
        num_content_events: int = 7
    ) -> UserHistory:
        profile = self.generate_user_profile(user_id)
        
        # User preference bias toward 2 preferred categories
        preferred_cats = random.sample(list(CATEGORIES.keys()), k=2)
        
        events: List[InteractionEvent] = []
        cat_counts: Dict[str, int] = {}
        for cat in CATEGORIES.keys():
            base = random.randint(1, 10)
            if cat in preferred_cats:
                base += random.randint(20, 100)
            cat_counts[cat] = base

        # Generate content events
        for _ in range(num_content_events):
            days = random.randint(5, 90)
            action, title = random.choice(CONTENT_TOPICS)
            events.append(InteractionEvent(
                days_ago=days,
                domain="content",
                behavior_type=action,
                title=title
            ))

        # Filter candidate ads biased to preferred categories
        biased_ads = [a for a in catalog if a.first_cat in preferred_cats]
        if not biased_ads:
            biased_ads = catalog

        # Generate ad interactions
        for _ in range(num_ad_events):
            days = random.randint(1, 90)
            chosen_ad = random.choice(biased_ads)
            action = random.choice(["click on ad", "Conversion ad", "click on ad"])
            events.append(InteractionEvent(
                days_ago=days,
                domain="ad",
                behavior_type=action,
                title=chosen_ad.name,
                s_ids=chosen_ad.s_ids,
                ad_id=chosen_ad.ad_id
            ))

        # Sort chronologically (oldest to most recent: descending days_ago)
        events.sort(key=lambda x: x.days_ago, reverse=True)

        return UserHistory(
            user_id=user_id,
            profile=profile,
            category_counts=cat_counts,
            events=events
        )

    def create_stage1_dataset(self, catalog: List[AdItem]) -> List[Dict[str, str]]:
        """Stage 1: Explicit Alignment (Ad Description -> S-ID)"""
        dataset = []
        for ad in catalog:
            prompt, response = self.prompt_builder.build_explicit_alignment_pair(ad)
            dataset.append({"prompt": prompt, "response": response, "task": "stage1_explicit"})
        return dataset

    def create_stage2_dataset(self, histories: List[UserHistory]) -> List[Dict[str, str]]:
        """Stage 2: Implicit Alignment (History with text descriptions -> Next Ad S-ID)"""
        dataset = []
        for h in histories:
            slices = self.prompt_builder.slice_positive_ad_interactions(h.events)
            for prior_events, target_ad_event in slices:
                prompt = self.prompt_builder.build_implicit_alignment_prompt(h, prior_events)
                target_sid_str = f"<{', '.join(target_ad_event.s_ids)}>" if target_ad_event.s_ids else ""
                dataset.append({
                    "prompt": prompt,
                    "response": target_sid_str,
                    "task": "stage2_implicit"
                })
        return dataset

    def create_stage3_dataset(self, histories: List[UserHistory]) -> List[Dict[str, str]]:
        """Stage 3: Main Task (Full Prompt with S-IDs -> Next Ad S-ID)"""
        dataset = []
        for h in histories:
            slices = self.prompt_builder.slice_positive_ad_interactions(h.events)
            for prior_events, target_ad_event in slices:
                # Augmentation 1: standard
                prompt = self.prompt_builder.build_main_prompt(h, prior_events, shuffle_profile=False)
                target_sid_str = f"<{', '.join(target_ad_event.s_ids)}>" if target_ad_event.s_ids else ""
                dataset.append({
                    "prompt": prompt,
                    "response": target_sid_str,
                    "task": "stage3_main"
                })
                # Augmentation 2: profile reordered
                prompt_shuffled = self.prompt_builder.build_main_prompt(h, prior_events, shuffle_profile=True)
                dataset.append({
                    "prompt": prompt_shuffled,
                    "response": target_sid_str,
                    "task": "stage3_main_augmented"
                })
        return dataset

    def create_stage4_dpo_dataset(
        self,
        histories: List[UserHistory],
        catalog_by_id: Dict[str, AdItem]
    ) -> List[Dict[str, str]]:
        """
        Stage 4: DPO Business Alignment.
        Pairs chosen (high ECPM) vs rejected (low ECPM) ads for identical context prompt.
        """
        dataset = []
        ads_list = list(catalog_by_id.values())

        for h in histories:
            slices = self.prompt_builder.slice_positive_ad_interactions(h.events)
            for prior_events, target_ad_event in slices:
                if not target_ad_event.ad_id or target_ad_event.ad_id not in catalog_by_id:
                    continue
                ad1 = catalog_by_id[target_ad_event.ad_id]
                
                # Sample another candidate from same or similar category
                category_ads = [a for a in ads_list if a.first_cat == ad1.first_cat and a.ad_id != ad1.ad_id]
                if not category_ads:
                    category_ads = [a for a in ads_list if a.ad_id != ad1.ad_id]
                ad2 = random.choice(category_ads)

                if ad1.ecpm >= ad2.ecpm:
                    chosen_ad, rejected_ad = ad1, ad2
                else:
                    chosen_ad, rejected_ad = ad2, ad1

                prompt = self.prompt_builder.build_main_prompt(h, prior_events)
                dataset.append({
                    "prompt": prompt,
                    "chosen": chosen_ad.s_id_string(),
                    "rejected": rejected_ad.s_id_string(),
                    "chosen_ecpm": chosen_ad.ecpm,
                    "rejected_ecpm": rejected_ad.ecpm,
                    "task": "stage4_dpo"
                })
        return dataset
