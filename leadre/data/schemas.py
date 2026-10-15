"""
Data Schemas for LEADRE (Display Advertisement Recommender System)
Defines models for Ads, User Profiles, Cross-Domain Interaction Events, and S-IDs.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class AdItem(BaseModel):
    """
    Representation of an advertisement item in the catalog.
    Matches Section 3.2 of the paper.
    """
    ad_id: str
    name: str
    product_type: str
    first_cat: str
    second_cate: str
    attributes: Dict[str, str] = Field(default_factory=dict)
    ecpm: float = 0.0  # Effective Cost Per Mille for DPO business alignment
    s_ids: Optional[List[str]] = None  # e.g., ["a_12", "b_6", "c_22", "d_36", "e_0"]

    def to_template_text(self) -> str:
        """
        Formats ad metadata according to Section 3.2:
        'The name of the ad is <ads_name>; The product type is <ads_type>;
        The first-level category is <first_cat>; The second-level category is <second_cate>;
        The attributes include: <basic_att>.'
        """
        attr_str = ", ".join(f"{k}_{v}" for k, v in self.attributes.items())
        return (
            f"The name of the ad is {self.name}; "
            f"The product type is {self.product_type}; "
            f"The first-level category is {self.first_cat}; "
            f"The second-level category is {self.second_cate}; "
            f"The attributes include: {attr_str}."
        )

    def s_id_string(self) -> str:
        """Returns the S-ID in bracketed format: <a_12, b_6, c_22, d_36, e_0>"""
        if not self.s_ids:
            return ""
        return f"<{', '.join(self.s_ids)}>"


class UserProfile(BaseModel):
    """
    User demographic profile.
    Matches Section 4.2.1 Component 2.
    """
    user_id: str
    age: int
    sex: str
    residence: str
    education_level: str
    occupation: str
    consumption_level: str
    arpu: float = 0.0  # Average Revenue Per User for nearline resource allocation

    def to_profile_text(self) -> str:
        return (
            f"{self.age}-year-old {self.sex}, resident in {self.residence}, "
            f"with a {self.education_level} degree, working in {self.occupation}, "
            f"with a {self.consumption_level} consumption level."
        )


class InteractionEvent(BaseModel):
    """
    Positive user interaction event in Ad or Content domain.
    Matches Section 4.2.1 Components 4 and 5.
    """
    days_ago: int
    domain: str  # 'ad' or 'content'
    behavior_type: str  # e.g. 'click on ad', 'Conversion ad', 'play short video', 'search'
    title: str  # Title of content or ad description
    s_ids: Optional[List[str]] = None  # S-IDs if ad interaction
    ad_id: Optional[str] = None

    def to_sequence_entry(self, use_sids_for_ads: bool = True) -> str:
        """
        Formats single event according to paper specification:
        '31 days ago^click on ad^<a_51, b_10, c_67, d_93, e_0>' (if use_sids_for_ads is True)
        or '31 days ago^click on ad^Bali 7-Day Tour' (for Implicit Alignment Stage 2).
        """
        if self.domain == "ad" and use_sids_for_ads and self.s_ids:
            sid_str = f"<{', '.join(self.s_ids)}>"
            return f"{self.days_ago} days ago^{self.behavior_type}^{sid_str}"
        return f"{self.days_ago} days ago^{self.behavior_type}^{self.title}"


class UserHistory(BaseModel):
    """
    Full chronological user history across Ad and Content domains.
    """
    user_id: str
    profile: UserProfile
    category_counts: Dict[str, int] = Field(default_factory=dict)
    events: List[InteractionEvent] = Field(default_factory=list)

    def to_interest_summary_text(self, top_n: int = 5) -> str:
        """
        Formats Section 4.2.1 Component 3:
        'The categories that have been frequently interacted recently are (format: category^interaction times): ...'
        """
        sorted_cats = sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        items_str = "; ".join(f"{cat}^{count} times" for cat, count in sorted_cats)
        return (
            f"The categories that have been frequently interacted recently are "
            f"(format: category^interaction times): {items_str};"
        )
