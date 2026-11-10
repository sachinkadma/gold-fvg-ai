import os
import json
import zoneinfo
from datetime import datetime, time
import numpy as np

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

class ICTKnowledgeStore:
    """
    Structured ICT Knowledge Store with metadata and temporal killzone filtering.
    Replaces legacy flat text notes with schema-validated JSON topics.
    """
    def __init__(self, knowledge_dir="knowledge"):
        self.knowledge_dir = knowledge_dir
        self.categories = ["liquidity", "structure", "entry_models", "risk", "elliott_wave"]
        self.topics = []
        self.topic_map = {}
        self._load_knowledge()

    def _load_knowledge(self):
        for cat in self.categories:
            fpath = os.path.join(self.knowledge_dir, f"{cat}.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for t in data.get("topics", []):
                            t["category"] = cat
                            self.topics.append(t)
                            self.topic_map[t["id"]] = t
                except Exception as e:
                    print(f"Warning: Failed to load {fpath}: {e}")

    @staticmethod
    def get_killzone_window(dt_ny=None):
        """
        Determines current New York Killzone and checks if it passes active liquidity window.
        Returns: is_pass (bool), active_kz_name (str), code ("KILLZONE_PASS" | "NO_KILLZONE_PASS"), details (str)
        """
        if dt_ny is None:
            dt_ny = datetime.now(NY_TZ)
        t = dt_ny.time()

        # London Killzone: 02:00 - 05:00 EST
        if time(2, 0) <= t < time(5, 0):
            return True, "London", "KILLZONE_PASS", "London Killzone Active (High Volatility Judas Swing Window)"
        # NY AM Killzone: 08:30 - 11:00 EST
        elif time(8, 30) <= t < time(11, 0):
            if time(10, 0) <= t < time(11, 0):
                return True, "Silver_Bullet", "KILLZONE_PASS", "NY AM Silver Bullet Window Active (10:00 - 11:00 EST)"
            return True, "NY_AM", "KILLZONE_PASS", "NY AM Killzone Active (Peak Gold Session Volatility)"
        # NY Lunch: 12:00 - 13:00 EST
        elif time(12, 0) <= t < time(13, 0):
            return False, "NY_Lunch", "NO_KILLZONE_PASS", "NY Lunch Window (Low Volume Consolidation & Whip Chop - Pass Avoided)"
        # NY PM Killzone: 13:30 - 16:00 EST
        elif time(13, 30) <= t < time(16, 0):
            if time(14, 0) <= t < time(15, 0):
                return True, "Silver_Bullet", "KILLZONE_PASS", "NY PM Silver Bullet Window Active (14:00 - 15:00 EST)"
            return True, "NY_PM", "KILLZONE_PASS", "NY PM Killzone Active (Afternoon Expansion)"
        # Asian Range: 20:00 - 00:00 EST
        elif time(20, 0) <= t or t < time(0, 0):
            return False, "Asian_Range", "NO_KILLZONE_PASS", "Asian Range Accumulation (Setup Formation Phase - Pass Avoided for Scalps)"
        else:
            return False, "Off_Hours", "NO_KILLZONE_PASS", "Off-Hours Trading (Interbank Volume Inactive - Pass Avoided)"

    def retrieve(self, query=None, active_kz=None, category=None, top_k=4):
        """
        Metadata-filtered hybrid retrieval with temporal filtering.
        """
        filtered = self.topics
        
        # Category filter
        if category:
            filtered = [t for t in filtered if t.get("category") == category]
            
        # Killzone metadata filter
        if active_kz and active_kz in ["London", "NY_AM", "Silver_Bullet", "NY_PM"]:
            filtered = [t for t in filtered if active_kz in t.get("killzones", [])]

        if not query:
            return filtered[:top_k]

        # Simple high-speed keyword & token overlap ranking
        q_tokens = set(query.lower().replace("-", " ").replace("/", " ").split())
        scored = []
        for t in filtered:
            text = f"{t.get('name', '')} {json.dumps(t.get('deterministic_criteria', {}))} {' '.join(t.get('rules', []))}".lower()
            score = sum(1 for token in q_tokens if token in text)
            if score > 0:
                scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [t for _, t in scored[:top_k]]
        return results if results else filtered[:top_k]

    def get_topic_by_id(self, topic_id):
        return self.topic_map.get(topic_id)

    def get_all_topics_summary(self):
        return {t["id"]: t["name"] for t in self.topics}

# Module-level convenience aliases
get_killzone_window = ICTKnowledgeStore.get_killzone_window
