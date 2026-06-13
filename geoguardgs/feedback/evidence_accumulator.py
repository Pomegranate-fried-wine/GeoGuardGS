"""Small helper for accumulating group evidence across triggers."""

from collections import defaultdict


class EvidenceAccumulator:
    def __init__(self):
        self.records = []

    def add_many(self, records):
        self.records.extend(records or [])

    def summarize_by_gaussian(self):
        table = defaultdict(lambda: {"count": 0, "max_confidence": 0.0, "actions": set()})
        for row in self.records:
            for gid in row.get("stable_gaussian_ids", []):
                item = table[int(gid)]
                item["count"] += 1
                item["max_confidence"] = max(item["max_confidence"], float(row.get("confidence", 0.0) or 0.0))
                item["actions"].add(str(row.get("future_action", "skip")))
        return {
            gid: {"count": v["count"], "max_confidence": v["max_confidence"], "actions": sorted(v["actions"])}
            for gid, v in table.items()
        }
