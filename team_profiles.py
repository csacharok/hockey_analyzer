"""Auditable game-relative light/dark uniform profiles."""

from __future__ import annotations

import math

HOME = "HOME"
AWAY = "AWAY"
UNKNOWN = "UNKNOWN"


class GameTeamProfiles:
    def __init__(self, configuration):
        if not isinstance(configuration, dict):
            raise ValueError("Game team profiles must be a JSON object.")
        clusters = configuration.get("clusters")
        if not isinstance(clusters, list) or len(clusters) != 2:
            raise ValueError("Game team profiles require exactly two clusters.")
        override = configuration.get("team_mapping", {})
        self.ambiguity_margin = float(configuration.get("ambiguity_margin", .5))
        self.profiles = []
        for cluster in clusters:
            center = cluster.get("lightness_center")
            scale = cluster.get("lightness_scale")
            maximum = cluster.get("maximum_distance")
            if (not isinstance(center, list) or not isinstance(scale, list)
                    or len(center) != 3 or len(scale) != 3
                    or any(not _finite(value) for value in center + scale)
                    or any(value <= 0 for value in scale)
                    or not _finite(maximum) or maximum <= 0):
                raise ValueError("Each game team profile requires three-value lightness center/scale and positive maximum_distance.")
            cluster_id = str(cluster.get("cluster"))
            team = override.get(cluster_id, cluster.get("default_team"))
            if team not in (HOME, AWAY):
                raise ValueError("Each profile must map to HOME or AWAY.")
            self.profiles.append(dict(cluster=cluster_id, team=team, center=center,
                                      scale=scale, maximum=float(maximum)))
        if {profile["team"] for profile in self.profiles} != {HOME, AWAY}:
            raise ValueError("Game profile mapping must assign one HOME and one AWAY cluster.")

    def classify(self, feature):
        if not isinstance(feature, (list, tuple)) or len(feature) != 3:
            return UNKNOWN, 0.0, {"reason": "invalid_uniform_feature"}
        distances = []
        for profile in self.profiles:
            distance = math.sqrt(sum(((value - center) / scale) ** 2
                                     for value, center, scale in zip(
                                         feature, profile["center"], profile["scale"])))
            distances.append((distance, profile))
        distances.sort(key=lambda item: item[0])
        best_distance, best = distances[0]
        second_distance = distances[1][0]
        evidence = {"uniform_feature": list(feature),
                    "profile_distances": {profile["team"]: distance
                                          for distance, profile in distances},
                    "profile_clusters": {profile["team"]: profile["cluster"]
                                         for _, profile in distances}}
        if (best_distance > best["maximum"]
                or second_distance - best_distance < self.ambiguity_margin):
            return UNKNOWN, 0.0, evidence
        confidence = min(1.0, max(0.0, 1 - best_distance / best["maximum"]))
        return best["team"], confidence, evidence


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))
