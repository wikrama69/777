"""
Battle-royale optimized strategy engine for Molty Royale.
Focus: kills first, HP second, while respecting EP, death zone, and facility value.
"""

import logging
from typing import Dict, Optional, Tuple, List

from .analyzer import StateAnalyzer
from config.settings import (
    WIN_PROBABILITY_ATTACK,
    PHASE_EARLY_PVP_BIAS,
    PHASE_MID_PVP_BIAS,
    PHASE_LATE_PVP_BIAS,
    ZONE_ESCAPE_PRIORITY,
    MEDICAL_FACILITY_HP_THRESHOLD,
    MIN_FREE_INVENTORY_SLOTS,
    MONSTER_FARM_SCORE_MIN,
)

logger = logging.getLogger("MoltyBot.Strategy")


class StrategyEngine:
    def __init__(self, analyzer: StateAnalyzer, memory, learning_engine):
        self.analyzer = analyzer
        self.memory = memory
        self.learning = learning_engine
        self.turn_number = 0
        self.explored_regions = set()
        self.last_region_id = None
        self.stuck_counter = 0
        self.known_dz_regions = set()
        self.dangerous_facilities = set()
        self.last_action_type = ""

    def decide(self, intel: Dict) -> Tuple[Dict, str, List[Dict]]:
        self.turn_number += 1
        weights = self.memory.action_weights
        attack_threshold = getattr(self.memory, "attack_threshold", WIN_PROBABILITY_ATTACK)
        phase = self.analyzer.detect_game_phase(intel)
        phase_bias = {
            "early": PHASE_EARLY_PVP_BIAS,
            "mid": PHASE_MID_PVP_BIAS,
            "late": PHASE_LATE_PVP_BIAS,
        }.get(phase, 0.0)
        effective_threshold = attack_threshold + phase_bias
        if hasattr(self.learning, "adaptive_threshold_for_phase"):
            effective_threshold = self.learning.adaptive_threshold_for_phase(phase, effective_threshold)
        if hasattr(self.learning, "adjust_aggressiveness"):
            effective_threshold -= self.learning.adjust_aggressiveness(phase)
        effective_threshold = max(0.45, min(0.90, effective_threshold))
        pvp_pve_bias = self.learning.pvp_pve_bias(phase) if hasattr(self.learning, "pvp_pve_bias") else {"pvp": 0.0, "pve": 0.0}
        zone_score = self.analyzer.zone_risk_score(intel)

        if intel["region_id"] == self.last_region_id:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_region_id = intel["region_id"]
        self.explored_regions.add(intel["region_id"])

        if intel.get("is_death_zone"):
            self.known_dz_regions.add(intel["region_id"])
        for dz_id in intel.get("pending_death_zones", []):
            self.known_dz_regions.add(dz_id)
        for rid, is_dz in intel.get("connections_status", {}).items():
            if is_dz:
                self.known_dz_regions.add(rid)

        free_actions = self._decide_free_actions(intel)

        # 1) Emergency death-zone escape
        if zone_score >= 1.0:
            target = self.analyzer.safest_escape_region(intel, self.known_dz_regions)
            if target:
                self.last_action_type = "move"
                return {"type": "move", "regionId": target}, f"Emergency zone escape -> {target[:8]}", free_actions

        # 2) Critical heal / med facility
        if intel["hp"] <= self.analyzer.hp_critical:
            heal_item = self._find_best_heal_item(intel["inventory"])
            if heal_item:
                self.last_action_type = "use_item"
                return {"type": "use_item", "itemId": heal_item["id"]}, f"Critical heal with {heal_item.get('typeId')}", free_actions
            med = self.analyzer.get_useful_facility(intel)
            if med and "medical" in (med.get("type") or "").lower():
                self.last_action_type = "interact"
                return {"type": "interact", "interactableId": med["id"]}, "Critical HP -> medical facility", free_actions
            escape = self.analyzer.safest_escape_region(intel, self.known_dz_regions)
            if escape and (intel["local_agents"] or intel["local_monsters"]):
                self.last_action_type = "move"
                return {"type": "move", "regionId": escape}, "Critical HP and enemies nearby -> evade", free_actions
            self.last_action_type = "rest"
            return {"type": "rest"}, "Critical HP -> rest", free_actions

        # 3) Pre-emptive death-zone move
        if zone_score >= 0.5:
            target = self.analyzer.safest_escape_region(intel, self.known_dz_regions)
            if target:
                self.last_action_type = "move"
                return {"type": "move", "regionId": target}, "Pre-emptive zone reposition", free_actions

        # 4) EP management
        if intel["ep"] < self.analyzer.ep_min_attack:
            if intel["local_agents"]:
                escape = self.analyzer.safest_escape_region(intel, self.known_dz_regions)
                if escape:
                    self.last_action_type = "move"
                    return {"type": "move", "regionId": escape}, "Low EP with enemy nearby -> evade", free_actions
            self.last_action_type = "rest"
            return {"type": "rest"}, "Recovering EP", free_actions

        # 5) Medical facility priority when low HP
        if intel["hp"] <= MEDICAL_FACILITY_HP_THRESHOLD:
            fac = self.analyzer.get_useful_facility(intel)
            if fac and "medical" in (fac.get("type") or "").lower():
                self.last_action_type = "interact"
                return {"type": "interact", "interactableId": fac["id"]}, "Low HP -> medical facility", free_actions

        # 6) PvP choice: kills matter more than HP, but only efficient kills
        if intel["local_agents"]:
            target, prob, reason = self._evaluate_combat_targets(
                intel, intel["local_agents"], effective_threshold, phase, zone_score, pvp_pve_bias.get("pvp", 0.0)
            )
            if target is not None:
                self.last_action_type = "attack"
                return {"type": "attack", "targetId": target["id"], "targetType": "agent"}, reason, free_actions
            if len(intel["local_agents"]) >= 2 or prob < max(0.52, effective_threshold - 0.05):
                escape = self.analyzer.safest_escape_region(intel, self.known_dz_regions)
                if escape:
                    self.last_action_type = "move"
                    return {"type": "move", "regionId": escape}, "Bad or crowded PvP -> reposition", free_actions

        # 7) Facilities (cache/watchtower especially early-mid)
        fac = self.analyzer.get_useful_facility(intel)
        if fac and weights.get("use_facility", 0.7) > 0.5 and intel["region_id"] not in self.dangerous_facilities:
            ftype = (fac.get("type") or "").lower()
            if "supply" in ftype or "watchtower" in ftype or ("medical" in ftype and intel["hp"] < 85):
                self.last_action_type = "interact"
                return {"type": "interact", "interactableId": fac["id"]}, f"Facility value -> {fac.get('type')}", free_actions

        # 8) PvE farming priority early / when safe
        if intel["local_monsters"]:
            target, prob, reason = self._evaluate_monster_targets(
                intel, intel["local_monsters"], phase, pvp_pve_bias.get("pve", 0.0)
            )
            if target is not None:
                self.last_action_type = "attack"
                return {"type": "attack", "targetId": target["id"], "targetType": "monster"}, reason, free_actions

        # 9) Rest if EP low-ish and no tactical edge
        if intel["ep"] <= self.analyzer.ep_rest_threshold and not intel["local_agents"]:
            self.last_action_type = "rest"
            return {"type": "rest"}, "Banking EP for next fight", free_actions

        # 10) Explore / move according to phase
        if phase == "early" and not intel["inventory_full"]:
            self.last_action_type = "explore"
            return {"type": "explore"}, "Early game -> gear/loot exploration", free_actions

        target_region = self._choose_move_target(intel)
        if target_region:
            self.last_action_type = "move"
            return {"type": "move", "regionId": target_region}, f"Move toward safer/value region {target_region[:8]}", free_actions

        self.last_action_type = "rest"
        return {"type": "rest"}, "No strong action -> rest", free_actions

    def _decide_free_actions(self, intel: Dict) -> List[Dict]:
        free = []
        if not intel["inventory_full"] and intel["local_items"]:
            for entry in intel["local_items"]:
                item = entry.get("item", {})
                if item.get("category") == "currency":
                    free.append({"type": "pickup", "itemId": item["id"]})
            if len(intel["inventory"]) < (10 - MIN_FREE_INVENTORY_SLOTS):
                best_entry = self.analyzer.get_best_item_on_ground(intel["local_items"], intel["inventory"])
                if best_entry:
                    item = best_entry.get("item", {})
                    if item.get("category") != "currency":
                        free.append({"type": "pickup", "itemId": item["id"]})
        best_weapon = self.analyzer.best_weapon_in_inventory(intel["inventory"])
        if best_weapon and self.analyzer.should_upgrade_weapon(intel["equipped_weapon"], best_weapon):
            free.append({"type": "equip", "itemId": best_weapon["id"]})
        return free

    def _evaluate_combat_targets(self, intel: Dict, targets: List[Dict], threshold: float, phase: str, zone_score: float, pvp_bias: float) -> Tuple[Optional[Dict], float, str]:
        my_stats = self._my_combat_stats(intel)
        best_target = None
        best_prob = 0.0
        best_score = -999.0
        for target in targets:
            enemy_stats = self._enemy_combat_stats(target)
            profile = self.memory.get_enemy_profile(target.get("id", "")) or {}
            model_prob = self.learning.predict_combat(my_stats, enemy_stats)
            tactical_prob = self.analyzer.estimate_win_probability(intel, target, profile)
            win_prob = (0.45 * model_prob) + (0.55 * tactical_prob)
            kill_confirm = 0.10 if target.get("hp", 100) <= 30 else 0.0
            isolated_bonus = 0.04 if len(targets) == 1 else -0.05 * max(0, len(targets) - 1)
            hp_penalty = max(0.0, (self.analyzer.hp_low - intel.get("hp", 100)) * 0.004)
            score = win_prob + kill_confirm + isolated_bonus + pvp_bias - hp_penalty - (zone_score * 0.08)
            score += self.analyzer.weapon_matchup_score(intel, target)
            if score > best_score:
                best_score = score
                best_prob = win_prob
                best_target = target
        if best_target is None:
            return None, 0.0, "No visible target"
        if best_target.get("hp", 100) <= 25 and intel.get("ep", 0) >= 2:
            return best_target, best_prob, f"Kill confirm on {best_target.get('name','?')}"
        if best_prob >= threshold:
            return best_target, best_prob, f"PvP engage {best_target.get('name','?')} prob={best_prob:.0%}"
        return None, best_prob, f"Best PvP prob={best_prob:.0%} below threshold={threshold:.0%}"

    def _evaluate_monster_targets(self, intel: Dict, monsters: List[Dict], phase: str, pve_bias: float) -> Tuple[Optional[Dict], float, str]:
        best = None
        best_prob = 0.0
        best_score = -999.0
        for monster in monsters:
            win_prob = self.analyzer.monster_win_probability(intel, monster)
            score = self.analyzer.score_monster_target(intel, monster) + pve_bias
            if phase == "early":
                score += 0.05
            if score > best_score:
                best_score = score
                best_prob = win_prob
                best = monster
        if best is not None and best_score >= MONSTER_FARM_SCORE_MIN:
            return best, best_prob, f"Farm {best.get('type','monster')} score={best_score:.2f}"
        return None, best_prob, "PvE not efficient"

    def _choose_move_target(self, intel: Dict) -> Optional[str]:
        connections = intel.get("connections") or []
        if not connections:
            return None
        pending_dz = set(str(x) for x in intel.get("pending_death_zones", []))
        all_dz = set(self.known_dz_regions) | pending_dz
        for rid, is_dz in intel.get("connections_status", {}).items():
            if is_dz:
                all_dz.add(rid)
        phase = self.analyzer.detect_game_phase(intel)
        terrain_scores = self.memory.weights.get("terrain_scores", {})

        def score_region(rid: str) -> float:
            score = 0.0
            if rid not in self.explored_regions:
                score += 2.0
            if rid in all_dz:
                score -= 100.0 * ZONE_ESCAPE_PRIORITY
            if phase == "mid":
                score += terrain_scores.get("hills", 0.7) * 0.1 + terrain_scores.get("ruins", 0.65) * 0.05
            elif phase == "late":
                score += terrain_scores.get("hills", 0.7) * 0.15
            if rid in self.dangerous_facilities:
                score -= 5.0
            return score

        safe = [rid for rid in connections if rid not in all_dz] or connections
        return max(safe, key=score_region)

    def _find_best_heal_item(self, inventory: List[Dict]) -> Optional[Dict]:
        heal_items = [i for i in inventory if i.get("category") == "recovery" and "energy" not in i.get("typeId", "").lower()]
        if not heal_items:
            return None
        priority = {"medkit": 3, "bandage": 2, "emergency_food": 1}
        return max(heal_items, key=lambda item: max((score for key, score in priority.items() if key in item.get("typeId", "").lower()), default=0))

    def _my_combat_stats(self, intel: Dict) -> Dict:
        weapon_bonus, weapon_range = self.analyzer.get_equipped_bonus(intel["equipped_weapon"])
        heal_stats = self.analyzer.inventory_heal_stats(intel.get("inventory", []))
        return {
            "hp": intel["hp"],
            "ep": intel["ep"],
            "atk": intel["atk"],
            "def": intel["def"],
            "weapon_bonus": weapon_bonus,
            "weapon_range": weapon_range,
            "heal_hp_total": heal_stats["heal_hp_total"],
            "heal_ep_total": heal_stats["heal_ep_total"],
            "heal_count": heal_stats["heal_count"],
            "best_heal_hp": heal_stats["best_heal_hp"],
            "effective_hp": intel["hp"] + heal_stats["heal_hp_total"],
            "inventory": intel.get("inventory", []),
        }

    def _enemy_combat_stats(self, target: Dict) -> Dict:
        weapon = target.get("equippedWeapon") or {}
        return {
            "hp": target.get("hp", 50),
            "atk": target.get("atk", 10),
            "def": target.get("def", 5),
            "weapon_bonus": weapon.get("atkBonus", 0),
        }

    def reset_for_new_game(self):
        self.turn_number = 0
        self.explored_regions = set()
        self.last_region_id = None
        self.stuck_counter = 0
        self.known_dz_regions = set()
        self.dangerous_facilities = set()
        self.last_action_type = ""
        logger.info("Strategy engine reset for new game")
