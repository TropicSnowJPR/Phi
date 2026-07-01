from typing import Any, Dict, List, Optional


class GameDataReader:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    # ------------------------
    # General
    # ------------------------

    @property
    def tick(self) -> int:
        return self.data["tick"]

    @property
    def timestamp(self) -> int:
        return self.data["timestamp"]

    # ------------------------
    # Player
    # ------------------------

    @property
    def player(self) -> Dict[str, Any]:
        return self.data["player"]

    @property
    def position(self) -> tuple[float, float, float]:
        p = self.player
        return p["x"], p["y"], p["z"]

    @property
    def rotation(self) -> tuple[float, float]:
        p = self.player
        return p["yaw"], p["pitch"]

    @property
    def health(self) -> float:
        return self.player["health"]

    @property
    def food(self) -> int:
        return self.player["food"]

    @property
    def armor(self) -> int:
        return self.player["armor"]

    @property
    def xp_level(self) -> int:
        return self.player["experienceLevel"]

    @property
    def xp_progress(self) -> float:
        return self.player["experienceProgress"]

    # ------------------------
    # World
    # ------------------------

    @property
    def world(self) -> Dict[str, Any]:
        return self.data["world"]

    @property
    def game_time(self) -> int:
        return self.world["gameTime"]

    @property
    def is_raining(self) -> bool:
        return self.world["raining"]

    @property
    def is_thundering(self) -> bool:
        return self.world["thundering"]

    @property
    def difficulty(self) -> str:
        return self.world["difficulty"]

    # ------------------------
    # Target Block
    # ------------------------

    @property
    def target_block(self) -> Optional[Dict[str, Any]]:
        return self.data.get("targetBlock")

    # ------------------------
    # Inventory
    # ------------------------

    @property
    def inventory(self) -> List[Dict[str, Any]]:
        return self.data["inventory"]

    def get_slot(self, slot: int) -> Optional[Dict[str, Any]]:
        for item in self.inventory:
            if item["slot"] == slot:
                return item
        return None

    def get_non_empty_inventory(self) -> List[Dict[str, Any]]:
        return [
            item
            for item in self.inventory
            if item["id"] != "minecraft:air" and item["count"] > 0
        ]

    # ------------------------
    # Entities
    # ------------------------

    @property
    def entities(self) -> List[Dict[str, Any]]:
        return self.data["entities"]

    def nearest_entity(self) -> Optional[Dict[str, Any]]:
        if not self.entities:
            return None
        return min(self.entities, key=lambda e: e["distance"])

    def entities_of_type(self, entity_type: str) -> List[Dict[str, Any]]:
        return [
            e
            for e in self.entities
            if e["type"] == entity_type
        ]

    # ------------------------
    # Blocks
    # ------------------------

    @property
    def blocks(self) -> List[Dict[str, Any]]:
        return self.data["blocks"]

    def get_block(self, x: int, y: int, z: int) -> Optional[Dict[str, Any]]:
        for block in self.blocks:
            if (
                block["x"] == x and
                block["y"] == y and
                block["z"] == z
            ):
                return block
        return None

    def blocks_of_type(self, block_id: str) -> List[Dict[str, Any]]:
        return [
            b
            for b in self.blocks
            if b["id"] == block_id
        ]