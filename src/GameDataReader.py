from pathlib import Path
import argparse
import json
import nbtlib


class GameDataReader:
    def __init__(self, world_path):
        self.world = Path(world_path)
        self.playerdata = self.world / 'playerdata'

    def list_player_files(self):
        if not self.playerdata.exists():
            raise FileNotFoundError(f"playerdata directory not found: {self.playerdata}")
        return sorted(self.playerdata.glob('*.dat'))

    def read_player_file(self, path: Path):
        nbt = nbtlib.load(path)
        root = nbt.root

        data = root.get('Data', root)


        health = float(data.get('Health', 0.0)) if 'Health' in data else float(data.get('health', 0.0))


        hunger = int(data.get('FoodLevel', data.get('foodLevel', data.get('food', 0))))


        inventory = []
        for item in data.get('Inventory', []):
            item_id = item.get('id', item.get('Name', 'unknown'))
            count = int(item.get('Count', 0))
            slot = int(item.get('Slot', -1))
            inventory.append({'id': str(item_id), 'count': count, 'slot': slot})

        return {
            'file': str(path),
            'health': health,
            'hunger': hunger,
            'inventory': inventory,
        }

    def read_all_players(self):
        for p in self.list_player_files():
            yield self.read_player_file(p)