"""Plancraft: one crafting example in text-only mode behind the Environment protocol."""
# Runs under data/plancraft/.venv/bin/python: standard library plus the simulator only.

from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

# The seed before every search, so a recipe query always returns the same grid.
SEARCH_SEED = 42


def load_example(split: str, example_id: str) -> Any:
    """The authors' ``PlancraftExample`` with this id, from the data the package ships."""
    import plancraft
    from plancraft.config import PlancraftExample

    path = Path(plancraft.__file__).parent / "data" / f"{split}.json"
    row = next((r for r in json.loads(path.read_text()) if r["id"] == example_id), None)
    if row is None:
        raise KeyError(f"no Plancraft example {example_id!r} in {path}")
    return PlancraftExample(**row)


def handlers() -> list[Any]:
    """move, smelt, seeded search and impossible, in the order the authors parse them."""
    from plancraft.environment import (
        GoldSearchActionHandler, ImpossibleActionHandler, MoveActionHandler, SmeltActionHandler,
    )

    class SeededSearch(GoldSearchActionHandler):
        def match(self, generated_text: str, **kwargs: Any) -> Any:
            random.seed(SEARCH_SEED)
            return super().match(generated_text, **kwargs)

    return [MoveActionHandler(), SmeltActionHandler(), SeededSearch(), ImpossibleActionHandler()]


class PlancraftTextEnvironment:
    """One example; the authors' ``check_done`` on the inventory is the whole score."""

    def __init__(self, example: Any) -> None:
        from plancraft.environment import PlancraftEnvironment

        self.example = example
        self.env = PlancraftEnvironment(inventory=deepcopy(example.slotted_inventory))
        self.handlers = handlers()
        self.success = False
        self.inventory: dict = {}
        self.reached_output_slot = False
        self.searches: list[str] = []
        self.give_up_reason: str | None = None

    def _observe(self, inventory: dict) -> str:
        """Track the inventory the agent sees; return its text."""
        self.inventory = inventory
        out = inventory.get(0) or {}
        self.reached_output_slot |= out.get("type") == self.example.target and out.get("quantity", 0) > 0
        return self._text(inventory)

    def _text(self, inventory: dict) -> str:
        from plancraft.environment.env import target_and_inventory_to_text_obs

        return target_and_inventory_to_text_obs(self.example.target, inventory)

    def _info(self, invalid: bool = False) -> dict[str, Any]:
        return {"success": self.success, "score": float(self.success), "failed": False,
                "invalid_command": invalid}

    def parse(self, command: str) -> Any:
        """The authors' ``parse_raw_model_response``: the first handler that matches wins."""
        for handler in self.handlers:
            out = handler.match(command)
            if out:
                if handler.action_name == "search" and not out.startswith("Format Error:"):
                    self.searches.append(re.search(r"search: *(\w+)", command).group(1))
                return out
        return f"Only select actions from the following: {', '.join(h.action_name for h in self.handlers)}"

    def reset(self) -> tuple[str, dict[str, Any]]:
        from plancraft.environment.env import get_objective_str

        obs = self.env.step()
        return self._observe(obs["inventory"]), {**self._info(), "task": get_objective_str(self.example.target)}

    def step(self, command: str) -> tuple[str, bool, dict[str, Any]]:
        from plancraft.environment.actions import StopAction
        from plancraft.simple import PlancraftGymWrapper

        action = self.parse(command)
        if isinstance(action, str):
            invalid = action.startswith(("Only select actions from the following:", "Format Error:"))
            return action, False, self._info(invalid=invalid)
        if isinstance(action, StopAction):
            # The server takes `impossible` as its end command, so only a string that hides
            # `impossible:` behind another word reaches here; the authors' loop stops on it.
            self.give_up_reason = action.reason
            return self._observe(self.env.step()["inventory"]), True, self._info()
        obs = self.env.step(action)
        self.success = bool(PlancraftGymWrapper.check_done(None, obs["inventory"], self.example.target))
        return self._observe(obs["inventory"]), self.success, self._info()

    def on_end_command(self, command: str) -> None:
        """The server's `impossible` stop: keep the reason the agent gave."""
        self.give_up_reason = command.partition(":")[2].strip()

    def close(self) -> None:
        pass

    def evaluation(self) -> dict[str, Any]:
        """What the episode reached beside the binary success; never shown to the agent or the wiki."""
        from plancraft.environment.env import convert_from_slot_index

        return {
            "target": self.example.target,
            "target_reached_output_slot": self.reached_output_slot,
            "final_inventory": [
                {"slot": convert_from_slot_index(slot), "item": item["type"], "quantity": item["quantity"]}
                for slot, item in sorted(self.inventory.items()) if item["quantity"] > 0],
            "searches": {"count": len(self.searches), "queries": self.searches},
            "give_up_reason": self.give_up_reason,
            "complexity_split": self.example.complexity_split,
        }


def make_env(spec: dict[str, Any]) -> PlancraftTextEnvironment:
    """``spec``: ``split`` (the package's data file) and ``example_id``."""
    return PlancraftTextEnvironment(load_example(str(spec["split"]), str(spec["example_id"])))
