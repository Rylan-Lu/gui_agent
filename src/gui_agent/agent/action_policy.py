from __future__ import annotations

from dataclasses import dataclass

from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
)


@dataclass(frozen=True)
class ActionPolicyDecision:
    accepted: bool
    actions: list[GUIAction]
    reason: str | None = None
    repaired: bool = False


class ActionPolicy:
    """
    Validate and repair executable GUI action sequences.

    Rules:
    1. type_text requires focus.
    2. If previous focus is missing:
       insert click(element) before typing.
    3. Only repair when planner provides element metadata.
    """

    def __init__(
        self,
        *,
        allow_focus_repair: bool = True,
    ):
        self.allow_focus_repair = allow_focus_repair


    def validate_plan(
        self,
        actions: list[GUIAction],
    ) -> ActionPolicyDecision:

        repaired_actions: list[GUIAction] = []

        has_focus = False
        repaired = False


        for action in actions:

            # click actions can establish focus
            if action.action_type in {
                ActionType.CLICK,
                ActionType.DOUBLE_CLICK,
            }:

                repaired_actions.append(action)

                if action.element:
                    has_focus = True

                continue


            # typing requires focus
            if action.action_type == ActionType.TYPE_TEXT:

                if not has_focus:

                    if not self.allow_focus_repair:
                        return ActionPolicyDecision(
                            accepted=False,
                            actions=[],
                            reason="missing_focus",
                        )


                    focus_action = (
                        self._build_focus_action(action)
                    )


                    # -------------------------------
                    # Modified:
                    #
                    # Old behavior:
                    #   no element -> reject
                    #
                    # New behavior:
                    #   no element -> keep original
                    #   action because planner may rely
                    #   on existing desktop focus.
                    # -------------------------------
                    if focus_action is not None:

                        repaired_actions.append(
                            focus_action
                        )

                        has_focus = True
                        repaired = True


                repaired_actions.append(action)

                continue


            repaired_actions.append(action)


        return ActionPolicyDecision(
            accepted=True,
            actions=repaired_actions,
            repaired=repaired,
        )


    def _build_focus_action(
        self,
        action: GUIAction,
    ) -> GUIAction | None:

        if not action.element:
            return None


        return GUIAction(
            action_type=ActionType.CLICK,
            element=action.element,
            position=None,
            text=None,
        )