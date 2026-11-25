from dataclasses import dataclass


@dataclass
class Grade:
    name: str
    passed: bool | None  # None = not applicable / could not be graded; never averaged in as a failure
    score: float | None = None
    detail: str = ""
    kind: str = "check"  # check | judge | metric | security | error
