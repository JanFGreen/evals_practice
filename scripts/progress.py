"""Summarise exercise progress from a pytest JUnit XML file: passing tests per module, as Markdown."""

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def main(path: str) -> None:
    per_module: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for case in ET.parse(path).getroot().iter("testcase"):
        module = case.get("classname", "").split(".")[-1].removeprefix("test_")
        if module == "harness":
            continue
        failed = any(child.tag in ("failure", "error", "skipped") for child in case)
        per_module[module][0] += 0 if failed else 1
        per_module[module][1] += 1
    done = sum(p for p, _ in per_module.values())
    total = sum(t for _, t in per_module.values())
    print(f"## Exercise progress: {done}/{total} tests passing\n")
    print("| module | passing | status |\n|---|---|---|")
    for module, (passed, n) in sorted(per_module.items()):
        status = "done" if passed == n else "in progress" if passed else "not started"
        print(f"| {module} | {passed}/{n} | {status} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "progress.xml")
