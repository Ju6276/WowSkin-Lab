"""Run an explicit test group; the default does not require devices or saved models."""
import argparse
from pathlib import Path
import unittest

GROUPS = ('unit', 'physics', 'official', 'artifacts', 'legacy')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', nargs='?', default='unit', choices=(*GROUPS, 'all'))
    parser.add_argument('--list', action='store_true', help='List checks without running them')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    groups = GROUPS if args.group == 'all' else (args.group,)
    suite = unittest.TestSuite()
    for group in groups:
        suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'/group),
                       pattern='test_*.py', top_level_dir=str(root)))
    if args.list:
        def show(items):
            for item in items:
                if isinstance(item, unittest.TestSuite): show(item)
                else: print(item.id())
        show(suite)
        return
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__': main()
