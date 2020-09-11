# License: GNU General Public License v3.0

__version__ = "0.1.0"

def log_step(step_msg: str):
    no_astrisks = 50

    if len(step_msg) % 2:
        l = len(step_msg) // 2
        r = l
    else:
        l = len(step_msg) // 2
        r = l + 1

    print("*" * (no_astrisks - l) + step_msg + "*" * (no_astrisks - r))


def dummy_preprocessor(doc):
    return doc
