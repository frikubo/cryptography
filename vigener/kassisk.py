import concurrent.futures as cf
from itertools import repeat

def thread_function(p, text, onlyNumber, results, id):
    for c in range(p + 1, len(text) - 3):
        if (text[p] == text[c]) and (text[p + 1] == text[c + 1]) and (text[p + 2] == text[c + 2]):
            if results[id] is None:
                results[id] = []

            results[id].append((c - p, text[p] + text[p + 1] + text[p + 2]))

def kassisk(texts, onlyNumbers = False):
    # pre kazdy text spusti rozbor
    results = [None] * len(texts)
    for t in range(len(texts)):
        with cf.ThreadPoolExecutor(max_workers=len(texts[t]) - 3) as executor:
            executor.map(thread_function, range(len(texts[t]) - 3), repeat(texts[t]), repeat(onlyNumbers), repeat(results), repeat(t))
            executor.shutdown(wait=True)

    return results