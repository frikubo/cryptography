splitchar = '%'

def decode(text, key):
    original = ''
    realPos = 0
    for c in text:
        if ord(c) - 65 >= 0 and ord(c) - 65 < 26:
            original += chr((ord(c) - 65 + 26 - ord(key[realPos % len(key)]) + 65) % 26 + 65)
            realPos += 1
        else:
            original += c
    return original

def decodeAll(texts, keys, decoded = ''):
    for ki in range(len(keys)):
        decoded += decode(texts[ki], keys[ki])
        decoded += '\nKEY: ' + keys[ki] + '\n' + splitchar + '\n'
    return decoded
